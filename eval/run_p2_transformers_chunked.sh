#!/usr/bin/env bash
# P2(全 CoT,thinking-on)——transformers 后端,分片版(2026-07-07)
# 用法: MODEL_KEY=<9b|35b_35> GPUS=<0,1,2,3|4,5,6,7|0,1,2,3,4,5,6,7> bash run_p2_transformers_chunked.sh
#   (anchor=Qwen3-VL-8B-Instruct、35b_36=Qwen3.6-35B-A3B 已从 P2 范围移除,见记忆/77report.md)
#   35B 系列专用并行:实测 device_map=auto 下 4 卡(12.3 tok/s)和 8 卡(11.8 tok/s)吞吐几乎一样
#   (只是模型并行,没有数据并行,多卡不增加吞吐)。所以 35B 应该拆成两个独立 4 卡实例并行跑
#   不同分片,而不是一个实例吃满 8 卡。用 SHARD/NUM_SHARDS 把分片列表按下标取模分给各实例:
#     终端A: MODEL_KEY=35b_35 GPUS=0,1,2,3 SHARD=0 NUM_SHARDS=2 bash run_p2_transformers_chunked.sh
#     终端B: MODEL_KEY=35b_35 GPUS=4,5,6,7 SHARD=1 NUM_SHARDS=2 bash run_p2_transformers_chunked.sh
#   anchor/9b 不需要这个(NPROC=4 已经用满传入的 GPUS 做数据并行),SHARD 默认 0/NUM_SHARDS 默认 1。
#
# 起因:实测发现同一模型、同样贪婪解码,vLLM 和 transformers 生成内容会明显分岔(社区已知、
# 尚未解决的问题,verl-project/verl#3392 对 Qwen3 系列有几乎一样的症状报告:vLLM 输出撞
# token 上限像是截断的,transformers 正常收尾)。根因是不同后端的 attention/GEMM kernel
# 实现导致浮点非结合律带来的数值差异,在贪婪解码的 argmax 决策点上可能被放大,一旦某一步
# 选择分岔,后续自回归生成整条轨迹都会跟着分岔——长 CoT 生成的决策点多,暴露得更明显。
# 用户决定:直接放弃 vLLM,P2 全部改用 transformers 后端重跑(包括已用 vLLM 跑完的锚模型/9B)。
#
# 依赖的本地补丁:
#   ① lmms_eval/models/simple/qwen3_vl.py _strip_thinking(): 2026-07-07 改成完全不剥离
#      (pass-through),原始 <think> 内容和最终答案一起原样保留在 filtered_resps 里
#      (用户要求:模型输出不能算完分就丢,组会要展示真实样例)。
#   ② lmms_eval/tasks/tomato/utils.py parse_multi_choice_response()、
#      lmms_eval/tasks/mvbench/utils.py mcq_acc(): 已换成共享的 extract_mcq_answer
#      (lmms_eval/tasks/_task_utils/mcq_extract.py),不是自己写的正则。
#   ③ --reasoning_tags none (2026-07-07 补,严重坑): lmms_eval/__main__.py 的 --reasoning_tags
#      默认值不是 None,是 '[["<think>", "</think>"], ["<analysis>", "</analysis>"]]'——哪怕
#      从没显式传这个参数,evaluator.py 也会在打分前默认剥离 <think>...</think>,和①的模型层
#      patch完全独立、发生在它之后,导致①白打了。harness 内部会把剥离前的原文存到
#      req.raw_filtered_resps,但 evaluation_tracker.py 根本不把这个字段写进 samples.jsonl,
#      所以之前所有跑出来的 filtered_resps 其实都是被默认剥离过的,不是真正的原始输出。
#      必须显式传 --reasoning_tags none 才能让 filtered_resps 保持原样。
#   ④ 采样参数(2026-07-08,重大发现): 之前全程 temperature=0(贪婪解码)。查 Qwen3.5 官方
#      README"Best Practices"发现:四种推荐配置(thinking通用/thinking编程/instruct通用/
#      instruct推理)都不用 temperature=0,thinking 通用任务推荐
#      temperature=1.0/top_p=0.95/top_k=20/presence_penalty=1.5/repetition_penalty=1.0,
#      且明确说 presence_penalty 就是用来"reduce endless repetitions"的——这正是我们花大量
#      精力排查的"卡住不收敛"现象。transformers.generate() 原生没有 presence_penalty(只有
#      乘性的 repetition_penalty),已在 simple/qwen3_vl.py 里加了一个
#      PresencePenaltyLogitsProcessor 自定义实现来对齐这个官方推荐值。
#   ⑤ 标准化输出格式(2026-07-08): README"Best Practices"第3条,MCQ 任务建议提示模型用
#      JSON 的 answer 字段回答(如 "answer": "C"),之前完全没做,已加进 REASONING_PROMPT。
#      配套在共享的 extract_mcq_answer 里加了 json_field 这个最高优先级的提取模式。
#   ⑥ max_new_tokens=4096(2026-07-08 消融实测): 在①②③④⑤都固定的前提下单独扫了
#      2048/4096/8192/32768,准确率分别是50.0/55.0/50.0/45.0%(n=40)——4096 是局部最优点,
#      不是"越大越好"也不是"越小越好"。32768 下无一样本真正撞满预算(max 31369),不是被截断
#      导致的假象。文献佐证:Anthropic《Inverse Scaling in Test-Time Compute》、
#      《Don't Overthink It》等已证明"更长推理不等于更准",视频推理任务尤其如此
#      (《Rethinking CoT Reasoning for Videos》)。
#
# 分片粒度同 run_p2_vllm_chunked.sh:
#   - mvbench:天然 20 个子任务(各200题)
#   - tomato(1484条)/videomme(2700条):--offset/--limit 切片
# 全部跑完后用 aggregate_chunks.py 重新算总分。
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lmmseval
cd /remote-home/ziyesong/videoPerception/eval
export HF_HOME=/root/hf_home
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy all_proxy 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
# 官方推荐采样参数(Qwen3.5 README "Best Practices" -> thinking mode for general tasks)
SAMPLING_KWARGS="temperature=1.0,top_p=0.95,top_k=20,presence_penalty=1.5,repetition_penalty=1.0"

FORMAT_PROMPT='\n\nPlease show your choice in the `answer` field with only the choice letter e.g. `"answer": "C"`.'

case "$MODEL_KEY" in
  9b)
    PRETRAINED=/root/models/Qwen3.5-9B
    MODEL_CLS=qwen3_5
    EXTRA=",enable_thinking=True,reasoning_prompt=${FORMAT_PROMPT}"
    OUT=results_p2_transformers_9b
    NPROC=4
    ;;
  35b_35)
    PRETRAINED=/root/models/Qwen3.5-35B-A3B
    MODEL_CLS=qwen3_5
    EXTRA=",enable_thinking=True,device_map=auto,reasoning_prompt=${FORMAT_PROMPT}"
    OUT=results_p2_transformers_35b_35
    NPROC=1   # device_map=auto 自动把权重分片到 GPUS 列出的所有卡上,单进程模型并行
    ;;
  *)
    echo "MODEL_KEY must be one of: 9b|35b_35 (anchor/35b_36 已从 P2 范围移除)"; exit 1 ;;
esac

GPUS="${GPUS:?must set GPUS=0,1,2,3 or 4,5,6,7 or 0,1,2,3,4,5,6,7}"
PORT="${PORT:-29519}"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
CHUNKROOT="${OUT}_chunks"
mkdir -p "$CHUNKROOT"
CHUNK_IDX=0

wait_for_gpu_ready () {
  local tries=0
  while true; do
    if nvidia-smi --query-gpu=memory.used --format=csv,noheader >/tmp/gpu_ready_check.$$ 2>/dev/null; then
      local dirty=0
      for g in $(echo "$GPUS" | tr ',' ' '); do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
        if [ -z "$m" ] || [ "$m" -gt 1000 ]; then dirty=1; fi
      done
      if [ "$dirty" -eq 0 ]; then rm -f /tmp/gpu_ready_check.$$; return 0; fi
    fi
    rm -f /tmp/gpu_ready_check.$$
    tries=$((tries+1))
    if [ "$tries" -gt 60 ]; then
      echo "ABORT: 等了5分钟 GPU $GPUS 还没就绪(NVML查不到或显存没释放干净)"; exit 1
    fi
    sleep 5
  done
}

# $1=task_name(单个) $2=nframes $3=chunk_tag(目录名) $4=--offset值(可空) $5=--limit值(可空)
run_chunk () {
  local task="$1" nframes="$2" tag="$3" off="${4:-}" lim="${5:-}"
  local my_idx=$CHUNK_IDX
  CHUNK_IDX=$((CHUNK_IDX+1))
  if [ $((my_idx % NUM_SHARDS)) -ne "$SHARD" ]; then
    return 0   # 这个分片归另一个并行实例处理
  fi
  local chunkdir="${CHUNKROOT}/${tag}"
  # local patch (2026-07-11,严重坑): 之前只检查 results.json 存不存在——磁盘写满崩溃那次,
  # lmms-eval 在真正写入内容之前就先创建了空文件(0字节),导致这个检测把"崩溃中途留下的空
  # 占位文件"误判成"已完成",让 tomato_p1(9B)、mvbench_action_localization(35B)这两个分片
  # 的数据从崩溃后就再也没被真正跑过,连续两次重启都被静默跳过,自己没有发现,是汇总时准确率
  # 分母对不上(1113/1484)才追出来的。现在必须要求 results.json 非空 且 对应的
  # samples_*.jsonl 也非空,两者都满足才算真正完成。
  if find "$chunkdir" -name "*results.json" -size +0c 2>/dev/null | head -1 | grep -q .; then
    local samples_ok=0
    for sf in "$chunkdir"/models__*/*samples_*.jsonl; do
      [ -s "$sf" ] && samples_ok=1 && break
    done
    if [ "$samples_ok" -eq 1 ]; then
      echo "===== $MODEL_KEY/$tag 已有结果,跳过 ====="
      return 0
    fi
  fi
  # 走到这里说明:要么完全没跑过,要么跑过但留下的是空文件(崩溃残留)——清掉重跑
  rm -rf "$chunkdir"
  local extra_args=""
  [ -n "$off" ] && extra_args="$extra_args --offset $off"
  [ -n "$lim" ] && extra_args="$extra_args --limit $lim"
  wait_for_gpu_ready
  sleep 5
  local leglog="${CHUNKROOT}_${tag}.leg.log"
  echo "===== $(date '+%F %T') START $MODEL_KEY/$tag (task=$task nframes=$nframes off=$off lim=$lim) ====="
  CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --num_processes $NPROC --main_process_port $PORT -m lmms_eval \
    --model ${MODEL_CLS} \
    --model_args "pretrained=${PRETRAINED},max_num_frames=${nframes}${EXTRA}" \
    --tasks "$task" --batch_size 1 --gen_kwargs max_new_tokens=${MAX_NEW_TOKENS},${SAMPLING_KWARGS} \
    --reasoning_tags none \
    $extra_args \
    --log_samples --output_path "$chunkdir/" \
    > "$leglog" 2>&1
  local rc=$?
  tail -30 "$leglog"
  if [ $rc -ne 0 ] || grep -q "Error during evaluation\|ChildFailedError\|Traceback (most recent call last)" "$leglog"; then
    echo "===== $(date '+%F %T') ABORT $MODEL_KEY/$tag (rc=$rc,详见 $leglog) ====="
    exit 1
  fi
  echo "===== $(date '+%F %T') DONE $MODEL_KEY/$tag (rc=$rc) ====="
}

MVBENCH_SUBTASKS="mvbench_action_antonym mvbench_action_count mvbench_action_localization mvbench_action_prediction mvbench_action_sequence mvbench_character_order mvbench_counterfactual_inference mvbench_egocentric_navigation mvbench_episodic_reasoning mvbench_fine_grained_action mvbench_fine_grained_pose mvbench_moving_attribute mvbench_moving_count mvbench_moving_direction mvbench_object_existence mvbench_object_interaction mvbench_object_shuffle mvbench_scene_transition mvbench_state_change mvbench_unexpected_action"
for sub in $MVBENCH_SUBTASKS; do
  run_chunk "$sub" 32 "mvbench_${sub#mvbench_}"
done

run_chunk tomato 16 tomato_p0 0 371
run_chunk tomato 16 tomato_p1 371 371
run_chunk tomato 16 tomato_p2 742 371
run_chunk tomato 16 tomato_p3 1113 371

run_chunk videomme 32 videomme_p0 0 450
run_chunk videomme 32 videomme_p1 450 450
run_chunk videomme 32 videomme_p2 900 450
run_chunk videomme 32 videomme_p3 1350 450
run_chunk videomme 32 videomme_p4 1800 450
run_chunk videomme 32 videomme_p5 2250 450

echo "P2_${MODEL_KEY}_TRANSFORMERS_CHUNKED_ALL_DONE $(date '+%F %T')"
