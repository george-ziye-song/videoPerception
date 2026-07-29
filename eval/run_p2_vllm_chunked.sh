#!/usr/bin/env bash
# P2(全 CoT,thinking-on)正式跑——vLLM 后端,**分片版**(2026-07-06)
# 用法: MODEL_KEY=<anchor|9b|35b_35|35b_36> GPUS=<0,1,2,3|4,5,6,7> bash run_p2_vllm_chunked.sh
#
# 起因:lmms-eval 的样本落盘(save_results_samples)是整个 task 跑完后一次性写(哪怕内部用
# "a"模式追加,也只在 simple_evaluate() 全部返回后才调用一次)——一个 benchmark 跑到一半才
# 崩(GPU/NVML/OOM/会话中断,已经发生过好几次),这一整条腿的数据全部丢失,之前记录的分数只
# 是"看起来完成"但实际 0 行 samples 的情况就是这么来的。
#
# 修法:把每个 benchmark 拆成小分片,每片独立一次 accelerate launch + 独立输出目录,跑完一片
# 落盘一片;某片崩了只丢那一片,其余片的结果已经在磁盘上,重跑时自动跳过已完成的分片。
#   - mvbench:天然就是 20 个子任务(各200题),直接按子任务分片,不用再切。
#   - tomato(1484条)/videomme(2700条):用 lmms-eval 原生支持的 --offset/--limit 按样本区间切,
#     每片 ~400-450 条。
# 全部分片跑完后,直接读所有分片的 samples_*.jsonl、用已经修复过的 parser 重新算总分
# (不依赖每个分片自己打印的局部 aggregate,那只是该分片内部的分数)。
#
# 依赖同 run_p2_vllm.sh 的三处本地 patch(nframes钳制/enable_thinking+reasoning_prompt/
# _strip_thinking回退)+ TOMATO/MVBench 提取器修复,GPU 就绪等待同 run_p2_vllm.sh。
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lmmseval
cd /remote-home/ziyesong/videoPerception/eval
export HF_HOME=/root/hf_home
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy all_proxy 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}

REASONING_PROMPT='\n\nFirst think through the visual evidence in the video step by step; reference specific moments or actions you observe. Wrap this reasoning in <think> and </think> tags. After the closing </think> tag give your final answer.'

GPU_MEM_UTIL=0.85
BATCH_SIZE=8
case "$MODEL_KEY" in
  anchor)
    PRETRAINED=/root/models/Qwen3-VL-8B-Instruct
    TP=4; DP=1
    EXTRA=",reasoning_prompt=${REASONING_PROMPT}"
    OUT=results_p2_vllm_anchor
    GPU_MEM_UTIL=0.70
    BATCH_SIZE=4
    ;;
  9b)
    PRETRAINED=/root/models/Qwen3.5-9B
    TP=4; DP=1
    EXTRA=""
    OUT=results_p2_vllm_9b
    ;;
  35b_35)
    PRETRAINED=/root/models/Qwen3.5-35B-A3B
    TP=4; DP=1
    EXTRA=""
    OUT=results_p2_vllm_35b_35
    ;;
  35b_36)
    PRETRAINED=/root/models/Qwen3.6-35B-A3B
    TP=4; DP=1
    EXTRA=""
    OUT=results_p2_vllm_35b_36
    ;;
  *)
    echo "MODEL_KEY must be one of: anchor|9b|35b_35|35b_36"; exit 1 ;;
esac

NPROC=$((TP * DP))
GPUS="${GPUS:?must set GPUS=0,1,2,3 or 4,5,6,7}"
PORT="${PORT:-29519}"
CHUNKROOT="${OUT}_chunks"
mkdir -p "$CHUNKROOT"

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

# $1=task_name(单个) $2=nframes $3=max_model_len $4=chunk_tag(目录名) $5=--offset值(可空) $6=--limit值(可空)
run_chunk () {
  local task="$1" nframes="$2" maxlen="$3" tag="$4" off="${5:-}" lim="${6:-}"
  local chunkdir="${CHUNKROOT}/${tag}"
  if find "$chunkdir" -name "*results.json" 2>/dev/null | head -1 | grep -q .; then
    echo "===== $MODEL_KEY/$tag 已有结果,跳过 ====="
    return 0
  fi
  local extra_args=""
  [ -n "$off" ] && extra_args="$extra_args --offset $off"
  [ -n "$lim" ] && extra_args="$extra_args --limit $lim"
  wait_for_gpu_ready
  sleep 10
  local leglog="${CHUNKROOT}_${tag}.leg.log"
  echo "===== $(date '+%F %T') START $MODEL_KEY/$tag (task=$task off=$off lim=$lim) ====="
  CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --num_processes $NPROC --main_process_port $PORT -m lmms_eval \
    --model vllm \
    --model_args "model=${PRETRAINED},tensor_parallel_size=${TP},data_parallel_size=${DP},enable_thinking=True,nframes=${nframes},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${maxlen}${EXTRA}" \
    --tasks "$task" --batch_size $BATCH_SIZE --gen_kwargs max_new_tokens=$MAX_NEW_TOKENS \
    $extra_args \
    --log_samples --output_path "$chunkdir/" \
    > "$leglog" 2>&1
  local rc=$?
  cat "$leglog"
  if [ $rc -ne 0 ] || grep -q "Error during evaluation\|ChildFailedError\|Traceback (most recent call last)" "$leglog"; then
    echo "===== $(date '+%F %T') ABORT $MODEL_KEY/$tag (rc=$rc,详见 $leglog) ====="
    exit 1
  fi
  echo "===== $(date '+%F %T') DONE $MODEL_KEY/$tag (rc=$rc) ====="
}

# --- MVBench: 20 个子任务天然分片 ---
MVBENCH_SUBTASKS="mvbench_action_antonym mvbench_action_count mvbench_action_localization mvbench_action_prediction mvbench_action_sequence mvbench_character_order mvbench_counterfactual_inference mvbench_egocentric_navigation mvbench_episodic_reasoning mvbench_fine_grained_action mvbench_fine_grained_pose mvbench_moving_attribute mvbench_moving_count mvbench_moving_direction mvbench_object_existence mvbench_object_interaction mvbench_object_shuffle mvbench_scene_transition mvbench_state_change mvbench_unexpected_action"
for sub in $MVBENCH_SUBTASKS; do
  run_chunk "$sub" 32 24576 "mvbench_${sub#mvbench_}"
done

# --- TOMATO: 1484 条,4 片(约370/片) ---
run_chunk tomato 16 16384 tomato_p0 0 371
run_chunk tomato 16 16384 tomato_p1 371 371
run_chunk tomato 16 16384 tomato_p2 742 371
run_chunk tomato 16 16384 tomato_p3 1113 371

# --- Video-MME: 2700 条,6 片(450/片) ---
run_chunk videomme 32 24576 videomme_p0 0 450
run_chunk videomme 32 24576 videomme_p1 450 450
run_chunk videomme 32 24576 videomme_p2 900 450
run_chunk videomme 32 24576 videomme_p3 1350 450
run_chunk videomme 32 24576 videomme_p4 1800 450
run_chunk videomme 32 24576 videomme_p5 2250 450

echo "P2_${MODEL_KEY}_CHUNKED_ALL_DONE $(date '+%F %T')"
