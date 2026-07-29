#!/usr/bin/env bash
# P2(全 CoT,thinking-on)正式跑——vLLM 后端
# 用法: MODEL_KEY=<anchor|9b|35b_35|35b_36> GPUS=<0,1,2,3|4,5,6,7> bash run_p2_vllm.sh
#   冒烟: LIMIT='--limit 8' MODEL_KEY=... GPUS=... bash run_p2_vllm.sh
#
# 2026-07-05 事故记录:第一版用 TP2/DP2(9B/锚模型)在满载下反复撞
#   `has_unfinished_requests_dp` -> gloo all_reduce -> "Connection closed by peer",
#   外加 rc=$? 检测失灵(accelerate launch 在子进程崩溃后仍返回 0,脚本误判"DONE"继续跑
#   下一个任务),导致整轮只有 9B/tomato 一条腿是真数据,其余全部静默失败。
#   修复:①全部改 TP=4/DP=1(不再用 vLLM 的 data_parallel_size 分组,规避这条 gloo 同步路径,
#   35B 冒烟阶段这个配置从没崩过)②不再信任 rc=$?,改为 grep 每条腿自己的日志找
#   "Error during evaluation"/Traceback,任何一条腿失败就中止整个脚本,不静默前进。
#
# 依赖的本地补丁(均已打在 lmmseval conda env 里,GPFS 备份需刷新):
#   ① lmms_eval/models/simple/qwen3_vl.py _strip_thinking(): </think> 缺失时不再清空为空串(回退原文)
#   ② lmms_eval/models/chat/vllm.py: 补 enable_thinking(转发 chat_template_kwargs)+ reasoning_prompt
#      (追加到最后一条 user message)——上游这个通用类原本两个都不支持。
#   ③ lmms_eval/protocol.py ChatMessages.to_openai_messages(): 视频帧数不足时钳 nframes(同 qwen3_vl 那版
#      补丁的逻辑),此前只在 qwen3_vl 专用模型类里打过,vllm/vllm_chat 走的是这条通用路径,没覆盖到。
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lmmseval
cd /remote-home/ziyesong/videoPerception/eval
export HF_HOME=/root/hf_home
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy all_proxy 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
LIMIT="${LIMIT:-}"
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
    # 2026-07-06 事故记录:Qwen3-VL(deepstack架构)在 vLLM 下 gpu_memory_utilization=0.85 +
    # batch_size=8 会先 CUDA OOM(vLLM自己就把GPU吃到22.95/23.64G)再触发下游
    # "Error during masked scatter operation"(多模态embedding merge失败,是OOM的下游症状,
    # 不是vLLM #31679那个async_scheduling race——试过关async_scheduling没用)。
    # 帧数(32帧mvbench/videomme、16帧tomato)没有动,协议不变,只降计算侧的显存预留/并发度。
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

# 启动守卫(2026-07-06 改成每条腿开始前都查,不只脚本启动时查一次):
# 连续两次 accelerate launch 抢占同一组 4 卡背靠背起(上一条腿刚 TP=4 释放,下一条紧接着
# 初始化)会撞 NVML/CUDA 设备探测的竞态(`RuntimeError: Device string must not be empty`,
# nvidia-smi 同一时刻也会报 `Failed to initialize NVML`)——已实测复现3次(锚模型
# mvbench→tomato、9B 的 tomato→videomme、9B→35b_35 切换时)。等 nvidia-smi 能正常查询
# 且显存干净,再留一点缓冲时间让驱动层彻底收尾。
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

run () { # $1=tasks $2=nframes $3=max_model_len $4=完成检测用的metric key子串(断点续跑)
  if [ -n "${4:-}" ] && grep -rl "$4" "$OUT"/models__*/*_results.json >/dev/null 2>&1; then
    echo "===== $MODEL_KEY/$1 已有结果(检测到 $4),跳过 ====="
    return 0
  fi
  wait_for_gpu_ready
  sleep 10  # 额外缓冲:让上一条腿的 vLLM/NCCL 资源彻底收尾,不要背靠背起新引擎
  local leglog="${OUT}_${1}.leg.log"
  echo "===== $(date '+%F %T') START $MODEL_KEY/$1 (TP=$TP DP=$DP max_new_tokens=$MAX_NEW_TOKENS) ====="
  CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --num_processes $NPROC --main_process_port $PORT -m lmms_eval \
    --model vllm \
    --model_args "model=${PRETRAINED},tensor_parallel_size=${TP},data_parallel_size=${DP},enable_thinking=True,nframes=$2,gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=$3${EXTRA}" \
    --tasks "$1" --batch_size $BATCH_SIZE --gen_kwargs max_new_tokens=$MAX_NEW_TOKENS \
    --log_samples $LIMIT --output_path $OUT/ \
    > "$leglog" 2>&1
  local rc=$?
  cat "$leglog"
  if [ $rc -ne 0 ] || grep -q "Error during evaluation\|ChildFailedError\|Traceback (most recent call last)" "$leglog"; then
    echo "===== $(date '+%F %T') ABORT $MODEL_KEY/$1 (rc=$rc, 日志里检测到报错,详见 $leglog) ====="
    exit 1
  fi
  echo "===== $(date '+%F %T') DONE $MODEL_KEY/$1 (rc=$rc, 已核实日志无报错) ====="
}

run mvbench 32 24576 mvbench_accuracy
run tomato 16 16384 tomato_score
run videomme 32 24576 videomme_perception_score
echo "P2_${MODEL_KEY}_ALL_DONE $(date '+%F %T')"
