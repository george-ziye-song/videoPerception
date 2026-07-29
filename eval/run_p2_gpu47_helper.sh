#!/usr/bin/env bash
# shard1(GPU4-7)原本按SHARD/NUM_SHARDS取模分到的那部分chunk已经全部跑完(2026-07-21确认,
# 30个chunk里只剩videomme_p1/p3/p5这3个,全部原本手动分给了gpu03_helper),再拉起
# 一次shard1只会立刻发现自己分到的chunk都做完了、瞬间退出、白占一次GPU健康检查周期
# ——不能再用modulo机制。这个脚本手动认领videomme_p5,和gpu03_helper(负责p1、p3,
# 顺序执行)并行跑,把最后3个chunk的总耗时从"GPU0-3串行3个"降到"两组并行,GPU0-3串行2个
# +GPU4-7单独1个",不需要为了3个chunk设计更复杂的动态抢单机制。
#
# 复用run_p2_gpu03_helper.sh同一套run_chunk/wait_for_gpu_ready逻辑,只是换了GPU/port/chunk。
#
# 用法: GPUS=4,5,6,7 PORT=<port> bash run_p2_gpu47_helper.sh

set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lmmseval
cd /remote-home/ziyesong/videoPerception/eval
export HF_HOME=/root/hf_home
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy all_proxy 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
SAMPLING_KWARGS="temperature=1.0,top_p=0.95,top_k=20,presence_penalty=1.5,repetition_penalty=1.0"
FORMAT_PROMPT='\n\nPlease show your choice in the `answer` field with only the choice letter e.g. `"answer": "C"`.'

MODEL_KEY=35b_35
PRETRAINED=/root/models/Qwen3.5-35B-A3B
MODEL_CLS=qwen3_5
EXTRA=",enable_thinking=True,device_map=auto,reasoning_prompt=${FORMAT_PROMPT}"
OUT=results_p2_transformers_35b_35
NPROC=1

GPUS="${GPUS:?must set GPUS=4,5,6,7}"
PORT="${PORT:-29919}"
CHUNKROOT="${OUT}_chunks"
mkdir -p "$CHUNKROOT"

wait_for_gpu_ready () {
  local tries=0
  while true; do
    if nvidia-smi --query-gpu=memory.used --format=csv,noheader >/tmp/gpu_ready_check47.$$ 2>/dev/null; then
      local dirty=0
      for g in $(echo "$GPUS" | tr ',' ' '); do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
        if [ -z "$m" ] || [ "$m" -gt 1000 ]; then dirty=1; fi
      done
      if [ "$dirty" -eq 0 ]; then rm -f /tmp/gpu_ready_check47.$$; return 0; fi
    fi
    rm -f /tmp/gpu_ready_check47.$$
    tries=$((tries+1))
    if [ "$tries" -gt 60 ]; then
      echo "ABORT: 等了5分钟 GPU $GPUS 还没就绪(NVML查不到或显存没释放干净)"; exit 1
    fi
    sleep 5
  done
}

run_chunk () {
  local task="$1" nframes="$2" tag="$3" off="${4:-}" lim="${5:-}"
  local chunkdir="${CHUNKROOT}/${tag}"
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
  rm -rf "$chunkdir"
  local extra_args=""
  [ -n "$off" ] && extra_args="$extra_args --offset $off"
  [ -n "$lim" ] && extra_args="$extra_args --limit $lim"
  wait_for_gpu_ready
  sleep 5
  local leglog="${CHUNKROOT}_${tag}.leg.log"
  echo "===== $(date '+%F %T') START $MODEL_KEY/$tag (task=$task nframes=$nframes off=$off lim=$lim) [gpu47_helper] ====="
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
  echo "===== $(date '+%F %T') DONE $MODEL_KEY/$tag (rc=$rc) [gpu47_helper] ====="
}

run_chunk videomme 32 videomme_p5 2250 450

echo "GPU47_HELPER_ALL_DONE $(date '+%F %T')"
