#!/usr/bin/env bash
# shard0(GPU0-3)自己分到的15个chunk已经全部跑完,不能再指望SHARD/NUM_SHARDS机制自动
# 分到新工作(那是基于全局下标取模的静态分配,shard1还剩的都是奇数下标,SHARD=0永远碰不到)。
# 这个脚本手动认领shard1剩余队列里"离shard1当前进度最远"的几个chunk,直接复用
# run_p2_transformers_chunked.sh同一套run_chunk逻辑(resume-check+accelerate launch),
# 跑完之后shard1走到这几个chunk时,resume-check会自动发现已经做完直接跳过,不会重复算,
# 也不会和shard1并发抢同一个chunk(手动挑的是shard1短期内到不了的位置,留了安全余量)。
#
# 用法: GPUS=0,1,2,3 PORT=<port> bash run_p2_gpu03_helper.sh

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

GPUS="${GPUS:?must set GPUS=0,1,2,3}"
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

# $1=task_name(单个) $2=nframes $3=chunk_tag(目录名) $4=--offset值(可空) $5=--limit值(可空)
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
  echo "===== $(date '+%F %T') START $MODEL_KEY/$tag (task=$task nframes=$nframes off=$off lim=$lim) [gpu03_helper] ====="
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
  echo "===== $(date '+%F %T') DONE $MODEL_KEY/$tag (rc=$rc) [gpu03_helper] ====="
}

# 手动认领的chunk,是shard1剩余9个里"最后几个"(离shard1当前进度moving_attribute最远,
# shard1要先做完moving_direction/object_interaction/scene_transition/unexpected_action
# 这4个MVBench子任务才会碰到tomato_p1,留了充足的安全余量,不会和shard1并发抢同一个chunk)。
# 2026-07-21: videomme_p5移出这个脚本,改由run_p2_gpu47_helper.sh在GPU4-7上单独认领
# (shard1原本的modulo分配已经跑完,让GPU4-7并行分担最后几个chunk,不是留给这里串行做)。
run_chunk tomato 16 tomato_p1 371 371
run_chunk tomato 16 tomato_p3 1113 371
run_chunk videomme 32 videomme_p1 450 450
run_chunk videomme 32 videomme_p3 1350 450

echo "GPU03_HELPER_ALL_DONE $(date '+%F %T')"
