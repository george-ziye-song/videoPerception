#!/usr/bin/env bash
# probe-experiment.md §1.5:TRD 专属前置检验——把 MVBench/TOMATO 的帧序打乱,用 P1 同款
# 直答协议(thinking关闭)跑一遍,按子任务/reason_type记录"打乱前(已有P1基线) vs 打乱后"的
# 准确率差,筛出哪些子任务真的在考时序推理(掉分明显),值得再往下做§4的hidden state probe。
#
# 依赖:qwen_vl_utils/vision_process.py 的 _maybe_shuffle_frame_order 补丁(2026-07-11),
# 通过环境变量 SYNRL_SHUFFLE_FRAMES=1 开关,默认关闭不影响任何已有评测。
#
# 用法: MODEL_KEY=<anchor|9b|35b_35|35b_36> GPUS=0,1,2,3 bash run_frameshuffle_precheck.sh
#   anchor=Qwen3-VL-8B-Instruct(P1原生,无thinking机制)
#   35b_36=Qwen3.6-35B-A3B(已从P2范围移除,但§1.5明确要求用P1同款四模型对照,这里单独保留)
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lmmseval
cd /remote-home/ziyesong/videoPerception/eval
export HF_HOME=/root/hf_home
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy all_proxy 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
export SYNRL_SHUFFLE_FRAMES=1

case "$MODEL_KEY" in
  anchor)
    PRETRAINED=/root/models/Qwen3-VL-8B-Instruct
    MODEL_CLS=qwen3_vl
    EXTRA=""
    OUT=results_shuffle_anchor
    NPROC=4
    ;;
  9b)
    PRETRAINED=/root/models/Qwen3.5-9B
    MODEL_CLS=qwen3_5
    EXTRA=",enable_thinking=False"
    OUT=results_shuffle_9b
    NPROC=4
    ;;
  35b_35)
    PRETRAINED=/root/models/Qwen3.5-35B-A3B
    MODEL_CLS=qwen3_5
    EXTRA=",enable_thinking=False,device_map=auto"
    OUT=results_shuffle_35b_35
    NPROC=1
    ;;
  35b_36)
    PRETRAINED=/root/models/Qwen3.6-35B-A3B
    MODEL_CLS=qwen3_5
    EXTRA=",enable_thinking=False,device_map=auto"
    OUT=results_shuffle_35b_36
    NPROC=1
    ;;
  *)
    echo "MODEL_KEY must be one of: anchor|9b|35b_35|35b_36"; exit 1 ;;
esac

GPUS="${GPUS:?must set GPUS=0,1,2,3 or 4,5,6,7 or 0,1,2,3,4,5,6,7}"
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
      echo "ABORT: 等了5分钟 GPU $GPUS 还没就绪"; exit 1
    fi
    sleep 5
  done
}

# $1=task $2=nframes $3=tag
run_chunk () {
  local task="$1" nframes="$2" tag="$3"
  local chunkdir="${CHUNKROOT}/${tag}"
  # 非空校验(2026-07-11教训:空results.json会被误判成"已完成",见9B/35B P2那次事故)
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
  wait_for_gpu_ready
  sleep 5
  local leglog="${CHUNKROOT}_${tag}.leg.log"
  echo "===== $(date '+%F %T') START $MODEL_KEY/$tag(shuffled) ====="
  CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --num_processes $NPROC --main_process_port $PORT -m lmms_eval \
    --model ${MODEL_CLS} \
    --model_args "pretrained=${PRETRAINED},max_num_frames=${nframes}${EXTRA}" \
    --tasks "$task" --batch_size 1 \
    --log_samples --output_path "$chunkdir/" \
    > "$leglog" 2>&1
  local rc=$?
  tail -20 "$leglog"
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

run_chunk tomato 16 tomato_all

echo "SHUFFLE_${MODEL_KEY}_ALL_DONE $(date '+%F %T')"
