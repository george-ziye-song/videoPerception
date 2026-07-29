#!/usr/bin/env bash
# 处理"GPU数量变化"(同事临时要回4张卡,或者用完还回来)的续训交接。
#
# 前提:调用这个脚本前,当前正在跑的 auto_resume_grpo.sh/grpo.sh 已经被外部停掉了
# (Claude session 里用 TaskStop 做,这一步脚本自己做不到,不在这个文件的职责内)。
#
# 用法:
#   OLD_CKPT_PATH=<正在跑的那个segment的CKPT_PATH> NEW_N_GPUS=4 bash handoff_gpu_count.sh
#
# 做的事:
#   1. 读 OLD_CKPT_PATH 下的 latest_checkpointed_iteration.txt,确认最新完整step
#   2. 用 model_merger 把该step的FSDP分片合并成一份consolidated HF格式权重
#   3. 把这次交接记录追加到 progress_manifest.txt(因为新segment自己的step计数会从0开始,
#      跨segment的累计进度只能靠这份manifest手动加总)
#   4. 用合并出来的权重当新的 actor_rollout_ref.model.path,新的 CKPT_PATH,
#      新的 N_GPUS_PER_NODE=$NEW_N_GPUS,通过 auto_resume_grpo.sh 续跑
#
# 权重是warm-restart,不是bit-perfect resume:optimizer动量、LR schedule位置在每次
# 交接后重置,global_step计数器也是新segment自己从0开始——模型权重本身是精确保留的。

set -euo pipefail
cd "$(dirname "$0")"

OLD_CKPT_PATH="${OLD_CKPT_PATH:?must set OLD_CKPT_PATH (the segment being handed off)}"
NEW_N_GPUS="${NEW_N_GPUS:?must set NEW_N_GPUS (4 or 8, whatever is now available)}"
MANIFEST="${MANIFEST:-outputs/video_qa_grpo/progress_manifest.txt}"

TRACKER="${OLD_CKPT_PATH}/latest_checkpointed_iteration.txt"
if [ ! -f "$TRACKER" ]; then
  echo "HANDOFF: 找不到 $TRACKER,无法确认最新step,中止"
  exit 1
fi
LATEST_STEP=$(cat "$TRACKER")
echo "HANDOFF: ${OLD_CKPT_PATH} 最新完整step = ${LATEST_STEP}"

MERGE_SRC="${OLD_CKPT_PATH}/global_step_${LATEST_STEP}/actor"
MERGE_TARGET="${OLD_CKPT_PATH}/global_step_${LATEST_STEP}_merged_hf"

if [ ! -d "$MERGE_SRC" ]; then
  echo "HANDOFF: 找不到 $MERGE_SRC,中止"
  exit 1
fi

echo "HANDOFF: 合并 $MERGE_SRC -> $MERGE_TARGET"
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$MERGE_SRC" \
  --target_dir "$MERGE_TARGET"

mkdir -p "$(dirname "$MANIFEST")"
PREV_CUMULATIVE=0
if [ -f "$MANIFEST" ]; then
  PREV_CUMULATIVE=$(awk -F'|' 'END{gsub(/ /,"",$3); print ($3=="" ? 0 : $3)}' "$MANIFEST")
fi
CUMULATIVE=$((PREV_CUMULATIVE + LATEST_STEP))
echo "${OLD_CKPT_PATH} | steps_this_segment=${LATEST_STEP} | cumulative=${CUMULATIVE} | $(date '+%F %T') | handoff_to_${NEW_N_GPUS}gpu" >> "$MANIFEST"
echo "HANDOFF: 记录到 $MANIFEST (累计 ${CUMULATIVE} 步)"

NEW_CKPT_PATH="outputs/video_qa_grpo/segment_$(date +%Y%m%d_%H%M%S)_${NEW_N_GPUS}gpu"
echo "HANDOFF: 新segment CKPT_PATH = ${NEW_CKPT_PATH}, N_GPUS_PER_NODE=${NEW_N_GPUS}"
echo "HANDOFF: 用合并出的权重 ${MERGE_TARGET} 作为新的 actor model path 续跑"

ACTOR_MODEL_PATH="$MERGE_TARGET" \
N_GPUS_PER_NODE="$NEW_N_GPUS" \
CKPT_PATH="$NEW_CKPT_PATH" \
TRAIN_DATASET="${TRAIN_DATASET:-../datasets/video_qa_mc_train.parquet}" \
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-video_qa_mc}" \
TEST_FILE="${TEST_FILE:-[\"../datasets/video_qa_mc_val.parquet\"]}" \
RETURN_MULTI_MODAL_INPUTS=True \
REWARD_FUNCTION_PATH="verl/utils/reward_score/video_mc/__init__.py" \
REWARD_FUNCTION_NAME=reward_func \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:?must set MAX_PROMPT_LENGTH (measured value from the smoke test)}" \
MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-128}" \
MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-128}" \
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-16}" \
N_RESPONSES="${N_RESPONSES:-8}" \
MODEL_DTYPE=bfloat16 \
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}" \
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-16384}" \
SAVE_FREQ=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
OPTIMIZER_OFFLOAD=True \
DATA_SHUFFLE=True \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}" \
TEST_FREQ="${TEST_FREQ:-50}" \
bash auto_resume_grpo.sh
