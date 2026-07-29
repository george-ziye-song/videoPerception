#!/usr/bin/env bash
# 自动重启循环:grpo.sh 因为显存碎片/边缘OOM偶发崩溃时,自动从最近checkpoint续训,
# 直到 trainer.total_training_steps(通过 TOTAL_STEPS 环境变量传入,默认500)达成为止。
# 用法: 和平时跑 grpo.sh 一样传环境变量,額外可设 MAX_RETRIES(默认1000,基本相当于不限)。
set -uo pipefail
cd "$(dirname "$0")"

CKPT_PATH="${CKPT_PATH:?must set CKPT_PATH}"
TOTAL_STEPS="${TOTAL_TRAINING_STEPS:-500}"
MAX_RETRIES="${MAX_RETRIES:-1000}"
TRACKER="${CKPT_PATH}/latest_checkpointed_iteration.txt"

attempt=0
while true; do
  attempt=$((attempt+1))
  if [ "$attempt" -gt "$MAX_RETRIES" ]; then
    echo "AUTO_RESUME: 达到最大重试次数 $MAX_RETRIES,停止"
    exit 1
  fi

  current_step=0
  if [ -f "$TRACKER" ]; then
    current_step=$(cat "$TRACKER")
  fi
  if [ "$current_step" -ge "$TOTAL_STEPS" ]; then
    echo "AUTO_RESUME: 已达到 $current_step/$TOTAL_STEPS 步,训练完成"
    exit 0
  fi

  echo "AUTO_RESUME: === 第 $attempt 次尝试,当前进度 $current_step/$TOTAL_STEPS ($(date '+%F %T')) ==="
  bash ../grpo.sh
  rc=$?
  echo "AUTO_RESUME: 本次尝试退出,rc=$rc ($(date '+%F %T'))"

  # 给 GPU/ray/NVML 一点缓冲时间再重启,避免背靠背抢占同一批GPU的竞态
  sleep 15
done
