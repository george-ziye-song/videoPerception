#!/usr/bin/env bash
# 自动重启循环:low_rank_rep_distillation.sh(on_policy_distillation.sh)因为显存碎片/边缘OOM/
# 共享服务器被抢占等原因偶发崩溃时,自动从最近checkpoint续训,直到
# trainer.total_training_steps(通过 TOTAL_TRAINING_STEPS 环境变量传入)达成为止。
# 依赖 on_policy_distillation.sh 的 CKPT_PATH=${CKPT_PATH:-...} 改动(2026-07-27)——
# 必须在调用前显式导出同一个 CKPT_PATH,否则每次重启都会生成新的时间戳目录,
# resume_mode=auto 永远找不到上一次的 checkpoint。
# 用法:和平时跑 low_rank_rep_distillation.sh 一样传环境变量(必须包含 CKPT_PATH 和
# TOTAL_TRAINING_STEPS),額外可设 MAX_RETRIES(默认1000,基本相当于不限)。
set -uo pipefail
cd "$(dirname "$0")"

CKPT_PATH="${CKPT_PATH:?must set CKPT_PATH}"
TOTAL_STEPS="${TOTAL_TRAINING_STEPS:?must set TOTAL_TRAINING_STEPS}"
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
  bash ../low_rank_rep_distillation.sh
  rc=$?
  echo "AUTO_RESUME: 本次尝试退出,rc=$rc ($(date '+%F %T'))"

  # 给 GPU/ray/NVML 一点缓冲时间再重启,避免背靠背抢占同一批GPU的竞态
  sleep 15
done
