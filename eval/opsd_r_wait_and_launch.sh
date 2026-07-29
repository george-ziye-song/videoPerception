#!/usr/bin/env bash
# 先等6小时(给借用GPU4-7的人用),之后开始轮询GPU4-7是否空闲;空闲才启动OPSD-R
# 训练sweep(run_opsd_r_sweep.sh),被占用就继续等待,不放弃、不去抢占。
# 全程不碰GPU0-3(P2 shard0/gpu03_helper继续跑,不受这个脚本影响)。
#
# 用法: nohup bash opsd_r_wait_and_launch.sh > /dev/null 2>&1 &
# 停止: pkill -f opsd_r_wait_and_launch.sh

cd /remote-home/ziyesong/videoPerception/eval || exit 1

LOG="/remote-home/ziyesong/videoPerception/eval/opsd_r_wait_and_launch.log"
INITIAL_WAIT_SECONDS=$((6 * 3600))
HEARTBEAT_INTERVAL=1800   # 初始6小时等待期间,每30分钟打一次心跳日志
POLL_INTERVAL=300         # 6小时之后,每5分钟检查一次GPU4-7是否空闲

log() { echo "[$(date '+%F %T')] $1" >> "$LOG"; }

gpu_healthy() {
  nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1
}

gpu47_idle() {
  for gpu in 4 5 6 7; do
    local mem
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null)
    if [ -z "$mem" ] || [ "$mem" -gt 1000 ]; then
      return 1
    fi
  done
  return 0
}

log "===== opsd_r_wait_and_launch 启动 (PID $$) ====="
log "第一阶段:先等${INITIAL_WAIT_SECONDS}秒(6小时),不检查GPU,单纯让借用者先用"

start_ts=$(date +%s)
end_ts=$((start_ts + INITIAL_WAIT_SECONDS))
while [ "$(date +%s)" -lt "$end_ts" ]; do
  sleep "$HEARTBEAT_INTERVAL"
  now=$(date +%s)
  remaining=$((end_ts - now))
  [ "$remaining" -lt 0 ] && remaining=0
  log "心跳:仍在初始等待期,剩余约$((remaining / 60))分钟"
done

log "第二阶段:6小时等待结束,开始轮询GPU4-7是否空闲(每${POLL_INTERVAL}秒检查一次;如果被占用就继续等,不设上限)"

while true; do
  if ! gpu_healthy; then
    log "GPU/NVML当前不健康,等待人工刷新实例,本轮跳过"
    sleep "$POLL_INTERVAL"
    continue
  fi
  if gpu47_idle; then
    log "GPU4-7空闲,启动run_opsd_r_sweep.sh(前台阻塞直到整个sweep跑完)"
    bash run_opsd_r_sweep.sh
    log "run_opsd_r_sweep.sh已退出,wait_and_launch任务完成,退出"
    break
  else
    log "GPU4-7仍被占用,继续等待"
  fi
  sleep "$POLL_INTERVAL"
done
