#!/usr/bin/env bash
# P2 35B-A3B shard0/shard1 自动看护脚本。
# 只负责一件事:GPU健康、且某个分片对应的GPU组空闲时,自动重新拉起该分片(checkpoint续跑,
# 不会重复算已完成的chunk)。不处理NVML故障本身(那个需要人工刷新实例)——
# 一旦实例刷新完、GPU恢复健康,这个脚本会在下一轮检查(至多CHECK_INTERVAL秒)内自动接上,
# 不需要人工手动重新执行拉起命令。
#
# 用法: nohup bash p2_supervisor.sh > /dev/null 2>&1 &
# 停止: pkill -f p2_supervisor.sh

cd /remote-home/ziyesong/videoPerception/eval || exit 1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lmmseval

LOG="/remote-home/ziyesong/videoPerception/eval/p2_supervisor.log"
CHECK_INTERVAL=60
TOTAL_CHUNKS=30
# 2026-07-21: GPU崩溃后重新提交,发现shard1原本靠SHARD/NUM_SHARDS取模分到的chunk已经
# 全部跑完(只剩videomme_p1/p3/p5这3个,原本就手动分给了gpu03_helper),再拉起modulo版
# shard1只会瞬间退出、白占检查周期。改成gpu47_helper,手动认领videomme_p5,和
# gpu03_helper(负责p1、p3)并行,GPU4-7不再跑shard1。
ENABLE_GPU47_HELPER=1

log() { echo "[$(date '+%F %T')] $1" >> "$LOG"; }

gpu_healthy() {
  nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1
}

gpu_group_busy() {
  local mem
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null)
  [ -n "$mem" ] && [ "$mem" -gt 1000 ]
}

completed_chunks() {
  command find results_p2_transformers_35b_35_chunks -name "*results.json" -size +0c 2>/dev/null | wc -l
}

# shard0(GPU0-3)原本分到的15个chunk已经全部跑完(2026-07-18),不能再靠SHARD/NUM_SHARDS
# 机制自动分到新工作(那是基于全局下标取模的静态分配)。改成手动认领的chunk
# (run_p2_gpu03_helper.sh里定的tomato_p1/p3、videomme_p1/p3;videomme_p5移交gpu47_helper),
# 这几个全部完成前才需要拉起GPU0-3;全部完成后就不用再管GPU0-3了。
gpu03_helper_done() {
  for chunk in tomato_p1 tomato_p3 videomme_p1 videomme_p3; do
    if ! command find "results_p2_transformers_35b_35_chunks/${chunk}" -name "*results.json" -size +0c 2>/dev/null | command grep -q .; then
      return 1
    fi
  done
  return 0
}

gpu47_helper_done() {
  command find "results_p2_transformers_35b_35_chunks/videomme_p5" -name "*results.json" -size +0c 2>/dev/null | command grep -q .
}

log "===== supervisor 启动 (PID $$) ====="

while true; do
  if ! gpu_healthy; then
    log "GPU/NVML不健康,等待人工刷新实例,本轮跳过"
    sleep "$CHECK_INTERVAL"
    continue
  fi

  n_done=$(completed_chunks)
  if [ "$n_done" -ge "$TOTAL_CHUNKS" ]; then
    log "全部${TOTAL_CHUNKS}个chunk已完成,supervisor退出"
    break
  fi

  if ! gpu03_helper_done && ! gpu_group_busy 0; then
    log "GPU0-3空闲且健康,拉起gpu03_helper (当前完成 $n_done/$TOTAL_CHUNKS)"
    port=$((29519 + RANDOM % 400))
    { GPUS=0,1,2,3 PORT=$port bash run_p2_gpu03_helper.sh
      log "gpu03_helper 本次运行退出"
    } >> "p2_supervisor_shard0.log" 2>&1 &
    sleep 30
  fi

  if [ "$ENABLE_GPU47_HELPER" -eq 1 ] && ! gpu47_helper_done && ! gpu_group_busy 4; then
    log "GPU4-7空闲且健康,拉起gpu47_helper (当前完成 $n_done/$TOTAL_CHUNKS)"
    port=$((29919 + RANDOM % 400))
    { GPUS=4,5,6,7 PORT=$port bash run_p2_gpu47_helper.sh
      log "gpu47_helper 本次运行退出"
    } >> "p2_supervisor_shard1.log" 2>&1 &
    sleep 30
  fi

  sleep "$CHECK_INTERVAL"
done
