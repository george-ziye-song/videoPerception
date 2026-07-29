#!/usr/bin/env bash
# 35B-A3B P2评估还没跑完(shard1原本在GPU4-7上,之前为了给别人腾GPU暂停了)。
# 先等6小时(留给借用GPU4-7的人用),之后把p2_supervisor.sh的ENABLE_SHARD1改回1并
# 重启supervisor——supervisor自己的主循环本来就会在每轮检查GPU4-7是否空闲
# (gpu_group_busy 4),空闲才拉起shard1,被占用就继续等、不会抢占,不需要在这个
# 脚本里重新实现一遍轮询逻辑。
#
# 注意:ENABLE_SHARD1是supervisor启动时读入内存的bash变量,只改文件不重启进程的话
# 旧supervisor不会生效(之前踩过这个坑)——所以这里改完文件必须kill旧进程再重启。
#
# 用法: nohup bash p2_shard1_delayed_resume.sh > /dev/null 2>&1 &

cd /remote-home/ziyesong/videoPerception/eval || exit 1

LOG="/remote-home/ziyesong/videoPerception/eval/p2_shard1_delayed_resume.log"
WAIT_SECONDS=$((6 * 3600))
HEARTBEAT_INTERVAL=1800

log() { echo "[$(date '+%F %T')] $1" >> "$LOG"; }

log "===== p2_shard1_delayed_resume 启动 (PID $$),先等${WAIT_SECONDS}秒(6小时) ====="

start_ts=$(date +%s)
end_ts=$((start_ts + WAIT_SECONDS))
while [ "$(date +%s)" -lt "$end_ts" ]; do
  sleep "$HEARTBEAT_INTERVAL"
  now=$(date +%s)
  remaining=$((end_ts - now))
  [ "$remaining" -lt 0 ] && remaining=0
  log "心跳:仍在等待期,剩余约$((remaining / 60))分钟"
done

log "6小时等待结束,把ENABLE_SHARD1改回1并重启supervisor(之后GPU4-7占用与否由supervisor自己的循环处理,占用就继续等)"

sed -i 's/^ENABLE_SHARD1=0$/ENABLE_SHARD1=1/' p2_supervisor.sh
sed -i "s/^# 2026-07-19:.*GPU4-7.*$/# 2026-07-19: p2_shard1_delayed_resume.sh 6小时等待结束,ENABLE_SHARD1重新打开。/" p2_supervisor.sh

if ! command grep -q "^ENABLE_SHARD1=1$" p2_supervisor.sh; then
  log "ERROR: sed替换ENABLE_SHARD1失败,当前内容:$(command grep '^ENABLE_SHARD1=' p2_supervisor.sh),需要人工检查"
  exit 1
fi

old_pids=$(pgrep -f "bash p2_supervisor.sh" || true)
if [ -n "$old_pids" ]; then
  log "kill旧supervisor进程: $old_pids"
  kill -9 $old_pids
  sleep 2
fi

nohup bash p2_supervisor.sh > /dev/null 2>&1 &
new_pid=$!
sleep 3
if pgrep -f "bash p2_supervisor.sh" > /dev/null; then
  log "新supervisor已启动(约PID $new_pid),ENABLE_SHARD1=1,后续由它自己的循环处理GPU4-7空闲检测与拉起。任务完成,退出。"
else
  log "ERROR: 新supervisor没有成功启动,需要人工检查"
  exit 1
fi
