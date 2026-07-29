#!/usr/bin/env bash
# opsd-r-implementation-plan.md的训练扫描:GPU4-7各分到1个"读出层gap"原语,
# 每张卡按顺序跑一组(beta, projection)配置(§3.4的beta-sweep + §3.3的投影消融)。
# GPU4额外在跑完自己的原语后,顺带做§5.3的2个回归检查任务(只需要beta=0)。
#
# 范围取舍(如实说明,不是漏了):完整方案是4任务×4个beta×2种投影=32组,这里按
# "核心beta-sweep(4档,带投影)+挑1个beta做投影消融(不是3个都做)"来控制总时长,
# 每张卡5组配置,不是32组全测——如果核心beta-sweep显示投影有明显影响,再补测
# 其余beta值的投影消融。
#
# 用法: bash run_opsd_r_sweep.sh
# 断点续跑: 已有results.json的配置会被跳过,可以直接重新执行整个脚本。

set -uo pipefail
cd /remote-home/ziyesong/videoPerception/probe_data || exit 1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lmmseval

MODEL_PATH="/root/models/Qwen3.5-9B"
RUNS_DIR="/remote-home/ziyesong/videoPerception/probe_data/opsd_r_runs"
LOG="/remote-home/ziyesong/videoPerception/eval/opsd_r_sweep.log"

log() { echo "[$(date '+%F %T')] $1" >> "$LOG"; }

run_config() {
  local gpu=$1 task=$2 beta=$3 out_tag=$4 extra=$5
  local result_path="$RUNS_DIR/$task/$out_tag/results.json"
  if [ -f "$result_path" ]; then
    log "SKIP (已完成): task=$task out_tag=$out_tag"
    return 0
  fi
  log "START: gpu=$gpu task=$task beta=$beta out_tag=$out_tag extra='$extra'"
  python3 train_opsd_r.py "$MODEL_PATH" "$gpu" "$task" "$beta" "$out_tag" $extra \
    >> "opsd_r_${task}_${out_tag}.log" 2>&1
  if [ -f "$result_path" ]; then
    log "DONE: task=$task out_tag=$out_tag"
  else
    log "FAILED (无results.json): task=$task out_tag=$out_tag,详情见 opsd_r_${task}_${out_tag}.log"
  fi
}

run_task_sweep() {
  local gpu=$1 task=$2
  run_config "$gpu" "$task" 0    "beta0"        ""
  run_config "$gpu" "$task" 0.1  "beta0.1_proj" ""
  run_config "$gpu" "$task" 1    "beta1_proj"   ""
  run_config "$gpu" "$task" 10   "beta10_proj"  ""
  run_config "$gpu" "$task" 1    "beta1_noproj" "--no_projection"
}

log "===== opsd_r_sweep 启动 (PID $$) ====="

{ run_task_sweep 4 "Rotation_Direction"
  log "gpu4本职任务(Rotation_Direction)扫描结束,顺带做§5.3回归检查"
  run_config 4 "Complex_Direction_Identification" 0 "beta0" ""
  run_config 4 "Event_Sequence" 0 "beta0" ""
  log "gpu4全部工作结束"
} &
pid4=$!

{ run_task_sweep 5 "Rotation_Count"
  log "gpu5全部工作结束"
} &
pid5=$!

{ run_task_sweep 6 "Bouncing_Counting"
  log "gpu6全部工作结束"
} &
pid6=$!

{ run_task_sweep 7 "Acceleration_Identification"
  log "gpu7全部工作结束"
} &
pid7=$!

wait $pid4 $pid5 $pid6 $pid7
log "===== opsd_r_sweep 全部结束 ====="
