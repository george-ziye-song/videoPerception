#!/usr/bin/env bash
# 标准 harness 评测:lmms-eval 0.7.2 + Qwen3-VL-8B-Instruct,4×4090 数据并行
# 冒烟:LIMIT='--limit 8' bash run_lmmseval.sh   正式:bash run_lmmseval.sh
# 协议(写进报告):transformers 5.12.1 后端 / bf16 / sdpa / greedy(任务 yaml 内建)
#   mvbench 32帧(lmms-eval qwen3_vl 默认)| tomato 16帧(官方协议)| videomme 32帧(w/o sub)
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lmmseval
cd /remote-home/ziyesong/videoPerception/eval
export HF_HOME=/root/hf_home
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy all_proxy 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
MODEL=${MODEL:-/root/models/Qwen3-VL-8B-Instruct}
MODEL_CLS=${MODEL_CLS:-qwen3_vl}   # qwen3_vl(锚) / qwen3_5(Qwen3.5/3.6 系)
EXTRA=${EXTRA:-}                   # 追加 model_args,如 ',enable_thinking=False'(Qwen3.5/3.6 P1 必须)
OUT=${OUT:-results_lmmseval}
LIMIT="${LIMIT:-}"

# 启动守卫:任一 GPU 显存占用 >1G 即拒跑(防止和上一轮评测叠跑互挤 OOM)
for m in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); do
  if [ "$m" -gt 1000 ]; then
    echo "ABORT: GPU 显存已被占用 ${m}MiB,先清理旧进程再跑(nvidia-smi 查看)"; exit 1
  fi
done

run () { # $1=tasks  $2=附加 model_args(以逗号开头)
  echo "===== $(date '+%F %T') START $1 ====="
  accelerate launch --num_processes ${NPROC:-4} --main_process_port 29517 -m lmms_eval \
    --model ${MODEL_CLS} \
    --model_args pretrained=${MODEL}${2}${EXTRA} \
    --tasks "$1" --batch_size 1 --log_samples $LIMIT \
    --output_path $OUT/
  echo "===== $(date '+%F %T') DONE $1 (rc=$?) ====="
}

run mvbench  ",max_num_frames=32"
run tomato   ",max_num_frames=16"
run videomme ",max_num_frames=32"
echo "ALL_LMMSEVAL_DONE $(date '+%F %T')"
# OOM 预案:max_num_frames 32->16,或加 ,max_pixels=802816(约 256 tok/帧)
