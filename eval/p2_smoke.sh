#!/usr/bin/env bash
# P2 (全 CoT) 协议验证冒烟脚本 —— 只跑 --limit 8,验证 <think> 剥离 + 答案抽取链路是否可靠
# 用法: bash p2_smoke.sh   (GPU 0-3, port 29519, 不与 GPU4-7 上的 P1 收尾任务冲突)
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lmmseval
cd /remote-home/ziyesong/videoPerception/eval
export HF_HOME=/root/hf_home
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy all_proxy 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 锚模型无原生 thinking,用模型类自带的 reasoning_prompt(追加到 user context 末尾,见
# lmms_eval/models/simple/qwen3_vl.py:300)引导它自己产出 <think></think> + 最终答案格式。
# 注意:model_args 用逗号分隔 key=value,所以这段文字里不能有逗号(否则被切碎);用句号/分号断句。
REASONING_PROMPT='\n\nFirst think through the visual evidence in the video step by step; reference specific moments or actions you observe. Wrap this reasoning in <think> and </think> tags. After the closing </think> tag give your final answer.'

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}

run () { # $1=model_cls $2=pretrained $3=extra_model_args(以逗号开头) $4=tasks $5=nframes
  echo "===== $(date '+%F %T') START $1/$4 (max_new_tokens=$MAX_NEW_TOKENS) ====="
  accelerate launch --num_processes 4 --main_process_port 29519 -m lmms_eval \
    --model "$1" \
    --model_args "pretrained=$2,max_num_frames=$5$3" \
    --tasks "$4" --batch_size 1 --gen_kwargs max_new_tokens=$MAX_NEW_TOKENS \
    --log_samples --limit 8 --output_path results_p2_smoke2/
  echo "===== $(date '+%F %T') DONE $1/$4 (rc=$?) ====="
}

# --- Qwen3.5-9B, enable_thinking=True(原生) ---
run qwen3_5 /root/models/Qwen3.5-9B ",enable_thinking=True" mvbench 32
run qwen3_5 /root/models/Qwen3.5-9B ",enable_thinking=True" tomato 16
run qwen3_5 /root/models/Qwen3.5-9B ",enable_thinking=True" videomme 32

# --- 锚 Qwen3-VL-8B,无原生 thinking,走 reasoning_prompt(model 类原生支持,非 system_prompt 硬凑)---
run qwen3_vl /remote-home/ziyesong/models/Qwen3-VL-8B-Instruct ",enable_thinking=True,reasoning_prompt=${REASONING_PROMPT}" mvbench 32
run qwen3_vl /remote-home/ziyesong/models/Qwen3-VL-8B-Instruct ",enable_thinking=True,reasoning_prompt=${REASONING_PROMPT}" tomato 16
run qwen3_vl /remote-home/ziyesong/models/Qwen3-VL-8B-Instruct ",enable_thinking=True,reasoning_prompt=${REASONING_PROMPT}" videomme 32

echo "P2_SMOKE_ALL_DONE $(date '+%F %T')"
