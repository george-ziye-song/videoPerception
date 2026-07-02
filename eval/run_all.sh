#!/usr/bin/env bash
# Full base-model eval: 4-GPU data-parallel per benchmark, sequential benchmarks.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl
cd /remote-home/ziyesong/videoPerception/eval
export TOKENIZERS_PARALLELISM=false
export NUM_FRAMES=16
mkdir -p results
BS=4

for bench in mvbench temporalbench_short temporalbench_long; do
  echo "===== $(date '+%F %T') START $bench ====="
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g python run_eval.py --benchmark "$bench" \
        --num-shards 4 --shard "$g" --batch-size "$BS" \
        --out "results/${bench}.shard${g}.jsonl" \
        > "results/${bench}.shard${g}.log" 2>&1 &
  done
  wait
  echo "===== $(date '+%F %T') DONE $bench ====="
done
echo "ALL_EVAL_DONE $(date '+%F %T')"
