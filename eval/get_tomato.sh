#!/usr/bin/env bash
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh; conda activate qwen3vl
DEST=/root/benchmarks/TOMATO_lmms
mkdir -p "$DEST"
echo "=== $(date '+%T') download lmms-lab/TOMATO (parquet + 3 video zips ~11.5G) ==="
env -u http_proxy -u https_proxy -u all_proxy HF_ENDPOINT=https://hf-mirror.com \
  hf download lmms-lab/TOMATO --repo-type dataset --local-dir "$DEST" 2>&1 | tail -3
echo "=== $(date '+%T') extract zips ==="
cd "$DEST"
python - <<'PY'
import zipfile, glob, os
for z in sorted(glob.glob('/root/benchmarks/TOMATO_lmms/part_*.zip')):
    with zipfile.ZipFile(z) as zf:
        zf.extractall('/root/benchmarks/TOMATO_lmms')
    print("extracted", os.path.basename(z), flush=True)
PY
echo "=== video count ==="
find /root/benchmarks/TOMATO_lmms/videos -name '*.mp4' 2>/dev/null | wc -l
touch /root/benchmarks/TOMATO_lmms/.tomato_ready
echo "TOMATO_READY $(date '+%T')"
