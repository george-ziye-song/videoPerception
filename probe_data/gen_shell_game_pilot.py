#!/usr/bin/env python3
"""distillation-readiness-experiments.md 实验2试点:生成basic_shell_game数据。
只跑 03_shell_game.py 的 basic_shell_game 类型(单球追踪,状态最干净),不走main()里
全部12种challenge_type的循环。CONFIG路径改到 probe_data/ShellGame/ 下。

用法: python3 gen_shell_game_pilot.py [n_samples]
"""
import sys
import os
import json
import importlib.util
import concurrent.futures
from tqdm import tqdm

REPO_DIR = "/remote-home/ziyesong/videoPerception/repos/Synthetic-Video"
OUT_DIR = "/remote-home/ziyesong/videoPerception/probe_data/ShellGame"
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def load_module():
    path = os.path.join(REPO_DIR, "03_shell_game.py")
    spec = importlib.util.spec_from_file_location("shell_game_pilot", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shell_game_pilot"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_module()

    # 修正CONFIG路径(原脚本硬编码了不存在的路径)
    mod.BASE_OUTPUT_DIR = OUT_DIR + "/"
    mod.OUTPUT_VIDEO_DIR = os.path.join(OUT_DIR, "videos")
    mod.SFT_OUTPUT_FILE = os.path.join(OUT_DIR, "sft.jsonl")
    mod.RL_OUTPUT_FILE = os.path.join(OUT_DIR, "rl.jsonl")
    mod.METADATA_OUTPUT_FILE = os.path.join(OUT_DIR, "metadata.jsonl")
    mod.VIDEO_PREFIX_IN_JSON = os.path.join(OUT_DIR, "videos") + "/"
    os.makedirs(mod.OUTPUT_VIDEO_DIR, exist_ok=True)

    # 修正ffmpeg路径(原脚本硬编码的路径在本机不存在,已用conda装了ffmpeg)
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    mod.plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    print(f"ffmpeg: {ffmpeg_path}")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(mod.process_single_challenge, i, "basic_shell_game"): i for i in range(N_SAMPLES)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=N_SAMPLES, desc="Generating basic_shell_game"):
            results.append(future.result())

    sft_recs, rl_recs, meta_recs, failed = [], [], [], []
    for res in results:
        if isinstance(res, tuple) and len(res) == 2:
            failed.append(res)
        elif res is not None and len(res) >= 3:
            sft_recs.append(res[0]); rl_recs.append(res[1]); meta_recs.append(res[2])
        else:
            failed.append((-1, f"incomplete: {res}"))

    for path, data in [(mod.SFT_OUTPUT_FILE, sft_recs), (mod.RL_OUTPUT_FILE, rl_recs), (mod.METADATA_OUTPUT_FILE, meta_recs)]:
        with open(path, "w") as f:
            for rec in data:
                f.write(json.dumps(rec) + "\n")

    print(f"\n成功生成 {len(sft_recs)}/{N_SAMPLES}")
    if failed:
        print(f"失败 {len(failed)} 条:")
        for tid, err in failed[:5]:
            print(f"  - {tid}: {err[:300]}")


if __name__ == "__main__":
    main()
