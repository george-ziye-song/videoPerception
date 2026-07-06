#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark 数据完整性扫描:文件数对账 + 零字节/迷你文件 + 抽样解码 + TOMATO 双源 md5 交叉验证。"""
import os, glob, random, hashlib, json

random.seed(0)
R = {}

def tiny(files, thresh=20 * 1024):
    return [f for f in files if os.path.getsize(f) < thresh]

def sample_decode(files, n=40, tag=""):
    import decord
    bad = []
    for f in random.sample(files, min(n, len(files))):
        try:
            vr = decord.VideoReader(f)
            _ = vr[0].shape, vr[len(vr) - 1].shape
        except Exception as e:
            bad.append((os.path.basename(f), repr(e)[:80]))
    print(f"  [{tag}] 抽样解码 {min(n,len(files))} 个, 失败 {len(bad)}", flush=True)
    return bad

# ---------- 1. Video-MME ----------
print("== Video-MME ==", flush=True)
vm = glob.glob("/root/benchmarks/VideoMME/data/*.mp4")
subs = glob.glob("/root/benchmarks/VideoMME/subtitle/*")
R["videomme"] = dict(n_mp4=len(vm), expect=900, n_sub=len(subs),
                     tiny=[os.path.basename(f) for f in tiny(vm)],
                     decode_fail=sample_decode(vm, 40, "videomme"))

# ---------- 2. TOMATO (lmms-lab, 评测用) ----------
print("== TOMATO_lmms ==", flush=True)
tl = glob.glob("/root/benchmarks/TOMATO_lmms/videos/*/*.mp4")
R["tomato_lmms"] = dict(n_mp4=len(tl), expect=1417,
                        tiny=[os.path.basename(f) for f in tiny(tl)],
                        decode_fail=sample_decode(tl, 40, "tomato_lmms"))

# ---------- 3. TOMATO (用户下载版) 与 lmms 版逐文件 md5 交叉 ----------
print("== TOMATO 双源 md5 交叉 ==", flush=True)
def md5(f):
    h = hashlib.md5()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

yale_root = "/root/benchmarks/TOMATO/videos"
if os.path.isdir(yale_root):
    ty = glob.glob(yale_root + "/*/*.mp4")
    rel = lambda f, root: os.path.relpath(f, root)
    map_l = {rel(f, "/root/benchmarks/TOMATO_lmms/videos"): f for f in tl}
    map_y = {rel(f, yale_root): f for f in ty}
    only_l = sorted(set(map_l) - set(map_y)); only_y = sorted(set(map_y) - set(map_l))
    both = sorted(set(map_l) & set(map_y))
    mismatch = []
    for i, k in enumerate(both):
        if md5(map_l[k]) != md5(map_y[k]):
            mismatch.append(k)
        if (i + 1) % 300 == 0:
            print(f"  md5 进度 {i+1}/{len(both)}", flush=True)
    R["tomato_cross"] = dict(n_user=len(ty), common=len(both),
                             only_lmms=only_l[:10], only_user=only_y[:10],
                             md5_mismatch=mismatch)
else:
    R["tomato_cross"] = "用户版未就位, 跳过"

# ---------- 4. MVBench(按 lmms-eval DATA_LIST 逐目录对账) ----------
print("== MVBench ==", flush=True)
MV = "/root/hf_home/mvbench_video"
FOLDERS = {  # 目录: (期望下限, 用途)
    "star/Charades_segment": 585, "sta/sta_video_segment": 200,
    "clevrer/video_validation": 900, "perception/videos": 500,
    "nturgbd_convert": 200, "ssv2_video_mp4": 200,
    "tvqa/video_fps3_hq_segment": 200, "Moments_in_Time_Raw/videos": 200,
    "FunQA_test/test": 200, "scene_qa/video": 200, "vlnqa": 200,
}
mv_counts, mv_tiny, all_mv = {}, {}, []
for d, lo in FOLDERS.items():
    fs = [f for f in glob.glob(os.path.join(MV, d, "**", "*"), recursive=True) if os.path.isfile(f)]
    mv_counts[d] = dict(n=len(fs), expect_min=lo, ok=len(fs) >= lo)
    t = tiny(fs)
    if t:
        mv_tiny[d] = [os.path.basename(x) for x in t[:5]]
    all_mv += [f for f in fs if f.endswith((".mp4", ".webm", ".avi"))]
R["mvbench"] = dict(folders=mv_counts, tiny=mv_tiny,
                    decode_fail=sample_decode(all_mv, 60, "mvbench"))

out = "/remote-home/ziyesong/videoPerception/eval/scan_report.json"
json.dump(R, open(out, "w"), ensure_ascii=False, indent=1)
print("SCAN_DONE ->", out, flush=True)
