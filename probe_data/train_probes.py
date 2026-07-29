#!/usr/bin/env python3
"""probe-experiment.md §4.2:对每个(model, task_type, layer)训一个线性probe,
用metadata.jsonl里的数值/类别GT(不是字母答案)当监督目标,故意用简单的线性分类器,
避免probe自己学会推理。

除Event_Sequence外,每条样本pooled成一个特征向量(mean over temporal groups),
每类原语一个多分类任务。Event_Sequence按§4.2的建议做"该temporal group是否落在
关键事件窗口内"的二分类,标签来自video_events_timeline_ms(用采样帧的均匀时间戳
近似还原每个group的时间中心,和qwen_vl_utils实际采样帧的方式一致)。

用法: python3 train_probes.py <hidden_states_dir> <model_tag>
"""
import sys
import os
import json
import glob
import collections
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

PROBE_DATA_DIR = "/remote-home/ziyesong/videoPerception/probe_data"
LAYERS = ["shallow", "mid", "deep"]

TASK_SUBDIR = {
    "Complex_Direction_Identification": "Atomic",
    "Rotation_Direction": "Atomic",
    "Rotation_Count": "Atomic",
    "Bouncing_Counting": "Atomic",
    "Directional_Event_Counting": "Atomic",
    "Acceleration_Identification": "Atomic",
    "Event_Sequence": "Atomic2",
}


def label_direction_path(rec):
    return "_".join(rec["ground_truth_details"]["other_details"]["path"])


def label_rotation_direction(rec):
    return rec["ground_truth_details"]["other_details"]["direction"]


def label_rotation_count(rec):
    return str(rec["ground_truth_details"]["other_details"]["total_rotations"])


def label_bouncing_count(rec):
    return str(rec["ground_truth_details"]["other_details"]["target_bounce_count"])


def label_directional_event_count(rec):
    od = rec["ground_truth_details"]["other_details"]
    return str(sum(1 for x in od["sequence"] if x == od["target_direction"]))


def label_acceleration(rec):
    return rec["ground_truth_details"]["other_details"]["motion_type"]


WHOLE_VIDEO_LABEL_FNS = {
    "Complex_Direction_Identification": label_direction_path,
    "Rotation_Direction": label_rotation_direction,
    "Rotation_Count": label_rotation_count,
    "Bouncing_Counting": label_bouncing_count,
    "Directional_Event_Counting": label_directional_event_count,
    "Acceleration_Identification": label_acceleration,
}


def video_duration_s(task, doc_id):
    subdir = TASK_SUBDIR[task]
    path = os.path.join(PROBE_DATA_DIR, subdir, "videos", f"{task}_{doc_id:04d}.mp4")
    import decord
    vr = decord.VideoReader(path)
    return len(vr) / vr.get_avg_fps()


def train_one_probe(X, y_raw, task_name):
    raw_counts = collections.Counter(y_raw)
    dropped = {k: v for k, v in raw_counts.items() if v < 2}
    if dropped:
        keep_mask = np.array([y_raw[i] not in dropped for i in range(len(y_raw))])
        X = X[keep_mask]
        y_raw = [y_raw[i] for i in range(len(y_raw)) if keep_mask[i]]
    if len(set(y_raw)) < 2:
        return None, len(y_raw), f"skip: only {len(set(y_raw))} distinct class after dropping singleton classes {dropped}"
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    counts = collections.Counter(y)
    min_class_n = min(counts.values())
    if min_class_n < 2 or len(X) < 10:
        return None, len(y_raw), f"skip: too few samples/class (min_class_n={min_class_n}, n={len(X)})"
    note = f" (dropped singleton classes {dropped})" if dropped else ""
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xval = scaler.transform(Xtr), scaler.transform(Xval)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xval, yval)
    baseline = max(counts.values()) / sum(counts.values())
    return acc, len(y), f"n_train={len(ytr)} n_val={len(yval)} n_classes={len(le.classes_)} majority_baseline={baseline:.3f}{note}"


def run_event_sequence_probe(records, layer_name):
    # split at video level first, then flatten groups
    idx = list(range(len(records)))
    train_idx, val_idx = train_test_split(idx, test_size=0.3, random_state=0)

    def build_xy(indices):
        X, y = [], []
        for i in indices:
            rec = records[i]
            T = rec["num_temporal_groups"]
            dur_s = video_duration_s(rec["task"], rec["doc_id"])
            events = rec["ground_truth_details"]["video_events_timeline_ms"]
            hs = rec[layer_name]  # (T, H)
            for t in range(T):
                center_ms = (t + 0.5) / T * dur_s * 1000
                is_event = any(e["start_ms"] <= center_ms <= e["end_ms"] for e in events)
                X.append(hs[t].numpy())
                y.append(1 if is_event else 0)
        return np.array(X), np.array(y)

    Xtr, ytr = build_xy(train_idx)
    Xval, yval = build_xy(val_idx)
    if len(set(ytr)) < 2 or len(set(yval)) < 2:
        return None, len(Xtr) + len(Xval), "skip: degenerate label distribution"
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xval = scaler.transform(Xtr), scaler.transform(Xval)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xval, yval)
    baseline = max(np.mean(ytr == 0), np.mean(ytr == 1))
    return acc, len(ytr) + len(yval), f"n_train={len(ytr)} n_val={len(yval)} (per-group rows) majority_baseline={baseline:.3f}"


def main():
    hs_dir, model_tag = sys.argv[1], sys.argv[2]
    print(f"{'task':<35}{'layer':<10}{'n':>6}{'probe_acc':>12}  detail")
    results = {}
    for task in TASK_SUBDIR:
        pt_path = os.path.join(hs_dir, model_tag, f"{task}.pt")
        if not os.path.exists(pt_path):
            print(f"{task:<35} MISSING {pt_path}")
            continue
        records = torch.load(pt_path, weights_only=False)
        for layer_name in LAYERS:
            if task == "Event_Sequence":
                acc, n, detail = run_event_sequence_probe(records, layer_name)
            else:
                X = np.stack([r[layer_name].mean(dim=0).numpy() for r in records])
                y_raw = [WHOLE_VIDEO_LABEL_FNS[task](r) for r in records]
                acc, n, detail = train_one_probe(X, y_raw, task)
            acc_str = f"{100*acc:.2f}%" if acc is not None else "N/A"
            print(f"{task:<35}{layer_name:<10}{n:>6}{acc_str:>12}  {detail}")
            results[(task, layer_name)] = acc
    out_path = os.path.join(PROBE_DATA_DIR, f"probe_results_{model_tag}.json")
    with open(out_path, "w") as f:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in results.items()}, f, indent=2)
    print(f"\n saved: {out_path}")


if __name__ == "__main__":
    main()
