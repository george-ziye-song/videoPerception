#!/usr/bin/env python3
"""用全部33层(9B:embedding+32层)的pooled hidden state,逐层训一个线性probe,
画出完整的"probe准确率/R² vs 层深度"曲线,回答:是不是U型?为什么mid有时候比deep好?

用法: python3 analyze_all_layers.py <hidden_states_alllayers_dir> <model_tag> <task_name> <classification|regression>
"""
import sys
import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score

LABEL_FNS = {
    "Acceleration_Identification": lambda r: r["ground_truth_details"]["other_details"]["motion_type"],
    "Rotation_Direction": lambda r: r["ground_truth_details"]["other_details"]["direction"],
}

# speed scalar per motion_type (hardcoded curve, matches train_regression_probes.py's known constants)
SPEED_BY_MOTION = {"accelerating": 0.5, "constant": 6.0, "decelerating": 12.0}


def main():
    hs_dir, model_tag, task_name, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    pt_path = os.path.join(hs_dir, model_tag, f"{task_name}.pt")
    records = torch.load(pt_path, weights_only=False)
    n_layers = records[0]["all_layers"].shape[0]
    print(f"[{model_tag}] {task_name} mode={mode}: {len(records)} samples, {n_layers} layers")

    if mode == "classification":
        y_raw = [LABEL_FNS[task_name](r) for r in records]
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        import collections
        baseline = max(collections.Counter(y).values()) / len(y)
    else:
        y = np.array([SPEED_BY_MOTION[r["ground_truth_details"]["other_details"]["motion_type"]] for r in records])
        baseline = None

    idx = np.arange(len(records))
    train_idx, val_idx = train_test_split(idx, test_size=0.3, random_state=0, stratify=y if mode == "classification" else None)

    print(f"{'layer':>6}{'%depth':>8}{'score':>10}")
    curve = []
    for li in range(n_layers):
        X = np.stack([r["all_layers"][li].numpy() for r in records])
        Xtr, Xval = X[train_idx], X[val_idx]
        ytr, yval = y[train_idx], y[val_idx]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xval_s = scaler.transform(Xtr), scaler.transform(Xval)
        if mode == "classification":
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr_s, ytr)
            score = clf.score(Xval_s, yval)
        else:
            reg = LinearRegression()
            reg.fit(Xtr_s, ytr)
            pred = reg.predict(Xval_s)
            score = r2_score(yval, pred)
        pct_depth = 100 * li / (n_layers - 1)
        curve.append((li, pct_depth, score))
        marker = ""
        print(f"{li:>6}{pct_depth:>7.0f}%{score:>10.4f}{marker}")

    scores = [c[2] for c in curve]
    best_i = int(np.argmax(scores))
    print(f"\nbest layer: {curve[best_i][0]} ({curve[best_i][1]:.0f}% depth), score={curve[best_i][2]:.4f}")
    if mode == "classification":
        print(f"majority baseline: {baseline:.4f}")
    print(f"layer 8 (25%): {scores[8]:.4f}   layer 16 (50%): {scores[16]:.4f}   layer 29 (90%): {scores[29]:.4f}")


if __name__ == "__main__":
    main()
