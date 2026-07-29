#!/usr/bin/env python3
"""distillation-readiness-experiments.md 实验2:basic_shell_game的"遮挡前/遮挡后"
probe对比。核心判定(和probe-experiment.md §4.2同样的线性probe方法,但预测目标从
"single timepoint"换成"before vs after"两个独立时间点):
- before probe准确率:预测球的起始位置(应该接近100%,画面里直接可见,是方法有效性的
  sanity check上限)
- after probe准确率(核心问题):预测球最终位置(=真实答案),这个时间点画面上从来没有
  显示过答案(basic_shell_game整个视频都不揭晓,swap阶段容器不透明遮住球,后段也不显示)。
  如果after probe准确率显著低于before,说明模型没能在内部维持"球在哪"这个状态穿越遮挡,
  这才是PSD(novelty.md §2.2)要解决的核心场景。

用法: python3 train_shell_game_probes.py <hidden_states_shellgame_dir> <model_tag>
"""
import sys
import os
import json
import collections
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

LAYERS = ["shallow", "mid", "deep"]
PROBE_DATA_DIR = "/remote-home/ziyesong/videoPerception/probe_data"


def train_one(X, y_raw):
    counts = collections.Counter(y_raw)
    dropped = {k: v for k, v in counts.items() if v < 2}
    if dropped:
        keep = np.array([y_raw[i] not in dropped for i in range(len(y_raw))])
        X = X[keep]
        y_raw = [y_raw[i] for i in range(len(y_raw)) if keep[i]]
    if len(set(y_raw)) < 2:
        return None, len(y_raw), f"skip: only {len(set(y_raw))} class after dropping singletons {dropped}"
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xval = scaler.transform(Xtr), scaler.transform(Xval)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xval, yval)
    baseline = max(collections.Counter(y).values()) / len(y)
    return acc, len(y), f"n_train={len(ytr)} n_val={len(yval)} n_classes={len(le.classes_)} majority_baseline={baseline:.3f}"


def main():
    hs_dir, model_tag = sys.argv[1], sys.argv[2]
    results = {}
    print(f"{'stage':<10}{'layer':<10}{'n':>6}{'probe_acc':>12}  detail")
    for stage, target_field in [("before", "initial_pos"), ("after", "final_pos")]:
        pt_path = os.path.join(hs_dir, model_tag, f"{stage}.pt")
        if not os.path.exists(pt_path):
            print(f"{stage:<10} MISSING {pt_path}")
            continue
        records = torch.load(pt_path, weights_only=False)
        for layer_name in LAYERS:
            X = np.stack([r[layer_name].numpy() for r in records])
            y_raw = [r[target_field] for r in records]
            acc, n, detail = train_one(X, y_raw)
            acc_str = f"{100*acc:.2f}%" if acc is not None else "N/A"
            print(f"{stage:<10}{layer_name:<10}{n:>6}{acc_str:>12}  {detail}")
            results[f"{stage}|{layer_name}"] = acc

    out_path = os.path.join(PROBE_DATA_DIR, f"shellgame_probe_results_{model_tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out_path}")

    # 直答baseline读取(如果已存在)
    da_path = os.path.join(PROBE_DATA_DIR, f"shellgame_direct_answer_{model_tag}.json")
    if os.path.exists(da_path):
        with open(da_path) as f:
            da = json.load(f)
        print(f"\n直答baseline(已有): {da}")


if __name__ == "__main__":
    main()
