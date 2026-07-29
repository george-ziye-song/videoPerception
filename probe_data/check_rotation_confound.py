#!/usr/bin/env python3
"""用户指出的confound检查(2026-07-13):Rotation_Count/Rotation_Direction的高R²有没有可能
只是"位置编码+大致固定的omega"的副产品,不是真的读懂了视觉旋转。见probe-experiment-report.md §4。

做两组对照:
1. position-only baseline: 只用frame_idx(不用hidden state)去回归theta,看纯位置能解释多少。
   如果这个baseline的R²已经接近hidden-state probe的R²,说明probe大概率在利用位置而非视觉信息。
2. sign-oracle baseline(只对Rotation_Direction有意义,因为|omega|是常数):假设完美知道方向
   (100%正确)、只用位置,重建的理论R²是多少;再和95%分类准确率对应的sign-noisy版本比较,
   看观测到的regression R²落在哪个区间。

用法: python3 check_rotation_confound.py
"""
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys
sys.path.insert(0, "/remote-home/ziyesong/videoPerception/probe_data")
from train_regression_probes import target_rotation_count, target_rotation_direction, group_time_ms, FPS

np.random.seed(0)


def position_only_baseline(recs, target_fn):
    X, Y, video_id = [], [], []
    for vi, rec in enumerate(recs):
        T = rec["num_temporal_groups"]
        for g in range(T):
            X.append([(g + 0.5) / T])  # normalized position only, no hidden state
            Y.append(target_fn(rec, g, T))
            video_id.append(vi)
    X, Y, video_id = np.array(X), np.array(Y), np.array(video_id)
    n_videos = len(recs)
    train_idx, val_idx = train_test_split(range(n_videos), test_size=0.3, random_state=0)
    train_mask, val_mask = np.isin(video_id, train_idx), np.isin(video_id, val_idx)
    reg = LinearRegression().fit(X[train_mask], Y[train_mask])
    pred = reg.predict(X[val_mask])
    return r2_score(Y[val_mask], pred)


def sign_oracle_direction(recs, noise_rate):
    """用完美已知(或按noise_rate掺假)的方向 + 精确位置,重建理论上限"""
    X, Y, video_id = [], [], []
    for vi, rec in enumerate(recs):
        T = rec["num_temporal_groups"]
        direction = rec["ground_truth_details"]["other_details"]["direction"]
        true_sign = -1.0 if direction == "Clockwise" else 1.0
        used_sign = true_sign
        if np.random.rand() < noise_rate:
            used_sign = -true_sign
        for g in range(T):
            t_ms = group_time_ms(g, T, rec["ground_truth_details"]["video_events_timeline_ms"][0]["end_ms"])
            frame_idx = t_ms / 1000 * FPS
            X.append([frame_idx * used_sign])  # "predictor": position combined with (possibly wrong) known sign
            Y.append(target_rotation_direction(rec, g, T))
            video_id.append(vi)
    X, Y, video_id = np.array(X), np.array(Y), np.array(video_id)
    n_videos = len(recs)
    train_idx, val_idx = train_test_split(range(n_videos), test_size=0.3, random_state=0)
    train_mask, val_mask = np.isin(video_id, train_idx), np.isin(video_id, val_idx)
    reg = LinearRegression().fit(X[train_mask], Y[train_mask])
    pred = reg.predict(X[val_mask])
    return r2_score(Y[val_mask], pred)


def omega_distribution(recs, task):
    omegas = []
    for r in recs:
        events = r["ground_truth_details"]["video_events_timeline_ms"]
        duration_ms = events[0]["end_ms"]
        if task == "Rotation_Count":
            num_frames = duration_ms / 1000 * FPS
            total_rotations = r["ground_truth_details"]["other_details"]["total_rotations"]
            omega = (np.pi * 2 * total_rotations) / num_frames
        else:
            base = (np.pi * 2 / FPS) / 2
            direction = r["ground_truth_details"]["other_details"]["direction"]
            omega = -base if direction == "Clockwise" else base
        omegas.append(omega)
    return np.array(omegas)


def residual_test(recs, target_fn):
    """扣掉position-only能解释的部分,看hidden state能不能解释残差(真正的样本特异信息)"""
    from sklearn.preprocessing import StandardScaler
    from train_regression_probes import LAYERS

    n_videos = len(recs)
    train_idx, val_idx = train_test_split(range(n_videos), test_size=0.3, random_state=0)
    pos_feat, Y, video_id = [], [], []
    for vi, rec in enumerate(recs):
        T = rec["num_temporal_groups"]
        for g in range(T):
            pos_feat.append([(g + 0.5) / T])
            Y.append(target_fn(rec, g, T))
            video_id.append(vi)
    pos_feat, Y, video_id = np.array(pos_feat), np.array(Y), np.array(video_id)
    train_mask, val_mask = np.isin(video_id, train_idx), np.isin(video_id, val_idx)

    pos_reg = LinearRegression().fit(pos_feat[train_mask], Y[train_mask])
    resid_train = Y[train_mask] - pos_reg.predict(pos_feat[train_mask])
    resid_val = Y[val_mask] - pos_reg.predict(pos_feat[val_mask])
    print(f"  residual variance fraction of total: {resid_val.var()/Y[val_mask].var():.3f}")

    for layer_name in LAYERS:
        X = []
        for rec in recs:
            T = rec["num_temporal_groups"]
            hs = rec[layer_name]
            for g in range(T):
                X.append(hs[g].numpy())
        X = np.array(X)
        Xtr, Xval = X[train_mask], X[val_mask]
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xval = scaler.transform(Xtr), scaler.transform(Xval)
        reg = LinearRegression().fit(Xtr, resid_train)
        r2_resid = r2_score(resid_val, reg.predict(Xval))
        print(f"  layer={layer_name}: hidden-state R2 on RESIDUAL = {r2_resid:.3f}")


if __name__ == "__main__":
    for model_tag in ["9b", "35b_35"]:
        print(f"\n===== {model_tag} =====")
        for task, fn in [("Rotation_Count", target_rotation_count), ("Rotation_Direction", target_rotation_direction)]:
            recs = torch.load(f"/remote-home/ziyesong/videoPerception/probe_data/hidden_states/{model_tag}/{task}.pt", weights_only=False)
            omegas = omega_distribution(recs, task)
            print(f"{task}: omega mean={omegas.mean():.5f} std={omegas.std():.5f} min={omegas.min():.5f} max={omegas.max():.5f}")
            r2_pos = position_only_baseline(recs, fn)
            print(f"{task}: position-only baseline R2 = {r2_pos:.3f}")
            if task == "Rotation_Count":
                residual_test(recs, fn)

        recs_dir = torch.load(f"/remote-home/ziyesong/videoPerception/probe_data/hidden_states/{model_tag}/Rotation_Direction.pt", weights_only=False)
        for noise in [0.0, 0.05, 0.10, 0.15, 0.47]:
            r2 = sign_oracle_direction(recs_dir, noise)
            print(f"Rotation_Direction sign-oracle (sign错误率={noise:.0%}): R2 = {r2:.3f}")
