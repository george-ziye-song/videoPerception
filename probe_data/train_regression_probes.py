#!/usr/bin/env python3
"""probe-experiment.md §4.4:物理量回归probe。

关键设计决策(2026-07-13):不重新生成数据、不重新抽hidden state——5/7种原语的连续
物理量可以从已经存的video_events_timeline_ms(时间戳)+other_details(参数)+CONFIG里
固定的fps=30/width=512解析算出来(渲染器的运动公式是确定性的,唯一的自由变量是这几个
已经存下来的参数),直接复用hidden_states/{9b,35b_35}/*.pt里已经pooling好的表示。

跳过的2种(如实留白,不勉强凑):
- Bouncing_Counting: 初始速度是random.uniform(-4,4),这个值没有存进metadata,无法从
  存下来的GT反推出精确轨迹(需要重新生成才能做)。
- Event_Sequence: 物体是离散的出现/消失,不是连续运动,没有自然的物理量可回归
  (呼应文档对03-10号生成器的排除逻辑)。

物理量选择(每种任务选一个信息量最丰富、且能严格从renderer公式反推的连续目标):
- Rotation_Direction / Rotation_Count: 累积旋转角度theta(t)=frame_idx*rot_speed
  (而不是角速度本身——角速度在单个样本内是常数,角度才是随时间连续变化的量,回归
  更有意义)。
- Acceleration_Identification: 速度标量(不做完整(x,y)position,因为起始点/运动轴
  没有存下来,但3种motion_type的速度曲线本身是硬编码常数,可以精确反推)。
- Complex_Direction_Identification / Directional_Event_Counting: 瞬时2D速度向量
  (vx, vy),按renderer的分段线性公式(含Directional_Event_Counting的"hold尾段"逻辑)
  精确重建。

用法: python3 train_regression_probes.py hidden_states 9b
"""
import sys
import os
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

FPS = 30
WIDTH = 512
LAYERS = ["shallow", "mid", "deep"]

MOVE_VEC = {"Right": (1, 0), "Left": (-1, 0), "Up": (0, -1), "Down": (0, 1),
            "right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1)}


def group_time_ms(g, T, duration_ms):
    return (g + 0.5) / T * duration_ms


def target_rotation_direction(rec, g, T):
    events = rec["ground_truth_details"]["video_events_timeline_ms"]
    duration_ms = events[0]["end_ms"]
    t_ms = group_time_ms(g, T, duration_ms)
    frame_idx = t_ms / 1000 * FPS
    base = (np.pi * 2 / FPS) / 2
    direction = rec["ground_truth_details"]["other_details"]["direction"]
    rot_speed = -base if direction == "Clockwise" else base
    return [frame_idx * rot_speed]


def target_rotation_count(rec, g, T):
    events = rec["ground_truth_details"]["video_events_timeline_ms"]
    duration_ms = events[0]["end_ms"]
    num_frames = duration_ms / 1000 * FPS
    total_rotations = rec["ground_truth_details"]["other_details"]["total_rotations"]
    rot_speed = (np.pi * 2 * total_rotations) / num_frames
    t_ms = group_time_ms(g, T, duration_ms)
    frame_idx = t_ms / 1000 * FPS
    return [frame_idx * rot_speed]


def target_acceleration(rec, g, T):
    events = rec["ground_truth_details"]["video_events_timeline_ms"]
    duration_ms = events[0]["end_ms"]
    t_ms = group_time_ms(g, T, duration_ms)
    frame_idx = t_ms / 1000 * FPS
    motion_type = rec["ground_truth_details"]["other_details"]["motion_type"]
    if motion_type == "accelerating":
        speed, accel = 0.5, 0.25
    elif motion_type == "decelerating":
        speed, accel = 12.0, -0.25
    else:
        speed, accel = 7.0, 0.0
    v = speed + accel * frame_idx
    return [max(0.0, v)]


def target_complex_direction(rec, g, T):
    events = rec["ground_truth_details"]["video_events_timeline_ms"]
    duration_ms = events[-1]["end_ms"]
    t_ms = group_time_ms(g, T, duration_ms)
    move_dist = WIDTH / 4
    for ev in events:
        if ev["start_ms"] <= t_ms <= ev["end_ms"]:
            leg_frames = (ev["end_ms"] - ev["start_ms"]) / 1000 * FPS
            vx, vy = MOVE_VEC[ev["direction"]]
            speed = move_dist / max(1, leg_frames - 1)
            return [vx * speed, vy * speed]
    return [0.0, 0.0]


def target_bouncing_counting(rec, g, T):
    # 2026-07-13追加:generate_bouncing_counting()现在直接dump逐帧(x,y,vx,vy),
    # 不需要解析重建(初始速度random.uniform(-4,4)之前没存,现在直接存物理状态本身)。
    psf = rec["ground_truth_details"]["other_details"]["physical_state_per_frame"]
    duration_ms = len(psf) / FPS * 1000
    t_ms = group_time_ms(g, T, duration_ms)
    frame_idx = int(round(t_ms / 1000 * FPS))
    frame_idx = max(0, min(len(psf) - 1, frame_idx))
    x, y, vx, vy = psf[frame_idx]
    return [vx, vy]


def target_directional_event_counting(rec, g, T):
    events = rec["ground_truth_details"]["video_events_timeline_ms"]
    duration_ms = events[-1]["end_ms"]
    t_ms = group_time_ms(g, T, duration_ms)
    move_dist = 80
    for ev in events:
        if ev["start_ms"] <= t_ms <= ev["end_ms"]:
            seg_frames = (ev["end_ms"] - ev["start_ms"]) / 1000 * FPS
            frame_in_seg = (t_ms - ev["start_ms"]) / 1000 * FPS
            if frame_in_seg >= seg_frames - 5:
                return [0.0, 0.0]  # hold tail, per renderer logic
            vx, vy = MOVE_VEC[ev["direction"]]
            speed = move_dist / max(1, seg_frames - 6)
            return [vx * speed, vy * speed]
    return [0.0, 0.0]


TARGET_FNS = {
    "Rotation_Direction": target_rotation_direction,
    "Rotation_Count": target_rotation_count,
    "Acceleration_Identification": target_acceleration,
    "Complex_Direction_Identification": target_complex_direction,
    "Directional_Event_Counting": target_directional_event_counting,
    "Bouncing_Counting": target_bouncing_counting,
}

TARGET_DESC = {
    "Rotation_Direction": "累积旋转角度theta(rad)",
    "Rotation_Count": "累积旋转角度theta(rad)",
    "Acceleration_Identification": "速度标量|v|",
    "Complex_Direction_Identification": "瞬时速度向量(vx,vy)",
    "Directional_Event_Counting": "瞬时速度向量(vx,vy)",
    "Bouncing_Counting": "瞬时速度向量(vx,vy)(生成器已改,直接dump真值,非解析重建)",
}

SKIPPED = {
    "Event_Sequence": "物体是离散出现/消失,不是连续运动,没有自然物理量可回归",
}


def build_xy(records, layer_name, target_fn):
    X, Y = [], []
    for rec in records:
        T = rec["num_temporal_groups"]
        hs = rec[layer_name]
        for g in range(T):
            X.append(hs[g].numpy())
            Y.append(target_fn(rec, g, T))
    return np.array(X), np.array(Y)


def main():
    hs_dir, model_tag = sys.argv[1], sys.argv[2]
    print(f"{'task':<38}{'target':<24}{'layer':<9}{'n_rows':>8}{'R2':>10}  detail")
    results = {}
    for task, target_fn in TARGET_FNS.items():
        pt_path = os.path.join(hs_dir, model_tag, f"{task}.pt")
        records = torch.load(pt_path, weights_only=False)
        for layer_name in LAYERS:
            X, Y = build_xy(records, layer_name, target_fn)
            n_videos = len(records)
            train_idx, val_idx = train_test_split(range(n_videos), test_size=0.3, random_state=0)
            row_video_id = np.repeat(np.arange(n_videos), [records[i]["num_temporal_groups"] for i in range(n_videos)])
            train_mask = np.isin(row_video_id, train_idx)
            val_mask = np.isin(row_video_id, val_idx)
            Xtr, Xval, Ytr, Yval = X[train_mask], X[val_mask], Y[train_mask], Y[val_mask]

            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xval_s = scaler.transform(Xtr), scaler.transform(Xval)
            reg = LinearRegression()
            reg.fit(Xtr_s, Ytr)
            pred = reg.predict(Xval_s)
            r2 = r2_score(Yval, pred, multioutput="variance_weighted" if Yval.shape[1] > 1 else "uniform_average")
            print(f"{task:<38}{TARGET_DESC[task]:<24}{layer_name:<9}{len(Yval):>8}{r2:>10.3f}  n_train_rows={len(Ytr)}")
            results[f"{task}|{layer_name}"] = r2

    print("\n跳过的原语(如实留白):")
    for task, reason in SKIPPED.items():
        print(f"  {task}: {reason}")

    import json
    out_path = f"/remote-home/ziyesong/videoPerception/probe_data/regression_probe_results_{model_tag}.json"
    with open(out_path, "w") as f:
        json.dump({"r2": results, "skipped": SKIPPED, "target_desc": TARGET_DESC}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
