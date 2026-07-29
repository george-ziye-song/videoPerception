#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在原有pass-rate过滤基础上的第二轮重建:
1. object_shuffle/character_order已拆分(见FINAL_TRAINING_DATASET.json里的task字段,现在是分开的)
2. 对"双方基本都会、teacher没有明显优势"的task做降重:保留全部真gap样本(teacher对/student错) +
   有限的双方都对样本(封顶=max(3倍gap数, 100)),避免陪跑样本占掉大部分体量
3. 常规层(broad_coverage)整体降量,按来源比例縮到约1000条(用户要求:student在这层gap很小,该给专项让位置)
4. Bouncing_Counting/Acceleration_Identification暂不变(扩容调研进行中,后续单独并入)
5. counterfactual_inference/moving_direction(CLEVRER)暂未加入(可行性调研进行中)

这是过渡版本,调研结果回来后还会再跑一版。
"""
import json, random

random.seed(42)

BASE = "/remote-home/ziyesong/videoPerception"
data = json.load(open(f"{BASE}/FINAL_TRAINING_DATASET.json"))
teacher = {r["id"]: r for r in json.load(open(f"{BASE}/eval/passrate_teacher_full.json"))}
student = {r["id"]: r for r in json.load(open(f"{BASE}/eval/passrate_student_full.json"))}

DOWNSIZE_TASKS = {"unexpected_action", "Rotation_Count", "fine_grained_pose", "object_shuffle"}
BROAD_COVERAGE_TARGET = 1000  # 从2,200(kept)缩到约1000,按来源比例保持不变

from collections import defaultdict
by_task = defaultdict(list)
for e in data:
    if teacher[e["id"]]["correct"]:
        by_task[e["task"]].append(e)

kept = []
report = []

for task, items in by_task.items():
    if task == "broad_coverage":
        continue  # 常规层单独按来源比例处理,见下面
    if task not in DOWNSIZE_TASKS:
        kept.extend(items)
        report.append((task, len(items), len(items), "unchanged"))
        continue
    gap = [e for e in items if not student[e["id"]]["correct"]]
    redundant = [e for e in items if student[e["id"]]["correct"]]
    cap = max(3 * len(gap), 100)
    random.shuffle(redundant)
    redundant_kept = redundant[:cap]
    task_kept = gap + redundant_kept
    kept.extend(task_kept)
    report.append((task, len(items), len(task_kept), f"gap={len(gap)}+redundant={len(redundant_kept)}/{len(redundant)}"))

# 常规层:按来源比例缩到BROAD_COVERAGE_TARGET
bc_items = by_task.get("broad_coverage", [])
by_source = defaultdict(list)
for e in bc_items:
    by_source[e["source"]].append(e)
bc_total = len(bc_items)
bc_kept = []
for src, items in by_source.items():
    target = round(len(items) / bc_total * BROAD_COVERAGE_TARGET)
    random.shuffle(items)
    bc_kept.extend(items[:target])
    report.append((f"broad_coverage/{src}", len(items), min(target, len(items)), "按比例缩量"))
kept.extend(bc_kept)

for e in kept:
    e["teacher_correct"] = True
    e["student_correct"] = bool(student[e["id"]]["correct"])
    e["teacher_pred"] = teacher[e["id"]]["pred"]
    e["student_pred"] = student[e["id"]]["pred"]

json.dump(kept, open(f"{BASE}/FINAL_TRAINING_DATASET_passrate_filtered_v2.json", "w"), indent=2, ensure_ascii=False)
json.dump(kept, open(f"{BASE}/FINAL_TRAINING_DATASET_final.json", "w"), indent=2, ensure_ascii=False)

print(f"{'task':45s} {'降重前':>8} {'降重后':>8}  说明")
for task, before, after, note in sorted(report, key=lambda x: -x[1]):
    print(f"{task:45s} {before:8d} {after:8d}  {note}")

zhuanxiang = sum(a for t, b, a, n in report if not t.startswith("broad_coverage"))
changgui = sum(a for t, b, a, n in report if t.startswith("broad_coverage"))
print()
print(f"专项合计: {zhuanxiang}  常规合计: {changgui}  总计: {len(kept)}")
print(f"专项:常规 = {zhuanxiang/(zhuanxiang+changgui)*100:.0f}:{changgui/(zhuanxiang+changgui)*100:.0f}")
