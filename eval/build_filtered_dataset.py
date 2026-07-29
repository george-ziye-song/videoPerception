#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用teacher/student的pass-rate结果过滤FINAL_TRAINING_DATASET.json:
保留teacher_correct=True的样本(teacher自己都答错的样本,其hidden state蒸馏信号不可信,予以剔除)。
student_correct不作为过滤条件(dense hidden-state distillation不需要GRPO式的非零方差要求),
但作为元信息保留在每条记录里,便于后续分析。
"""
import json

BASE = "/remote-home/ziyesong/videoPerception"
FINAL = f"{BASE}/FINAL_TRAINING_DATASET.json"
TEACHER = f"{BASE}/eval/passrate_teacher_full.json"
STUDENT = f"{BASE}/eval/passrate_student_full.json"
OUT = f"{BASE}/FINAL_TRAINING_DATASET_passrate_filtered.json"
OUT_STATS = f"{BASE}/FINAL_TRAINING_DATASET_passrate_filtered_stats.json"

data = json.load(open(FINAL))
teacher = {r["id"]: r for r in json.load(open(TEACHER))}
student = {r["id"]: r for r in json.load(open(STUDENT))}

assert len(data) == len(teacher) == len(student), \
    f"条数不一致: data={len(data)} teacher={len(teacher)} student={len(student)}"
missing = [e["id"] for e in data if e["id"] not in teacher or e["id"] not in student]
assert not missing, f"有{len(missing)}条在pass-rate结果里找不到,例如: {missing[:5]}"

kept = []
by_layer_task = {}
for e in data:
    t = teacher[e["id"]]
    s = student[e["id"]]
    layer_task = f"{e['layer']}/{e['task']}"
    stat = by_layer_task.setdefault(layer_task, {"total": 0, "kept": 0})
    stat["total"] += 1
    if t["correct"]:
        e2 = dict(e)
        e2["teacher_correct"] = True
        e2["student_correct"] = bool(s["correct"])
        e2["teacher_pred"] = t["pred"]
        e2["student_pred"] = s["pred"]
        kept.append(e2)
        stat["kept"] += 1

json.dump(kept, open(OUT, "w"), indent=2, ensure_ascii=False)

stats = {
    "total_before": len(data),
    "total_after": len(kept),
    "kept_ratio": round(len(kept) / len(data) * 100, 1),
    "teacher_pass_rate": round(sum(1 for r in teacher.values() if r["correct"]) / len(teacher) * 100, 1),
    "student_pass_rate_on_kept": round(sum(1 for e in kept if e["student_correct"]) / len(kept) * 100, 1) if kept else None,
    "by_layer_task": {
        k: {**v, "kept_ratio": round(v["kept"] / v["total"] * 100, 1)}
        for k, v in sorted(by_layer_task.items())
    },
}
json.dump(stats, open(OUT_STATS, "w"), indent=2, ensure_ascii=False)

print(f"过滤前: {len(data)}")
print(f"过滤后(teacher_correct): {len(kept)} ({stats['kept_ratio']}%)")
print(f"保留样本里student也对的比例: {stats['student_pass_rate_on_kept']}%")
print(f"已写入: {OUT}")
print(f"统计已写入: {OUT_STATS}")
print()
print("按 layer/task 拆分保留率(过滤掉超过一半的task需要留意):")
for k, v in sorted(by_layer_task.items(), key=lambda x: x[1]["kept"] / x[1]["total"]):
    print(f"  {k}: {v['kept']}/{v['total']} ({v['kept']/v['total']*100:.1f}%)")
