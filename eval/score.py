#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge shard outputs and compute per-benchmark metrics for base Qwen3-VL-8B."""
import os, json, glob, collections
import pandas as pd

RES = os.path.join(os.path.dirname(__file__), "results")
TB_DATA = "/root/benchmarks/TemporalBench/data"


def load(bench):
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, f"{bench}.shard*.jsonl"))):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    # dedup by uid (in case of overlap on resume)
    seen = {}
    for r in rows:
        seen[r["uid"]] = r
    return list(seen.values())


def score_mvbench():
    rows = load("mvbench")
    if not rows:
        return None
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r["correct"])
    per_task = {t: sum(v) / len(v) for t, v in sorted(by.items())}
    mean_of_tasks = sum(per_task.values()) / len(per_task)
    micro = sum(r["correct"] for r in rows) / len(rows)
    missing = sum(1 for r in rows if r.get("raw") in ("MEDIA_MISSING",) or (r.get("raw", "").startswith("ERR")))
    return dict(benchmark="MVBench", n=len(rows), n_tasks=len(per_task),
                mean_of_tasks_acc=round(mean_of_tasks, 4), micro_acc=round(micro, 4),
                media_errors=missing, per_task={t: round(a, 4) for t, a in per_task.items()})


def _tb_category_map(split):
    fn = "test_short_qa" if split == "short" else "test_long_qa"
    df = pd.read_parquet(os.path.join(TB_DATA, f"{fn}-00000-of-00001.parquet"))
    m = {}
    for _, r in df.iterrows():
        m[f"temporalbench_{split}/{r['idx']}"] = str(r.get("category", "") or "uncat")
    return m


def score_temporalbench(split):
    bench = f"temporalbench_{split}"
    rows = load(bench)
    if not rows:
        return None
    binary = sum(r["correct"] for r in rows) / len(rows)
    # Multiple Binary Accuracy: group by video (group field), all correct
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r["correct"])
    mba = sum(1 for v in groups.values() if all(v)) / len(groups)
    # per-category
    catmap = _tb_category_map(split)
    bycat = collections.defaultdict(list)
    for r in rows:
        bycat[catmap.get(r["uid"], "uncat")].append(r["correct"])
    per_cat = {c: round(sum(v) / len(v), 4) for c, v in sorted(bycat.items())}
    # per-dataset
    byds = collections.defaultdict(list)
    for r in rows:
        byds[r["task"]].append(r["correct"])
    per_ds = {d: round(sum(v) / len(v), 4) for d, v in sorted(byds.items())}
    return dict(benchmark=f"TemporalBench-{split}", n=len(rows), n_videos=len(groups),
                binary_acc=round(binary, 4), multiple_binary_acc=round(mba, 4),
                per_category=per_cat, per_dataset=per_ds)


def score_tomato():
    rows = load("tomato")
    if not rows:
        return None
    micro = sum(r["correct"] for r in rows) / len(rows)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r["correct"])
    per_type = {t: round(sum(v) / len(v), 4) for t, v in sorted(by.items())}
    errs = sum(1 for r in rows if r.get("raw", "").startswith("ERR") or r.get("raw") == "MEDIA_MISSING")
    return dict(benchmark="TOMATO", n=len(rows), acc=round(micro, 4),
                media_errors=errs, per_reason_type=per_type)


def main():
    out = {}
    for name, fn in [("mvbench", score_mvbench),
                     ("temporalbench_short", lambda: score_temporalbench("short")),
                     ("temporalbench_long", lambda: score_temporalbench("long")),
                     ("tomato", score_tomato)]:
        try:
            r = fn()
        except Exception as e:
            r = {"error": repr(e)}
        if r:
            out[name] = r
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(os.path.join(RES, "scores.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
