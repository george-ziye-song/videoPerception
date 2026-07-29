#!/usr/bin/env python3
"""probe-experiment.md §1.5: 用当前(已修复)的共享 extract_mcq_answer,
对 P1基线(未打乱, results_q35_9b) 和 shuffle前置检验(打乱后, results_shuffle_*_chunks)
两边的原始样本重新打分,保证同一把尺子比较,输出每个子任务的掉分幅度表。

用法: python3 compare_shuffle_vs_p1.py <baseline_root> <shuffle_root> <model_tag>
"""
import sys
import json
import glob
import os
import collections

sys.path.insert(0, "/root/miniconda3/envs/lmmseval/lib/python3.12/site-packages")
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer


def score_mvbench(root):
    per_task = collections.defaultdict(list)
    for samples_file in glob.glob(f"{root}/**/*samples_mvbench_*.jsonl", recursive=True):
        task_name = os.path.basename(samples_file).split("samples_")[-1].replace(".jsonl", "")
        with open(samples_file) as f:
            for line in f:
                d = json.loads(line)
                pred_text = d.get("filtered_resps", "") or ""
                gt = d.get("mvbench_accuracy", {}).get("gt_answer")
                if gt is None:
                    continue
                pred_letter = extract_mcq_answer(pred_text, choices=["A", "B", "C", "D", "E"])
                score = 1 if pred_letter and pred_letter == gt.upper() else 0
                per_task[task_name].append(score)
    return {k: (100 * sum(v) / len(v), len(v)) for k, v in per_task.items()}


def score_tomato(root):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/TOMATO", split="test")
    docs = {i: ds[i] for i in range(len(ds))}

    scores = []
    per_reason = collections.defaultdict(list)
    seen = set()
    for samples_file in glob.glob(f"{root}/**/*samples_tomato.jsonl", recursive=True):
        with open(samples_file) as f:
            for line in f:
                d = json.loads(line)
                doc_id = d["doc_id"]
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                doc = docs[doc_id]
                resp = d.get("filtered_resps", "") or ""
                gt_letter = chr(65 + doc["answer"])
                n_opts = len(doc["options"])
                choices = [chr(65 + i) for i in range(n_opts)]
                pred = extract_mcq_answer(resp, choices=choices)
                correct = bool(pred) and pred == gt_letter
                scores.append(1.0 if correct else 0.0)
                per_reason[doc["reason_type"]].append(1.0 if correct else 0.0)
    if not scores:
        return None
    per_reason_acc = {k: (100 * sum(v) / len(v), len(v)) for k, v in per_reason.items()}
    return 100 * sum(scores) / len(scores), len(scores), per_reason_acc


if __name__ == "__main__":
    baseline_root, shuffle_root, model_tag = sys.argv[1], sys.argv[2], sys.argv[3]

    base_mv = score_mvbench(baseline_root)
    shuf_mv = score_mvbench(shuffle_root)

    print(f"=== {model_tag}: MVBench 子任务 打乱前(P1) vs 打乱后 ===")
    print(f"{'task':<28}{'baseline_n':>11}{'baseline_acc':>14}{'shuffle_n':>11}{'shuffle_acc':>13}{'delta(pp)':>11}")
    all_tasks = sorted(set(base_mv) | set(shuf_mv))
    deltas = []
    for t in all_tasks:
        b = base_mv.get(t)
        s = shuf_mv.get(t)
        if b is None or s is None:
            print(f"{t:<28} missing on one side (baseline={b}, shuffle={s})")
            continue
        delta = b[0] - s[0]
        deltas.append((t, delta, b[0], s[0]))
        print(f"{t:<28}{b[1]:>11}{b[0]:>13.2f}%{s[1]:>11}{s[0]:>12.2f}%{delta:>10.2f}pp")

    if deltas:
        mean_b = sum(d[2] for d in deltas) / len(deltas)
        mean_s = sum(d[3] for d in deltas) / len(deltas)
        print(f"\n{'MEAN OF TASKS':<28}{'':>11}{mean_b:>13.2f}%{'':>11}{mean_s:>12.2f}%{mean_b-mean_s:>10.2f}pp")
        print("\n按掉分幅度排序(掉分越多=越依赖时序):")
        for t, delta, b, s in sorted(deltas, key=lambda x: -x[1]):
            print(f"  {t:<28} baseline={b:6.2f}%  shuffle={s:6.2f}%  delta={delta:+6.2f}pp")

    base_to = score_tomato(baseline_root)
    shuf_to = score_tomato(shuffle_root)
    print(f"\n=== {model_tag}: TOMATO 整体 打乱前 vs 打乱后 ===")
    if base_to and shuf_to:
        print(f"baseline: n={base_to[1]} acc={base_to[0]:.2f}%")
        print(f"shuffle:  n={shuf_to[1]} acc={shuf_to[0]:.2f}%")
        print(f"delta: {base_to[0]-shuf_to[0]:+.2f}pp")
        print(f"\n{'reason_type':<20}{'baseline_n':>11}{'baseline_acc':>14}{'shuffle_n':>11}{'shuffle_acc':>13}{'delta(pp)':>11}")
        b_reason, s_reason = base_to[2], shuf_to[2]
        rt_deltas = []
        for rt in sorted(set(b_reason) | set(s_reason)):
            b = b_reason.get(rt)
            s = s_reason.get(rt)
            if b is None or s is None:
                print(f"{rt:<20} missing on one side (baseline={b}, shuffle={s})")
                continue
            delta = b[0] - s[0]
            rt_deltas.append((rt, delta, b[0], s[0]))
            print(f"{rt:<20}{b[1]:>11}{b[0]:>13.2f}%{s[1]:>11}{s[0]:>12.2f}%{delta:>10.2f}pp")
        print("\n按掉分幅度排序:")
        for rt, delta, b, s in sorted(rt_deltas, key=lambda x: -x[1]):
            print(f"  {rt:<20} baseline={b:6.2f}%  shuffle={s:6.2f}%  delta={delta:+6.2f}pp")
    else:
        print(f"baseline={base_to}  shuffle={shuf_to}")
