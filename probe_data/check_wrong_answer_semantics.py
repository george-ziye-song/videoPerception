#!/usr/bin/env python3
"""用户指出的另一处未排除的混淆(2026-07-13):"probe高、直答低"的4类原语,错误会不会
是选项语义混淆(比如顺时针/逆时针表述让模型选错)而不是真的推理/时序理解问题。见
probe-experiment-report.md §3。

把每条wrong answer的字母还原成真实语义值(选项在每条样本里是随机打乱的,不能直接比字母),
按原语类型分类统计错误模式。

用法: python3 check_wrong_answer_semantics.py
"""
import json
import re
import collections

SOURCES = [
    "/remote-home/ziyesong/videoPerception/probe_data/Atomic/sft.jsonl",
    "/remote-home/ziyesong/videoPerception/probe_data/Atomic2/sft.jsonl",
]

id_to_options = {}
for path in SOURCES:
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d["messages"][1]["content"][1]["text"]
            m = re.search(r"Possible answer choices: (.+?)\nThe best answer is:", text, re.DOTALL)
            opts = {}
            for line2 in m.group(1).split("\n"):
                mm = re.match(r"([A-Z])\. (.+)", line2.strip())
                if mm:
                    opts[mm.group(1)] = mm.group(2).strip()
            id_to_options[d["id"]] = opts


def analyze(task, wrongs):
    print(f"\n--- {task} (n_wrong={len(wrongs)}) ---")
    if task == "Rotation_Direction":
        confuse_other_real = 0
        confuse_distractor = 0
        for w in wrongs:
            opts = id_to_options[w["id"]]
            pred_val = opts.get(w["pred"])
            if pred_val == "It is not spinning":
                confuse_distractor += 1
            elif pred_val in ("Clockwise", "Counter-clockwise"):
                confuse_other_real += 1
        print(f"  误选成另一个真实方向(顺/逆时针弄反): {confuse_other_real} ({100*confuse_other_real/len(wrongs):.0f}%)")
        print(f"  误选成'没有转'这个无关distractor: {confuse_distractor} ({100*confuse_distractor/len(wrongs):.0f}%)")
    elif task == "Acceleration_Identification":
        pairs = collections.Counter()
        for w in wrongs:
            opts = id_to_options[w["id"]]
            gt_val, pred_val = opts.get(w["gt"]), opts.get(w["pred"])
            pairs[f"{gt_val} -> 误选 {pred_val}"] += 1
        for k, v in pairs.most_common():
            print(f"  {k}: {v} ({100*v/len(wrongs):.0f}%)")
    elif task in ("Rotation_Count", "Bouncing_Counting"):
        diffs = []
        for w in wrongs:
            opts = id_to_options[w["id"]]
            gt_val, pred_val = opts.get(w["gt"]), opts.get(w["pred"])
            try:
                diffs.append(int(pred_val) - int(gt_val))
            except (TypeError, ValueError):
                continue
        near_miss = sum(1 for d in diffs if abs(d) == 1)
        far_miss = sum(1 for d in diffs if abs(d) > 1)
        print(f"  差1(近似正确,数错1个): {near_miss} ({100*near_miss/len(diffs):.0f}%)")
        print(f"  差>1(离谱): {far_miss} ({100*far_miss/len(diffs):.0f}%)")
        print(f"  误差分布: {collections.Counter(diffs)}")


if __name__ == "__main__":
    for model_tag in ["9b", "35b_35"]:
        print(f"\n===== {model_tag} =====")
        d = json.load(open(f"/remote-home/ziyesong/videoPerception/probe_data/direct_answer_{model_tag}.json"))
        for task in ["Rotation_Direction", "Rotation_Count", "Bouncing_Counting", "Acceleration_Identification"]:
            wrongs = [w for w in d["wrong"] if w["task"] == task]
            analyze(task, wrongs)
