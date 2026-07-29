#!/usr/bin/env python3
"""probe-experiment.md §4.3:同一批合成样本,记录模型自己的直接作答准确率
(直答协议,和§4.1抽hidden state用一样的prompt构造,只是这次真正generate),
用共享的官方extract_mcq_answer(和P1/P2用的是同一个函数)算accuracy,不用另写解析器。

用法: python3 direct_answer_baseline.py <model_path> <model_tag> <gpu_ids> <nframes> [limit] [--task=TaskName]
--task=TaskName 只跑指定task类型(用于补跑单个原语,避免重跑全部1400条)。
"""
import sys
import os
import json
import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

sys.path.insert(0, "/root/miniconda3/envs/lmmseval/lib/python3.12/site-packages")
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

PROBE_DATA_DIR = "/remote-home/ziyesong/videoPerception/probe_data"

SOURCES = [
    os.path.join(PROBE_DATA_DIR, "Atomic", "sft.jsonl"),
    os.path.join(PROBE_DATA_DIR, "Atomic2", "sft.jsonl"),
]


def load_samples():
    samples = []
    for sft_path in SOURCES:
        with open(sft_path) as f:
            for line in f:
                samples.append(json.loads(line))
    return samples


def main():
    task_filter = None
    positional = []
    for arg in sys.argv[1:]:
        if arg.startswith("--task="):
            task_filter = arg.split("=", 1)[1]
        else:
            positional.append(arg)
    model_path, model_tag, gpu_ids, nframes = positional[0], positional[1], positional[2], int(positional[3])
    limit = int(positional[4]) if len(positional) > 4 else None

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    n_gpus = len(gpu_ids.split(","))
    device_map = "auto" if n_gpus > 1 else "cuda:0"

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True, device_map=device_map
    ).eval()
    first_device = next(model.parameters()).device

    samples = load_samples()
    if task_filter:
        samples = [d for d in samples if d["task"] == task_filter]
    if limit:
        samples = samples[:limit]
    print(f"[{model_tag}] total samples: {len(samples)}" + (f" (filtered to task={task_filter})" if task_filter else ""))

    per_task_scores = {}
    per_task_unresolved = {}
    wrong = []
    t0 = time.time()
    for si, d in enumerate(samples):
        task = d["task"]
        user_content = d["messages"][1]["content"]
        video_path = None
        question_text = None
        for c in user_content:
            if c["type"] == "video":
                video_path = c["video"]
            elif c["type"] == "text":
                question_text = c["text"]
        gt_letter = d["messages"][2]["content"]
        n_choices = question_text.count("\n") + 1  # rough upper bound, extract_mcq_answer just needs the choice set
        choices = [chr(65 + i) for i in range(6)]  # generous superset; extractor only matches what's actually in the choices list AND present in text

        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "video", "video": video_path, "nframes": nframes},
                {"type": "text", "text": question_text},
            ]},
        ]
        try:
            text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            image_inputs, video_inputs, processed_video_kwargs = process_vision_info(
                [message], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True,
            )
            video_metadata_list = None
            if video_inputs is not None:
                video_inputs, video_metadata_list = map(list, zip(*video_inputs))
            inputs = processor(
                text=text, images=image_inputs, videos=video_inputs, video_metadata=video_metadata_list,
                **processed_video_kwargs, do_resize=False, return_tensors="pt",
            )
            inputs = {k: (v.to(first_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
            new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
            resp = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

            pred = extract_mcq_answer(resp, choices=choices)
            correct = bool(pred) and pred == gt_letter.upper()
            per_task_scores.setdefault(task, []).append(1.0 if correct else 0.0)
            if not pred:
                per_task_unresolved[task] = per_task_unresolved.get(task, 0) + 1
            if not correct:
                wrong.append({"task": task, "id": d["id"], "gt": gt_letter, "pred": pred or "(未解析)", "resp": resp})
        except Exception as e:
            print(f"[{model_tag}] FAILED {d.get('id')}: {e}")

        if (si + 1) % 100 == 0 or (si + 1) == len(samples):
            elapsed = time.time() - t0
            print(f"[{model_tag}] {si+1}/{len(samples)} elapsed={elapsed:.1f}s avg={elapsed/(si+1):.3f}s/sample")

    print(f"\n=== {model_tag} 直答baseline (合成数据) ===")
    for task in sorted(per_task_scores):
        scores = per_task_scores[task]
        acc = 100 * sum(scores) / len(scores)
        unresolved = per_task_unresolved.get(task, 0)
        print(f"{task:<35} n={len(scores):<5} acc={acc:6.2f}%  unresolved={unresolved}({100*unresolved/len(scores):.1f}%)")

    out_path = os.path.join(PROBE_DATA_DIR, f"direct_answer_{model_tag}.json")
    new_data = {
        "per_task_acc": {t: 100 * sum(s) / len(s) for t, s in per_task_scores.items()},
        "per_task_n": {t: len(s) for t, s in per_task_scores.items()},
        "per_task_unresolved": per_task_unresolved,
        "wrong": wrong,
    }
    if task_filter and os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        for key in ["per_task_acc", "per_task_n", "per_task_unresolved"]:
            existing.setdefault(key, {}).update(new_data[key])
        existing["wrong"] = [w for w in existing.get("wrong", []) if w["task"] != task_filter] + new_data["wrong"]
        new_data = existing
    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2)
    print(f"\nsaved: {out_path}" + (f" (合并了task={task_filter}的新结果,其余task保留原值)" if task_filter else ""))


if __name__ == "__main__":
    main()
