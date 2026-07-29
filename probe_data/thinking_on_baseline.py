#!/usr/bin/env python3
"""distillation-readiness-experiments.md 实验0:把4类"读出层gap"原语的直答协议从
"thinking关闭,max_new_tokens=32,贪婪解码"换成P2同款的"thinking开启,官方采样参数",
看会不会白捡分数。

P2同款配置(和run_p2_transformers_chunked.sh完全一致,照搬,不自己发明):
enable_thinking=True(无需reasoning_prompt,9B/35B原生支持thinking)
temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, repetition_penalty=1.0
(presence_penalty原生transformers.generate()不支持,和P2一样用自定义LogitsProcessor)
唯一改动:max_new_tokens从P2的4096降到1024(spec原文:这4类原语题目不长,1024应该够,
如果发现经常撞到上限再加——本脚本会记录hit_cap比例,供判断要不要调大)。

答案提取用共享的官方extract_mcq_answer,不解析/剥离think内容,直接对全部生成文本找答案
(和P2的--reasoning_tags none是同一个逻辑:不丢弃、不预处理,交给提取器自己找)。

用法: python3 thinking_on_baseline.py <model_path> <model_tag> <gpu_ids> <max_new_tokens> [limit] [--tasks=A,B,C]
"""
import sys
import os
import json
import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, LogitsProcessor, LogitsProcessorList
from qwen_vl_utils import process_vision_info

sys.path.insert(0, "/root/miniconda3/envs/lmmseval/lib/python3.12/site-packages")
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

PROBE_DATA_DIR = "/remote-home/ziyesong/videoPerception/probe_data"
GAP_TASKS = ["Rotation_Direction", "Rotation_Count", "Bouncing_Counting", "Acceleration_Identification"]

SOURCES = [
    os.path.join(PROBE_DATA_DIR, "Atomic", "sft.jsonl"),
    os.path.join(PROBE_DATA_DIR, "Atomic2", "sft.jsonl"),
]


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        for i in range(input_ids.shape[0]):
            seen = torch.unique(input_ids[i])
            scores[i, seen] -= self.penalty
        return scores


def load_samples(task_filter):
    samples = []
    for sft_path in SOURCES:
        with open(sft_path) as f:
            for line in f:
                d = json.loads(line)
                if d["task"] in task_filter:
                    samples.append(d)
    return samples


def main():
    tasks_filter = GAP_TASKS
    positional = []
    for arg in sys.argv[1:]:
        if arg.startswith("--tasks="):
            tasks_filter = arg.split("=", 1)[1].split(",")
        else:
            positional.append(arg)
    model_path, model_tag, gpu_ids, max_new_tokens = positional[0], positional[1], positional[2], int(positional[3])
    limit_per_task = int(positional[4]) if len(positional) > 4 else None

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    n_gpus = len(gpu_ids.split(","))
    device_map = "auto" if n_gpus > 1 else "cuda:0"

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True, device_map=device_map
    ).eval()
    first_device = next(model.parameters()).device

    samples = load_samples(tasks_filter)
    if limit_per_task:
        # 每个task类型各取前limit_per_task条,不是简单truncate整个列表(避免只pilot到第一个task)
        capped = []
        counts = {}
        for d in samples:
            counts.setdefault(d["task"], 0)
            if counts[d["task"]] < limit_per_task:
                capped.append(d)
                counts[d["task"]] += 1
        samples = capped
    print(f"[{model_tag}] total samples: {len(samples)}  tasks={tasks_filter}  max_new_tokens={max_new_tokens}")

    logits_processor = LogitsProcessorList([PresencePenaltyLogitsProcessor(1.5)])

    per_task_scores = {}
    per_task_unresolved = {}
    per_task_hit_cap = {}
    per_task_gen_tokens = {}
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
        choices = [chr(65 + i) for i in range(6)]

        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "video", "video": video_path, "nframes": 16},
                {"type": "text", "text": question_text},
            ]},
        ]
        try:
            text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, enable_thinking=True)
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
                gen_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=1.0, top_p=0.95, top_k=20, repetition_penalty=1.0,
                    logits_processor=logits_processor, use_cache=True,
                )
            new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
            n_new = new_tokens.shape[0]
            resp = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            hit_cap = (n_new >= max_new_tokens)

            pred = extract_mcq_answer(resp, choices=choices)
            correct = bool(pred) and pred == gt_letter.upper()
            per_task_scores.setdefault(task, []).append(1.0 if correct else 0.0)
            per_task_gen_tokens.setdefault(task, []).append(n_new)
            if hit_cap:
                per_task_hit_cap[task] = per_task_hit_cap.get(task, 0) + 1
            if not pred:
                per_task_unresolved[task] = per_task_unresolved.get(task, 0) + 1
            if not correct:
                wrong.append({"task": task, "id": d["id"], "gt": gt_letter, "pred": pred or "(未解析)", "resp": resp, "n_tokens": n_new, "hit_cap": hit_cap})
        except Exception as e:
            print(f"[{model_tag}] FAILED {d.get('id')}: {e}")

        if (si + 1) % 20 == 0 or (si + 1) == len(samples):
            elapsed = time.time() - t0
            print(f"[{model_tag}] {si+1}/{len(samples)} elapsed={elapsed:.1f}s avg={elapsed/(si+1):.3f}s/sample")

    print(f"\n=== {model_tag} thinking-on baseline (合成数据, max_new_tokens={max_new_tokens}) ===")
    for task in sorted(per_task_scores):
        scores = per_task_scores[task]
        acc = 100 * sum(scores) / len(scores)
        unresolved = per_task_unresolved.get(task, 0)
        hit_cap = per_task_hit_cap.get(task, 0)
        toks = per_task_gen_tokens[task]
        avg_toks = sum(toks) / len(toks)
        print(f"{task:<30} n={len(scores):<5} acc={acc:6.2f}%  unresolved={unresolved}({100*unresolved/len(scores):.1f}%)  "
              f"hit_cap={hit_cap}({100*hit_cap/len(scores):.1f}%)  avg_gen_tokens={avg_toks:.0f}")

    out_path = os.path.join(PROBE_DATA_DIR, f"thinking_on_{model_tag}.json")
    new_data = {
        "max_new_tokens": max_new_tokens,
        "per_task_acc": {t: 100 * sum(s) / len(s) for t, s in per_task_scores.items()},
        "per_task_n": {t: len(s) for t, s in per_task_scores.items()},
        "per_task_unresolved": per_task_unresolved,
        "per_task_hit_cap": per_task_hit_cap,
        "per_task_avg_gen_tokens": {t: sum(v) / len(v) for t, v in per_task_gen_tokens.items()},
        "wrong": wrong,
    }
    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
