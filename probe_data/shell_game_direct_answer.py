#!/usr/bin/env python3
"""distillation-readiness-experiments.md 实验2:basic_shell_game的直接作答baseline,
和§4.3同款方法——同一个"after"截断视频(video_end=最后一次swap结束时刻,和extract_shell_game_probes.py
的after stage完全相同的输入),这次真正generate,用共享的官方extract_mcq_answer解析。

用法: python3 shell_game_direct_answer.py <model_path> <model_tag> <gpu_ids> [limit]
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
SG_DIR = os.path.join(PROBE_DATA_DIR, "ShellGame")
NFRAMES = 8


def load_samples():
    metas = {}
    with open(os.path.join(SG_DIR, "metadata.jsonl")) as f:
        for line in f:
            m = json.loads(line)
            metas[m["problem_id"]] = m
    samples = []
    with open(os.path.join(SG_DIR, "sft.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            d["_metadata"] = metas.get(d["extra"]["id"])
            samples.append(d)
    return samples


def get_t_after(meta):
    tl = meta["ground_truth_details"]["video_events_timeline_ms"]
    swaps = [e for e in tl if e["event_type"] in ("swap", "magic_swap")]
    return swaps[-1]["end_ms"] / 1000.0


def main():
    model_path, model_tag, gpu_ids = sys.argv[1], sys.argv[2], sys.argv[3]
    max_new_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else None

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    n_gpus = len(gpu_ids.split(","))
    device_map = "auto" if n_gpus > 1 else "cuda:0"

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True, device_map=device_map
    ).eval()
    first_device = next(model.parameters()).device

    samples = load_samples()
    if limit:
        samples = samples[:limit]
    print(f"[{model_tag}] total samples: {len(samples)}")

    scores, unresolved, wrong, hit_cap = [], 0, [], 0
    t0 = time.time()
    for si, d in enumerate(samples):
        meta = d["_metadata"]
        video_path = d["extra"]["original_video_path"]
        question_text = meta["problem"].replace("<video>\n", "")
        gt_letter = meta["answer"]
        n_opts = len(meta["ground_truth_details"]["question_data"]["options"])
        choices = [chr(65 + i) for i in range(n_opts)]
        t_after = get_t_after(meta)

        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "video", "video": video_path, "nframes": NFRAMES, "video_start": 0.0, "video_end": t_after},
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
                gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
            new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
            n_new = new_tokens.shape[0]
            if n_new >= max_new_tokens:
                hit_cap += 1
            resp = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

            pred = extract_mcq_answer(resp, choices=choices)
            correct = bool(pred) and pred == gt_letter.upper()
            scores.append(1.0 if correct else 0.0)
            if not pred:
                unresolved += 1
            if not correct:
                wrong.append({"id": d["extra"]["id"], "gt": gt_letter, "pred": pred or "(未解析)", "resp": resp, "n_tokens": n_new})
        except Exception as e:
            print(f"[{model_tag}] FAILED {d['extra']['id']}: {e}")

        if (si + 1) % 50 == 0 or (si + 1) == len(samples):
            elapsed = time.time() - t0
            print(f"[{model_tag}] {si+1}/{len(samples)} elapsed={elapsed:.1f}s avg={elapsed/(si+1):.3f}s/sample")

    n = len(scores)
    acc = 100 * sum(scores) / n if n else 0
    print(f"\n[{model_tag}] basic_shell_game 直答(after截断, max_new_tokens={max_new_tokens}) n={n} acc={acc:.2f}% unresolved={unresolved}({100*unresolved/n:.1f}%) hit_cap={hit_cap}({100*hit_cap/n:.1f}%)")

    out_path = os.path.join(PROBE_DATA_DIR, f"shellgame_direct_answer_{model_tag}_mt{max_new_tokens}.json")
    with open(out_path, "w") as f:
        json.dump({"n": n, "acc": acc, "unresolved": unresolved, "hit_cap": hit_cap, "max_new_tokens": max_new_tokens, "wrong": wrong}, f, indent=2)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
