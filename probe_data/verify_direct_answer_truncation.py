#!/usr/bin/env python3
"""用户要求核实(2026-07-13):direct_answer_baseline.py是不是真的"generate→官方提取器判分"，
有没有答案因为撞到max_new_tokens=32被意外截断。这是在thinking-on pilot发现1024 tokens
腰斩思考过程之后,回头检查同样逻辑是否也影响了(thinking关闭的)direct-answer baseline。

方法:用完全相同的greedy解码(do_sample=False)重新generate全部样本——贪婪解码是确定性的,
相同输入应该复现相同输出,所以这不是"换一批新样本",是精确复现原始判分过程,但这次记录
每条样本的生成token数,看有没有触达32这个上限。

用法: python3 verify_direct_answer_truncation.py <model_path> <model_tag> <gpu_ids>
"""
import sys
import os
import json
import time
import collections
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

sys.path.insert(0, "/root/miniconda3/envs/lmmseval/lib/python3.12/site-packages")
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

PROBE_DATA_DIR = "/remote-home/ziyesong/videoPerception/probe_data"
MAX_NEW_TOKENS = 32

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
    model_path, model_tag, gpu_ids = sys.argv[1], sys.argv[2], sys.argv[3]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    n_gpus = len(gpu_ids.split(","))
    device_map = "auto" if n_gpus > 1 else "cuda:0"

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True, device_map=device_map
    ).eval()
    first_device = next(model.parameters()).device

    samples = load_samples()
    print(f"[{model_tag}] total samples: {len(samples)}, max_new_tokens={MAX_NEW_TOKENS} (复现原始direct-answer baseline配置)")

    token_counts = []
    per_task_token_counts = collections.defaultdict(list)
    hit_cap_records = []
    mismatch_vs_original = 0
    t0 = time.time()

    # load original results for cross-check (greedy decoding is deterministic -> should reproduce identical pred)
    orig = json.load(open(os.path.join(PROBE_DATA_DIR, f"direct_answer_{model_tag}.json")))
    orig_wrong_by_id = {w["id"]: w for w in orig["wrong"]}

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
                gen_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
            new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
            n_new = new_tokens.shape[0]
            resp = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred = extract_mcq_answer(resp, choices=choices)
            correct = bool(pred) and pred == gt_letter.upper()

            token_counts.append(n_new)
            per_task_token_counts[task].append(n_new)
            if n_new >= MAX_NEW_TOKENS:
                hit_cap_records.append({"id": d["id"], "task": task, "resp": resp, "n_tokens": n_new})

            # cross-check against original stored result (determinism check)
            if not correct:
                orig_w = orig_wrong_by_id.get(d["id"])
                if orig_w is None:
                    # original run had this marked correct, but replay says wrong -> mismatch
                    mismatch_vs_original += 1
                elif orig_w["pred"] != (pred or "(未解析)"):
                    mismatch_vs_original += 1
            else:
                if d["id"] in orig_wrong_by_id:
                    mismatch_vs_original += 1
        except Exception as e:
            print(f"[{model_tag}] FAILED {d.get('id')}: {e}")

        if (si + 1) % 200 == 0 or (si + 1) == len(samples):
            elapsed = time.time() - t0
            print(f"[{model_tag}] {si+1}/{len(samples)} elapsed={elapsed:.1f}s avg={elapsed/(si+1):.3f}s/sample  hit_cap_so_far={len(hit_cap_records)}  mismatch_so_far={mismatch_vs_original}")

    print(f"\n=== {model_tag}: token count分布(全部{len(token_counts)}条) ===")
    print(f"min={min(token_counts)} max={max(token_counts)} mean={sum(token_counts)/len(token_counts):.2f}")
    dist = collections.Counter(token_counts)
    for k in sorted(dist):
        print(f"  {k} tokens: {dist[k]} 条")
    print(f"\n撞到{MAX_NEW_TOKENS}上限的样本数: {len(hit_cap_records)} ({100*len(hit_cap_records)/len(token_counts):.2f}%)")
    for r in hit_cap_records[:10]:
        print(f"  {r['task']}#{r['id']}: {repr(r['resp'])}")

    print(f"\n=== 按task的token数 ===")
    for task in sorted(per_task_token_counts):
        tc = per_task_token_counts[task]
        print(f"{task:<35} n={len(tc)} min={min(tc)} max={max(tc)} mean={sum(tc)/len(tc):.2f}")

    print(f"\n=== 复现一致性检查(greedy解码应确定性复现原始判分结果) ===")
    print(f"与原始存档结果不一致的样本数: {mismatch_vs_original} / {len(token_counts)}")

    out_path = os.path.join(PROBE_DATA_DIR, f"verify_truncation_{model_tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "token_count_distribution": dict(collections.Counter(token_counts)),
            "hit_cap_count": len(hit_cap_records),
            "hit_cap_records": hit_cap_records,
            "mismatch_vs_original": mismatch_vs_original,
            "per_task_stats": {t: {"n": len(v), "min": min(v), "max": max(v), "mean": sum(v)/len(v)} for t, v in per_task_token_counts.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
