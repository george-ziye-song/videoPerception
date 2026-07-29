#!/usr/bin/env python3
"""用户追问:shallow/mid/deep(25%/50%/90%)只是抽样3层,真实的逐层曲线是U型还是别的形状?
为什么mid有时候比deep分数更高?这个脚本对1-2个task抽取全部33层(9B: embedding+32层)的
pooled hidden state,不再只抽3层,拿到完整的逐层曲线来回答这个问题,不是靠猜。

用法: python3 extract_all_layers.py <model_path> <model_tag> <gpu_ids> <task_name> [limit]
"""
import sys
import os
import json
import time
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

PROBE_DATA_DIR = "/remote-home/ziyesong/videoPerception/probe_data"
OUT_DIR = os.path.join(PROBE_DATA_DIR, "hidden_states_alllayers")

SOURCES = [
    (os.path.join(PROBE_DATA_DIR, "Atomic", "sft.jsonl"), os.path.join(PROBE_DATA_DIR, "Atomic", "metadata.jsonl")),
    (os.path.join(PROBE_DATA_DIR, "Atomic2", "sft.jsonl"), os.path.join(PROBE_DATA_DIR, "Atomic2", "metadata.jsonl")),
]


def load_samples(task_name):
    samples = []
    for sft_path, meta_path in SOURCES:
        metas = {}
        with open(meta_path) as f:
            for line in f:
                m = json.loads(line)
                metas[m["problem_id"]] = m
        with open(sft_path) as f:
            for line in f:
                d = json.loads(line)
                if d["task"] != task_name:
                    continue
                d["_metadata"] = metas.get(d["id"])
                samples.append(d)
    return samples


def find_video_runs(input_ids, video_token_id):
    mask = (input_ids == video_token_id)
    idx = mask.nonzero(as_tuple=True)[0].tolist()
    runs = []
    if not idx:
        return runs
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            runs.append((start, prev + 1))
            start = i
        prev = i
    runs.append((start, prev + 1))
    return runs


def main():
    model_path, model_tag, gpu_ids, task_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else None

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    n_gpus = len(gpu_ids.split(","))
    device_map = "auto" if n_gpus > 1 else "cuda:0"

    out_dir = os.path.join(OUT_DIR, model_tag)
    os.makedirs(out_dir, exist_ok=True)

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    video_token_id = cfg.video_token_id
    text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    L = text_cfg.num_hidden_layers
    print(f"[{model_tag}] num_hidden_layers={L}, will extract all {L+1} hidden_states (0=embedding..{L}=final)")

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True, device_map=device_map
    ).eval()
    first_device = next(model.parameters()).device

    samples = load_samples(task_name)
    if limit:
        samples = samples[:limit]
    print(f"[{model_tag}] task={task_name} total samples: {len(samples)}")

    records = []
    n_ok, n_fail = 0, 0
    t0 = time.time()
    for si, d in enumerate(samples):
        try:
            user_content = d["messages"][1]["content"]
            video_path = None
            question_text = None
            for c in user_content:
                if c["type"] == "video":
                    video_path = c["video"]
                elif c["type"] == "text":
                    question_text = c["text"]
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "video", "video": video_path, "nframes": 16},
                    {"type": "text", "text": question_text},
                ]},
            ]
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
                out = model(**inputs, output_hidden_states=True, use_cache=False)

            runs = find_video_runs(inputs["input_ids"][0], video_token_id)

            result = {"task": task_name, "id": d["id"], "doc_id": d["doc_id"], "answer": d["messages"][2]["content"]}
            all_layers = torch.zeros(L + 1, text_cfg.hidden_size, dtype=torch.float32)
            for li in range(L + 1):
                hs = out.hidden_states[li][0]
                pooled = torch.stack([hs[s:e].mean(dim=0) for s, e in runs]).mean(dim=0)
                all_layers[li] = pooled.to(torch.float32).cpu()
            result["all_layers"] = all_layers  # (L+1, hidden_size)

            gt = d.get("_metadata")
            if gt is not None:
                result["ground_truth_details"] = gt.get("ground_truth_details")

            records.append(result)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[{model_tag}] FAILED {d.get('id')}: {e}")

        if (si + 1) % 50 == 0 or (si + 1) == len(samples):
            elapsed = time.time() - t0
            print(f"[{model_tag}] {task_name}: {si+1}/{len(samples)} ok={n_ok} fail={n_fail} elapsed={elapsed:.1f}s")

    out_path = os.path.join(out_dir, f"{task_name}.pt")
    torch.save(records, out_path)
    print(f"[{model_tag}] {task_name}: saved -> {out_path} (ok={n_ok} fail={n_fail})")


if __name__ == "__main__":
    main()
