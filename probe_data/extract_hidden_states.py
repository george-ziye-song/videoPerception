#!/usr/bin/env python3
"""probe-experiment.md §4.1:冻结前向,抽取每条合成样本在3层(浅/中/深)的hidden state,
按视频temporal-group(=temporal_patch_size=2帧合并后的组,不是原始单帧)做mean pooling。

关键实现细节(2026-07-12 用test_extract_one.py验证过,不是猜的):
- Qwen3.5的video token不是一整块连续区域——每隔256个video token(=32x32 patch grid
  经spatial_merge_size=2下采样后每组的token数)会插入一段形如
  `<|vision_end|><X.Y seconds><|vision_start|>`的时间戳文本token,把video token切成
  T段(T=nframes/temporal_patch_size)。必须用"连续run检测",不能假设固定stride切片,
  否则pooling会把时间戳文本的embedding也混进去。
- 直答协议(enable_thinking=False,不加reasoning_prompt),只做到最后一个输入token的
  forward,不生成——所以gen_kwargs/sampling都不需要。

用法: python3 extract_hidden_states.py <model_path> <model_tag> <gpu_ids> <nframes> [limit]
例:   python3 extract_hidden_states.py /root/models/Qwen3.5-9B 9b 4 16
      python3 extract_hidden_states.py /root/models/Qwen3.5-35B-A3B 35b_35 4,5,6,7 16
"""
import sys
import os
import json
import glob
import time
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

PROBE_DATA_DIR = "/remote-home/ziyesong/videoPerception/probe_data"
OUT_DIR = os.path.join(PROBE_DATA_DIR, "hidden_states")

SOURCES = [
    (os.path.join(PROBE_DATA_DIR, "Atomic", "sft.jsonl"), os.path.join(PROBE_DATA_DIR, "Atomic", "metadata.jsonl")),
    (os.path.join(PROBE_DATA_DIR, "Atomic2", "sft.jsonl"), os.path.join(PROBE_DATA_DIR, "Atomic2", "metadata.jsonl")),
]


def load_samples():
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
                d["_metadata"] = metas.get(d["id"])
                samples.append(d)
    return samples


def group_by_task(samples):
    by_task = {}
    for d in samples:
        by_task.setdefault(d["task"], []).append(d)
    return by_task


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
    model_path, model_tag, gpu_ids, nframes = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
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
    shallow, mid, deep = round(0.25 * L), round(0.5 * L), round(0.9 * L)
    print(f"[{model_tag}] num_hidden_layers={L} -> layers picked: shallow={shallow} mid={mid} deep={deep}")

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True, device_map=device_map
    ).eval()
    first_device = next(model.parameters()).device

    samples = load_samples()
    if limit:
        samples = samples[:limit]
    by_task_all = group_by_task(samples)
    print(f"[{model_tag}] total samples to process: {len(samples)} across {len(by_task_all)} task types")

    n_ok_total, n_fail_total = 0, 0
    t0 = time.time()
    for task, task_samples in by_task_all.items():
        out_path = os.path.join(out_dir, f"{task}.pt")
        if os.path.exists(out_path):
            print(f"[{model_tag}] {task}: 已有checkpoint({out_path}),跳过(断点续跑)")
            continue

        records = []
        n_ok, n_fail = 0, 0
        for si, d in enumerate(task_samples):
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
                        {"type": "video", "video": video_path, "nframes": nframes},
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
                T = len(runs)

                result = {"task": task, "id": d["id"], "doc_id": d["doc_id"], "answer": d["messages"][2]["content"], "num_temporal_groups": T}
                for layer_name, li in [("shallow", shallow), ("mid", mid), ("deep", deep)]:
                    hs = out.hidden_states[li][0]
                    pooled = torch.stack([hs[s:e].mean(dim=0) for s, e in runs]).to(torch.float32).cpu()
                    result[layer_name] = pooled

                gt = d.get("_metadata")
                if gt is not None:
                    result["ground_truth_details"] = gt.get("ground_truth_details")

                records.append(result)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"[{model_tag}] FAILED sample {d.get('id')}: {e}")

            if (si + 1) % 100 == 0 or (si + 1) == len(task_samples):
                elapsed = time.time() - t0
                print(f"[{model_tag}] {task}: {si+1}/{len(task_samples)} done, ok={n_ok} fail={n_fail}, elapsed={elapsed:.1f}s")

        torch.save(records, out_path)
        print(f"[{model_tag}] {task}: checkpoint保存 -> {out_path} (ok={n_ok} fail={n_fail})")
        n_ok_total += n_ok
        n_fail_total += n_fail

    print(f"[{model_tag}] DONE. ok={n_ok_total} fail={n_fail_total}. saved to {out_dir}")


if __name__ == "__main__":
    main()
