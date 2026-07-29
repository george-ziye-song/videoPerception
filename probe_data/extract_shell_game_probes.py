#!/usr/bin/env python3
"""distillation-readiness-experiments.md 实验2试点:basic_shell_game的"遮挡前/遮挡后"
多时间点hidden state抽取。

方法论关键点(和01/02的§4.1不同,这是PSD要求的核心差异):不是在完整视频的最后一刻抽一次
hidden state,而是用qwen_vl_utils原生支持的video_start/video_end(秒)把同一条视频截成
两段独立输入喂给模型:
  - "before": [0, T_before] —— T_before取initial_reveal事件的中点,此时球的位置是画面里
    直接可见的信息,probe预测"起始位置"应该接近100%(这是方法有效性的sanity check上限)。
  - "after": [0, T_after] —— T_after取最后一个swap事件的end_ms,此时球已经被完全遮挡
    (整个shuffle阶段容器不透明,球从不可见),probe预测"最终位置"(=真实答案)——这才是
    PSD要回答的核心问题:模型内部还记不记得。
两段是完全独立的forward,不共享KV cache,和extract_hidden_states.py用同一套
find_video_runs()连续run检测pooling逻辑(3层:浅/中/深)。

用法: python3 extract_shell_game_probes.py <model_path> <model_tag> <gpu_ids> [limit]
"""
import sys
import os
import json
import time
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

PROBE_DATA_DIR = "/remote-home/ziyesong/videoPerception/probe_data"
SG_DIR = os.path.join(PROBE_DATA_DIR, "ShellGame")
OUT_DIR = os.path.join(PROBE_DATA_DIR, "hidden_states_shellgame")
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
            pid = d["extra"]["id"]
            d["_metadata"] = metas.get(pid)
            samples.append(d)
    return samples


def get_timepoints(meta):
    tl = meta["ground_truth_details"]["video_events_timeline_ms"]
    initial = tl[0]
    assert initial["event_type"] == "initial_reveal"
    t_before_s = (initial["start_ms"] + initial["end_ms"]) / 2 / 1000.0
    swap_events = [e for e in tl if e["event_type"] in ("swap", "magic_swap")]
    t_after_s = swap_events[-1]["end_ms"] / 1000.0
    return t_before_s, t_after_s


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


def forward_and_pool(model, processor, video_token_id, video_path, video_end_s, question_text, layers, first_device):
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "nframes": NFRAMES, "video_start": 0.0, "video_end": video_end_s},
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
    result = {}
    for layer_name, li in layers:
        hs = out.hidden_states[li][0]
        pooled = torch.stack([hs[s:e].mean(dim=0) for s, e in runs]).mean(dim=0).to(torch.float32).cpu()
        result[layer_name] = pooled
    result["num_temporal_groups"] = len(runs)
    return result


def main():
    model_path, model_tag, gpu_ids = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    n_gpus = len(gpu_ids.split(","))
    device_map = "auto" if n_gpus > 1 else "cuda:0"

    out_dir = os.path.join(OUT_DIR, model_tag)
    os.makedirs(out_dir, exist_ok=True)

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    video_token_id = cfg.video_token_id
    text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    L = text_cfg.num_hidden_layers
    layers = [("shallow", round(0.25 * L)), ("mid", round(0.5 * L)), ("deep", round(0.9 * L))]
    print(f"[{model_tag}] num_hidden_layers={L} -> {layers}")

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True, device_map=device_map
    ).eval()
    first_device = next(model.parameters()).device

    samples = load_samples()
    if limit:
        samples = samples[:limit]
    print(f"[{model_tag}] total samples: {len(samples)}")

    for stage in ["before", "after"]:
        out_path = os.path.join(out_dir, f"{stage}.pt")
        if os.path.exists(out_path):
            print(f"[{model_tag}] {stage}: 已有checkpoint,跳过(断点续跑)")
            continue

        records = []
        n_ok, n_fail = 0, 0
        t0 = time.time()
        for si, d in enumerate(samples):
            meta = d["_metadata"]
            try:
                t_before_s, t_after_s = get_timepoints(meta)
                video_end = t_before_s if stage == "before" else t_after_s
                video_path = d["extra"]["original_video_path"]
                question_text = meta["problem"].replace("<video>\n", "")

                res = forward_and_pool(model, processor, video_token_id, video_path, video_end, question_text, layers, first_device)
                res["id"] = d["extra"]["id"]
                initial_pos = list(meta["ground_truth_details"]["initial_state"].keys())[0]
                final_pos = meta["ground_truth_details"]["question_data"]["correct_answer_value"]
                res["initial_pos"] = initial_pos
                res["final_pos"] = final_pos
                res["answer_letter"] = meta["answer"]
                records.append(res)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"[{model_tag}] {stage} FAILED {d.get('extra', {}).get('id')}: {e}")

            if (si + 1) % 50 == 0 or (si + 1) == len(samples):
                elapsed = time.time() - t0
                print(f"[{model_tag}] {stage}: {si+1}/{len(samples)} ok={n_ok} fail={n_fail} elapsed={elapsed:.1f}s")

        torch.save(records, out_path)
        print(f"[{model_tag}] {stage}: saved -> {out_path} (ok={n_ok} fail={n_fail})")


if __name__ == "__main__":
    main()
