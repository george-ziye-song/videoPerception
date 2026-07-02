#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Qwen3-VL-8B evaluation harness (transformers backend, single-GPU shard).
vLLM 0.8.5 does NOT support qwen3_vl -> use transformers Qwen3VLForConditionalGeneration.
Run 4 shards (one per GPU) for data-parallel, then merge + score.

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python run_eval.py --benchmark mvbench \
      --num-shards 4 --shard 0 --out results/mvbench.shard0.jsonl
"""
import os, sys, re, json, glob, argparse, traceback
import pandas as pd

MODEL_PATH = "/remote-home/ziyesong/models/Qwen3-VL-8B-Instruct"
MV_ROOT    = "/root/benchmarks/MVBench_video"
MV_JSON    = "/remote-home/ziyesong/videoPerception/data/benchmarks/MVBench/json"
TB_ROOT    = "/root/benchmarks/TemporalBench"
TB_DATA    = "/root/benchmarks/TemporalBench/data"
TOMATO_LMMS = os.environ.get("TOMATO_LMMS", "/root/benchmarks/TOMATO_lmms")  # lmms-lab/TOMATO: parquet has video_path

NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "16"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(256 * 28 * 28)))  # ~448px/frame, ~1500 tok/16frames
LETTERS = "ABCDEFGH"
MCQ_INSTR = "Answer with the option's letter from the given choices directly. Respond with only the single capital letter of the correct option."

# ---------------- MVBench task -> (video subdir prefix, data_type, has_bound) --------------
MV_TASKS = {
    "action_sequence":         ("star/Charades_v1_480/",      "video", True),
    "action_prediction":       ("star/Charades_v1_480/",      "video", True),
    "action_antonym":          ("ssv2_video/",                "video", False),
    "fine_grained_action":     ("Moments_in_Time_Raw/videos/","video", False),
    "unexpected_action":       ("FunQA_test/test/",           "video", False),
    "object_existence":        ("clevrer/video_validation/",  "video", False),
    "object_interaction":      ("star/Charades_v1_480/",      "video", True),
    "object_shuffle":          ("perception/videos/",         "video", False),
    "moving_direction":        ("clevrer/video_validation/",  "video", False),
    "action_localization":     ("sta/sta_video/",             "video", True),
    "scene_transition":        ("scene_qa/video/",            "video", False),
    "action_count":            ("perception/videos/",         "video", False),
    "moving_count":            ("clevrer/video_validation/",  "video", False),
    "moving_attribute":        ("clevrer/video_validation/",  "video", False),
    "state_change":            ("perception/videos/",         "video", False),
    "fine_grained_pose":       ("nturgbd/",                   "video", False),  # NTU videos NOT downloaded -> skipped
    "character_order":         ("perception/videos/",         "video", False),
    "egocentric_navigation":   ("vlnqa/",                     "video", False),
    "episodic_reasoning":      ("tvqa/frames_fps3_hq/",       "frame", True),
    "counterfactual_inference":("clevrer/video_validation/",  "video", False),
}
MV_SKIP = {"fine_grained_pose"}  # missing nturgbd videos


def sample_frames(folder, n):
    imgs = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    if not imgs:
        return None
    if len(imgs) <= n:
        return imgs
    step = len(imgs) / n
    return [imgs[min(len(imgs) - 1, int(i * step))] for i in range(n)]


def build_mvbench():
    samples = []
    for task, (prefix, dtype, _bound) in MV_TASKS.items():
        if task in MV_SKIP:
            continue
        jp = os.path.join(MV_JSON, f"{task}.json")
        data = json.load(open(jp))
        for i, item in enumerate(data):
            cands = item["candidates"]
            try:
                gt_idx = cands.index(item["answer"])
            except ValueError:
                continue
            opt_txt = "\n".join(f"({LETTERS[j]}) {c}" for j, c in enumerate(cands))
            prompt = f"{item['question']}\nOptions:\n{opt_txt}\n{MCQ_INSTR}"
            if dtype == "frame":
                vpath = os.path.join(MV_ROOT, prefix, item["video"])
                media = ("frame", vpath)
            else:
                vpath = os.path.join(MV_ROOT, prefix, item["video"])
                media = ("video", vpath)
            samples.append(dict(uid=f"mvbench/{task}/{i}", benchmark="mvbench", task=task,
                                media=media, prompt=prompt, gt=LETTERS[gt_idx],
                                nopts=len(cands), group=None))
    return samples


def build_temporalbench(split):
    # split in {short, long}
    fn = "test_short_qa" if split == "short" else "test_long_qa"
    df = pd.read_parquet(os.path.join(TB_DATA, f"{fn}-00000-of-00001.parquet"))
    samples = []
    for _, r in df.iterrows():
        vpath = os.path.join(TB_ROOT, r["video_name"])
        prompt = f"{r['question']}\n{MCQ_INSTR}"
        samples.append(dict(uid=f"temporalbench_{split}/{r['idx']}", benchmark=f"temporalbench_{split}",
                            task=str(r.get("dataset", "")), media=("video", vpath), prompt=prompt,
                            gt=str(r["GT"]).strip(), nopts=None, group=str(r["video_name"])))
    return samples


def build_tomato():
    # lmms-lab/TOMATO parquet: id, question, options, answer(int), video_path, reason_type, demonstration_type
    df = pd.read_parquet(os.path.join(TOMATO_LMMS, "data", "test-00000-of-00001.parquet"))
    samples = []
    for _, r in df.iterrows():
        opts = list(r["options"])
        gt = LETTERS[int(r["answer"])]
        opt_txt = "\n".join(f"({LETTERS[j]}) {o}" for j, o in enumerate(opts))
        prompt = f"{r['question']}\nOptions:\n{opt_txt}\n{MCQ_INSTR}"
        vpath = os.path.join(TOMATO_LMMS, r["video_path"])
        samples.append(dict(uid=f"tomato/{r['reason_type']}/{r['id']}", benchmark="tomato",
                            task=str(r["reason_type"]), media=("video", vpath), prompt=prompt,
                            gt=gt, nopts=len(opts), group=None))
    return samples


def build_manifest(benchmark):
    if benchmark == "mvbench":
        return build_mvbench()
    if benchmark == "temporalbench_short":
        return build_temporalbench("short")
    if benchmark == "temporalbench_long":
        return build_temporalbench("long")
    if benchmark == "tomato":
        return build_tomato()
    raise ValueError(benchmark)


# ---------------- answer parsing ----------------
def parse_letter(text, nopts=None):
    if not text:
        return None
    hi = LETTERS[(nopts - 1)] if nopts else "H"
    rng = f"A-{hi}"
    t = text.strip()
    m = re.match(rf"^\(?\s*([{rng}])\b", t)
    if m:
        return m.group(1)
    m = re.search(rf"answer\s*(?:is|:)?\s*\(?\s*([{rng}])\b", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(rf"\b([{rng}])\b", t)
    if m:
        return m.group(1)
    m = re.search(rf"([{rng}])", t)
    return m.group(1) if m else None


# ---------------- media -> qwen message ----------------
def media_to_content(media):
    kind, path = media
    if kind == "frame":
        frames = sample_frames(path, NUM_FRAMES)
        if not frames:
            return None
        return {"type": "video", "video": frames, "max_pixels": MAX_PIXELS}
    return {"type": "video", "video": path, "nframes": NUM_FRAMES, "max_pixels": MAX_PIXELS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    manifest = build_manifest(args.benchmark)
    manifest = manifest[args.shard::args.num_shards]
    if args.limit:
        manifest = manifest[:args.limit]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    manifest = [m for m in manifest if m["uid"] not in done]
    print(f"[shard {args.shard}/{args.num_shards}] {args.benchmark}: {len(manifest)} to do "
          f"({len(done)} already done)", flush=True)
    if not manifest:
        return

    print("loading model...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        device_map="cuda:0")
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = "left"

    fout = open(args.out, "a", buffering=1)

    def run_batch(batch):
        msgs_list, valid = [], []
        for s in batch:
            content = media_to_content(s["media"])
            if content is None:
                fout.write(json.dumps(dict(uid=s["uid"], benchmark=s["benchmark"], task=s["task"],
                    gt=s["gt"], pred=None, correct=False, raw="MEDIA_MISSING", group=s["group"]),
                    ensure_ascii=False) + "\n")
                continue
            msgs_list.append([{"role": "user", "content": [content, {"type": "text", "text": s["prompt"]}]}])
            valid.append(s)
        if not valid:
            return
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_list]
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            msgs_list, return_video_kwargs=True, return_video_metadata=True)
        vt = vm = None
        if video_inputs is not None:
            if all(isinstance(v, tuple) and len(v) == 2 for v in video_inputs):
                vt = [v[0] for v in video_inputs]
                vm = [v[1] for v in video_inputs]  # metadata -> correct <t seconds> timestamps
            else:
                vt = [v[0] if isinstance(v, tuple) else v for v in video_inputs]
        pkw = dict(text=texts, images=image_inputs, videos=vt,
                   return_tensors="pt", padding=True, **video_kwargs)
        if vm is not None:
            pkw["video_metadata"] = vm
        inputs = processor(**pkw)
        inputs = inputs.to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        outs = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for s, raw in zip(valid, outs):
            pred = parse_letter(raw, s["nopts"])
            fout.write(json.dumps(dict(uid=s["uid"], benchmark=s["benchmark"], task=s["task"],
                gt=s["gt"], pred=pred, correct=(pred == s["gt"]), raw=raw.strip()[:200],
                group=s["group"]), ensure_ascii=False) + "\n")

    bs = args.batch_size
    i = 0
    n = len(manifest)
    while i < n:
        batch = manifest[i:i + bs]
        try:
            run_batch(batch)
        except Exception as e:
            # fall back to per-sample to isolate the bad one / OOM
            torch.cuda.empty_cache()
            for s in batch:
                try:
                    run_batch([s])
                except Exception as e2:
                    torch.cuda.empty_cache()
                    fout.write(json.dumps(dict(uid=s["uid"], benchmark=s["benchmark"], task=s["task"],
                        gt=s["gt"], pred=None, correct=False, raw=f"ERR:{repr(e2)[:120]}",
                        group=s["group"]), ensure_ascii=False) + "\n")
        i += bs
        if (i // bs) % 10 == 0:
            print(f"[shard {args.shard}] {min(i,n)}/{n}", flush=True)
    fout.close()
    print(f"[shard {args.shard}] DONE {args.benchmark}", flush=True)


if __name__ == "__main__":
    main()
