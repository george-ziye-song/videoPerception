#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对FINAL_TRAINING_DATASET.json做样本级pass-rate过滤:
teacher(Qwen3.5-9B)和student(Qwen3.5-4B)各自直接作答(enable_thinking=False,只要求一个字母),
和ground truth比对,标注每条的teacher_correct/student_correct。
用法: CUDA_VISIBLE_DEVICES=0,1,2,3 python passrate_filter.py --role teacher --tp 4 --limit 8
"""
import os, re, json, argparse

import sys
_role_for_cache = None
for _i, _a in enumerate(sys.argv):
    if _a == "--role" and _i + 1 < len(sys.argv):
        _role_for_cache = sys.argv[_i + 1]
if _role_for_cache:
    os.environ["VLLM_CACHE_ROOT"] = f"/root/.cache/vllm_{_role_for_cache}"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/root/.cache/torchinductor_{_role_for_cache}"

from PIL import Image

MAX_PIXELS = 256 * 28 * 28  # 和 run_eval.py 同一惯例(~448px/帧)


def resize_for_vlm(img):
    w, h = img.size
    if w * h <= MAX_PIXELS:
        return img
    scale = (MAX_PIXELS / (w * h)) ** 0.5
    new_w, new_h = max(28, int(w * scale)), max(28, int(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)

DATASET_PATH = "/remote-home/ziyesong/videoPerception/FINAL_TRAINING_DATASET.json"
MODEL_PATHS = {
    "student": "/root/models/Qwen3.5-4B",
    "teacher": "/root/models/Qwen3.5-9B",
}
MCQ_INSTR = "Answer with the option's letter from the given choices directly. Respond with only the single capital letter of the correct option, nothing else."


def parse_letter(text, valid_letters):
    text = text.strip()
    # 1. 纯字母或"A."这种最干净的情况
    m = re.match(r'^\(?([A-H])\)?[\.\)]?\s*$', text)
    if m and m.group(1) in valid_letters:
        return m.group(1)
    # 2. \boxed{A} 或 <answer>A</answer>
    m = re.search(r'\\boxed\{([A-H])\}|<answer>\s*\(?([A-H])\)?\s*</answer>', text)
    if m:
        letter = m.group(1) or m.group(2)
        if letter in valid_letters:
            return letter
    # 3. "answer is A" / "answer: A" 这种明确声明,取最后一个匹配(通常是最终结论)
    matches = re.findall(r'answer\s*(?:is|:)\s*\(?([A-H])\)?', text, re.I)
    if matches and matches[-1] in valid_letters:
        return matches[-1]
    # 4. 文本第一个字符就是选项字母,且紧跟标点(常见于"A. xxx"这种直接给了完整选项的回答)
    m = re.match(r'^([A-H])[\.\):]', text)
    if m and m.group(1) in valid_letters:
        return m.group(1)
    # 找不到就诚实返回None,不猜
    return None


def extract_gt_letter(answer_field):
    m = re.match(r'^([A-H])[\.\)]', answer_field.strip())
    return m.group(1) if m else None


def build_prompt(entry):
    opt_lines = "\n".join(entry["options"])
    return f"{entry['question']}\n{opt_lines}\n{MCQ_INSTR}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["student", "teacher"], required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    data = json.load(open(DATASET_PATH))
    if args.num_shards > 1:
        data = data[args.shard::args.num_shards]
    if args.limit:
        data = data[:args.limit]
    print(f"[{args.role}] 待处理样本数: {len(data)}", flush=True)

    model_path = MODEL_PATHS[args.role]
    import time
    t_load0 = time.time()
    llm = LLM(model=model_path, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util,
              trust_remote_code=True, max_model_len=8192, limit_mm_per_prompt={"image": 16})
    print(f"[{args.role}] 模型加载耗时: {time.time()-t_load0:.1f}s", flush=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=16)
    chat_kwargs = dict(chat_template_kwargs={"enable_thinking": False})

    out_path = args.out or f"/remote-home/ziyesong/videoPerception/eval/passrate_{args.role}_shard{args.shard}.json"
    CHUNK = int(os.environ.get("PR_CHUNK", "50"))
    results = []
    n_correct, n_unparsed = 0, 0
    n_prev = 0
    if os.path.exists(out_path):
        results = json.load(open(out_path))
        done_ids = {r["id"] for r in results}
        n_correct = sum(1 for r in results if r["correct"])
        n_unparsed = sum(1 for r in results if r["pred"] is None)
        n_prev = len(results)
        before = len(data)
        data = [e for e in data if e["id"] not in done_ids]
        print(f"[{args.role}] 断点续跑: 已完成{n_prev}条, 剩余{len(data)}/{before}条", flush=True)
    total_all = n_prev + len(data)
    t_start = time.time()
    for ci in range(0, len(data), CHUNK):
        chunk = data[ci:ci + CHUNK]
        messages_list = []
        for e in chunk:
            imgs = [resize_for_vlm(Image.open(fp).convert("RGB")) for fp in e["frames"]]
            content = [{"type": "image_pil", "image_pil": img} for img in imgs]
            content.append({"type": "text", "text": build_prompt(e)})
            messages_list.append([{"role": "user", "content": content}])
        t_chunk0 = time.time()
        outputs = llm.chat(messages_list, sampling_params=sampling, **chat_kwargs)
        chunk_dt = time.time() - t_chunk0
        for e, out in zip(chunk, outputs):
            raw = out.outputs[0].text
            valid_letters = set(o.strip()[0] for o in e["options"])
            pred = parse_letter(raw, valid_letters)
            gt = extract_gt_letter(e["answer"])
            correct = (pred == gt) if (pred and gt) else False
            if pred is None:
                n_unparsed += 1
            if correct:
                n_correct += 1
            results.append({"id": e["id"], "raw_output": raw, "pred": pred, "gt": gt, "correct": correct})
        done_total = len(results)
        new_done = done_total - n_prev
        elapsed = time.time() - t_start
        rate = new_done / elapsed if elapsed > 0 else 0
        eta_min = (len(data) - new_done) / rate / 60 if rate > 0 else -1
        print(f"[{args.role}] 累计{done_total}/{total_all} (本次新跑{new_done}/{len(data)}, 本chunk{chunk_dt:.1f}s), "
              f"正确率{n_correct/done_total*100:.1f}%, 速度{rate:.2f}条/s, 预计剩余{eta_min:.1f}分钟", flush=True)
        json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)

    print(f"[{args.role}] 最终: 正确 {n_correct}/{total_all} ({n_correct/total_all*100:.1f}%), 解析失败: {n_unparsed}", flush=True)
    print(f"已写入 {out_path}", flush=True)


if __name__ == "__main__":
    main()
