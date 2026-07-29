#!/usr/bin/env python3
"""causal-verification-plan.md:对Qwen3.5-9B的8个full_attention层(0-indexed [3,7,11,15,19,23,27,31],
共32层里唯二的标准softmax attention层,其余24层是Gated DeltaNet线性注意力,不在本次范围)逐层逐head做
零消融/均值消融,测"4类读出层gap原语"(Rotation_Direction/Rotation_Count/Bouncing_Counting/
Acceleration_Identification)的直答准确率相对baseline掉多少,定位TRD/CAD该监督哪几层哪几个head。

实现要点(如实记录设计选择,不是拍脑袋):
- hook点:每层self_attn.o_proj的forward pre-hook。o_proj输入shape=(batch,seq,num_attention_heads
  *head_dim)=(1,seq,4096)(9B: num_attention_heads=16,head_dim=256),每个head占连续256列
  (head_idx*256:(head_idx+1)*256)。这是gate(sigmoid门控)已经乘过之后、真正进入输出投影前的
  最后一个"可以按head切开"的点,不需要碰self_attn.forward内部实现。
- "均值消融"的均值怎么定义:baseline pass(不消融,generate完整答案)期间,用同一个hook机制
  以"只记录不修改"模式跑一遍,对每个(layer,head)累积该head在所有生成步骤、所有当前task的
  样本上的激活值总和与计数,pass结束后算出一个(head_dim,)的全局均值向量。消融时不管当前
  生成到第几步、是哪条样本,统一替换成这个固定均值向量——这是"给定这个head存在,但不携带
  当前这个具体位置信息"的一个可操作实现,不是跨样本在同一个position对齐(视频内容/长度都
  不同,那样定义不出有意义的对齐)。
- 整层消融=该层16个head的输出(整个4096维)一起替换,不是循环16次单head替换的叠加。

用法:
  python3 causal_ablation.py <model_path> <gpu_ids> --limit_per_task=15 --modes=mean,zero \
      --layers=3,7,11,15,19,23,27,31 --heads=all
  # 只测特定几个head(比如stage2交叉验证阶段,只测stage1筛出来的候选):
  python3 causal_ablation.py <model_path> <gpu_ids> --limit_per_task=50 --modes=zero \
      --layers=15,23 --heads=2,7,9 --targets_only=15:2,15:7,23:9
"""
import sys
import os
import json
import time
import argparse
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

sys.path.insert(0, "/root/miniconda3/envs/lmmseval/lib/python3.12/site-packages")
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

PROBE_DATA_DIR = "/remote-home/ziyesong/videoPerception/probe_data"
GAP_TASKS = ["Rotation_Direction", "Rotation_Count", "Bouncing_Counting", "Acceleration_Identification"]
FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]  # 0-indexed,Qwen3.5-9B的layer_types确认过

SOURCES = [
    (os.path.join(PROBE_DATA_DIR, "Atomic", "sft.jsonl")),
]


def load_samples(limit_per_task):
    samples = []
    counts = {t: 0 for t in GAP_TASKS}
    with open(SOURCES[0]) as f:
        for line in f:
            d = json.loads(line)
            if d["task"] not in GAP_TASKS:
                continue
            if counts[d["task"]] >= limit_per_task:
                continue
            counts[d["task"]] += 1
            samples.append(d)
    return samples


class HeadHookManager:
    """管理单个o_proj forward-pre-hook,可以在"记录均值"和"消融"两种模式间切换。"""

    def __init__(self, model, head_dim, num_heads):
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.mode = "off"  # off | record | ablate
        self.record_sum = {}   # (layer_idx, head_idx) -> running sum tensor (head_dim,)
        self.record_count = {}
        self.mean_vectors = {}  # (layer_idx, head_idx) -> mean tensor (head_dim,)
        self.ablate_layer = None
        self.ablate_head = None   # None = 整层(全部head)
        self.ablate_value = None  # "zero" | "mean"
        self.handles = []

        for layer_idx in FULL_ATTENTION_LAYERS:
            attn = model.model.language_model.layers[layer_idx].self_attn
            h = attn.o_proj.register_forward_pre_hook(self._make_hook(layer_idx))
            self.handles.append(h)

    def _make_hook(self, layer_idx):
        def hook(module, args):
            x = args[0]  # (batch, seq, num_heads*head_dim)
            if self.mode == "record":
                with torch.no_grad():
                    xf = x.detach().to(torch.float32)
                    for h in range(self.num_heads):
                        key = (layer_idx, h)
                        seg = xf[..., h * self.head_dim:(h + 1) * self.head_dim].reshape(-1, self.head_dim)
                        s = seg.sum(dim=0)
                        c = seg.shape[0]
                        if key in self.record_sum:
                            self.record_sum[key] += s
                            self.record_count[key] += c
                        else:
                            self.record_sum[key] = s
                            self.record_count[key] = c
                return None
            elif self.mode == "ablate" and layer_idx == self.ablate_layer:
                x = x.clone()
                heads = range(self.num_heads) if self.ablate_head is None else [self.ablate_head]
                for h in heads:
                    if self.ablate_value == "zero":
                        x[..., h * self.head_dim:(h + 1) * self.head_dim] = 0.0
                    else:  # mean
                        mean_vec = self.mean_vectors[(layer_idx, h)].to(x.dtype).to(x.device)
                        x[..., h * self.head_dim:(h + 1) * self.head_dim] = mean_vec
                return (x,) + args[1:]
            return None
        return hook

    def finalize_means(self):
        for key in self.record_sum:
            self.mean_vectors[key] = self.record_sum[key] / self.record_count[key]

    def remove(self):
        for h in self.handles:
            h.remove()


def build_message(d):
    user_content = d["messages"][1]["content"]
    video_path, question_text = None, None
    for c in user_content:
        if c["type"] == "video":
            video_path = c["video"]
        elif c["type"] == "text":
            question_text = c["text"]
    return video_path, question_text, d["messages"][2]["content"]


def run_pass(model, processor, first_device, samples, hook_mgr, mode, ablate_layer=None, ablate_head=None, ablate_value=None):
    hook_mgr.mode = mode
    hook_mgr.ablate_layer = ablate_layer
    hook_mgr.ablate_head = ablate_head
    hook_mgr.ablate_value = ablate_value

    per_task_scores = {}
    for d in samples:
        task = d["task"]
        video_path, question_text, gt_letter = build_message(d)
        choices = [chr(65 + i) for i in range(6)]
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
            gen_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
        new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
        resp = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = extract_mcq_answer(resp, choices=choices)
        correct = bool(pred) and pred == gt_letter.upper()
        per_task_scores.setdefault(task, []).append(1.0 if correct else 0.0)

    return {t: 100 * sum(s) / len(s) for t, s in per_task_scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("gpu_ids")
    ap.add_argument("--limit_per_task", type=int, default=15)
    ap.add_argument("--modes", default="mean,zero")
    ap.add_argument("--layers", default=",".join(str(x) for x in FULL_ATTENTION_LAYERS))
    ap.add_argument("--heads", default="all")  # "all" | "whole" (只做整层) | 逗号分隔的head下标
    ap.add_argument("--out_tag", default="stage1")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    modes = args.modes.split(",")
    layers = [int(x) for x in args.layers.split(",")]
    do_whole_layer = True
    head_list = None
    if args.heads == "whole":
        head_list = []
    elif args.heads != "all":
        head_list = [int(x) for x in args.heads.split(",")]

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tc = cfg.text_config
    num_heads = tc.num_attention_heads
    head_dim = tc.head_dim
    print(f"num_attention_heads={num_heads} head_dim={head_dim} layers={layers} heads={'whole-only' if head_list == [] else (head_list or 'all-'+str(num_heads))}")

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    ).eval()
    first_device = next(model.parameters()).device

    samples = load_samples(args.limit_per_task)
    print(f"total samples: {len(samples)} ({args.limit_per_task}/task x {len(GAP_TASKS)} tasks)")

    hook_mgr = HeadHookManager(model, head_dim, num_heads)

    t0 = time.time()
    print("=== baseline pass (record activations for mean-ablation, no modification) ===")
    baseline_acc = run_pass(model, processor, first_device, samples, hook_mgr, mode="record")
    hook_mgr.finalize_means()
    print(f"baseline_acc: {baseline_acc}  elapsed={time.time()-t0:.1f}s")

    results = {"baseline": baseline_acc, "ablations": []}

    heads_to_test = head_list if head_list is not None else list(range(num_heads))

    for mode in modes:
        # 整层消融
        if do_whole_layer:
            for layer_idx in layers:
                t1 = time.time()
                acc = run_pass(model, processor, first_device, samples, hook_mgr, mode="ablate",
                                ablate_layer=layer_idx, ablate_head=None, ablate_value=mode)
                delta = {t: baseline_acc[t] - acc[t] for t in acc}
                print(f"[whole-layer] layer={layer_idx} mode={mode} acc={acc} delta={delta} ({time.time()-t1:.1f}s)")
                results["ablations"].append({"layer": layer_idx, "head": None, "mode": mode, "acc": acc, "delta": delta})

        # 单head消融
        if head_list != []:
            for layer_idx in layers:
                for head_idx in heads_to_test:
                    t1 = time.time()
                    acc = run_pass(model, processor, first_device, samples, hook_mgr, mode="ablate",
                                    ablate_layer=layer_idx, ablate_head=head_idx, ablate_value=mode)
                    delta = {t: baseline_acc[t] - acc[t] for t in acc}
                    print(f"[head] layer={layer_idx} head={head_idx} mode={mode} acc={acc} delta={delta} ({time.time()-t1:.1f}s)")
                    results["ablations"].append({"layer": layer_idx, "head": head_idx, "mode": mode, "acc": acc, "delta": delta})

    hook_mgr.remove()

    out_path = os.path.join(PROBE_DATA_DIR, f"causal_ablation_{args.out_tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out_path}  total_elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
