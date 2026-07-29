#!/usr/bin/env python3
"""opsd-r-implementation-plan.md的工程落地:对单个原语、单个beta做一次OPSD-R训练+评估。

复用清单(严格照抄计划§2,不重新计算):
- h^enc: probe_data/hidden_states/9b/{task}.pt (2026-07-12已抽好,shallow/mid/deep三层,
  每条record是(T,4096),按temporal group mean pooling后再对T做mean得到h^enc目标)。
- 70/30切分:严格复刻train_probes.py的train_one_probe()逻辑(先按WHOLE_VIDEO_LABEL_FNS
  的标签丢弃singleton类,再对剩下的做train_test_split(test_size=0.3,random_state=0,
  stratify=y)),这样测试集才能和已发表的probe/直答baseline数字对齐,不是另起一套切分。
- prompt构造:逐行照抄extract_hidden_states.py/direct_answer_baseline.py。
- 答案解析:共享的lmms_eval.tasks._task_utils.mcq_extract.extract_mcq_answer。

h^gen怎么拿(计划§3.2的工程简化,不需要调用generate()):
- 输入=完整prompt(不含答案),做一次output_hidden_states=True的forward,最后一个位置
  (=模型即将生成第一个答案token的那个位置)的hidden state就是h^gen;这同一次forward的
  最后位置logits,直接拿来算CE(因为已验证过这4类gap原语的答案本质上是1个token)。

LoRA:peft的target_modules="all-linear"(自动发现全部nn.Linear,含q/k/v/o、mlp、以及24层
linear_attention自己的in_proj/out_proj等,peft自动排除lm_head)——比计划字面写的"q/k/v/o+
mlp gate/up/down"更宽,但是它的严格超集,用meta-device dry run验证过不会崩溃,不需要为这个
非标准的hybrid attention架构手写模块名正则。

用法:
  python3 train_opsd_r.py <model_path> <gpu_id> <task> <beta> <out_tag> \
      [--epochs=5] [--lr=2e-5] [--lora_r=16] [--no_projection] [--eval_max_new_tokens=16]
  task=Complex_Direction_Identification 或 Event_Sequence 时是§5.3回归检查,只支持beta=0
  (不需要h^enc,跳过OPSD-R loss本身)。
"""
import sys
import os
import json
import time
import argparse
import collections
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "/root/miniconda3/envs/lmmseval/lib/python3.12/site-packages")
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_probes import WHOLE_VIDEO_LABEL_FNS, TASK_SUBDIR  # noqa: E402

PROBE_DATA_DIR = "/remote-home/ziyesong/videoPerception/probe_data"
HIDDEN_STATES_DIR = os.path.join(PROBE_DATA_DIR, "hidden_states", "9b")
RUNS_DIR = os.path.join(PROBE_DATA_DIR, "opsd_r_runs")

GAP_TASKS = ["Rotation_Direction", "Rotation_Count", "Bouncing_Counting", "Acceleration_Identification"]
LAYER_FOR_TASK = {
    "Rotation_Direction": "shallow",
    "Rotation_Count": "mid",
    "Bouncing_Counting": "mid",
    "Acceleration_Identification": "mid",
}
CHOICES = [chr(65 + i) for i in range(6)]  # generous superset, matches direct_answer_baseline.py


def load_sft_samples(task):
    samples = []
    for subdir in ("Atomic", "Atomic2"):
        path = os.path.join(PROBE_DATA_DIR, subdir, "sft.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d["task"] == task:
                    samples.append(d)
    return samples


def get_split_indices(records, task):
    """严格复刻train_probes.py.train_one_probe()的singleton丢弃+stratified split逻辑。
    Event_Sequence(不在WHOLE_VIDEO_LABEL_FNS里)、或者records本身没有ground_truth_details
    (=直接传sft.jsonl的原始record,只在beta=0回归检查用§5.3两个task时发生,那两个task不需要
    严格复刻probe切分,只是个健全性检查)时,退化成普通(非stratified)切分。"""
    try:
        if task not in WHOLE_VIDEO_LABEL_FNS:
            raise KeyError(f"{task} not in WHOLE_VIDEO_LABEL_FNS")
        y_raw = [WHOLE_VIDEO_LABEL_FNS[task](r) for r in records]
    except KeyError:
        idx = list(range(len(records)))
        train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=0)
        return train_idx, test_idx

    raw_counts = collections.Counter(y_raw)
    dropped = {k: v for k, v in raw_counts.items() if v < 2}
    keep_idx = [i for i in range(len(records)) if y_raw[i] not in dropped]
    y_keep = [y_raw[i] for i in keep_idx]
    le = LabelEncoder()
    y_enc = le.fit_transform(y_keep)
    train_pos, test_pos = train_test_split(
        list(range(len(keep_idx))), test_size=0.3, random_state=0, stratify=y_enc
    )
    train_idx = [keep_idx[p] for p in train_pos]
    test_idx = [keep_idx[p] for p in test_pos]
    return train_idx, test_idx


def build_message_inputs(processor, sft_record, first_device):
    user_content = sft_record["messages"][1]["content"]
    video_path, question_text = None, None
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
    return {k: (v.to(first_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def get_answer_token_id(processor, letter):
    ids = processor.tokenizer(letter, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        print(f"[WARN] answer letter '{letter}' tokenized to {len(ids)} tokens (expected 1): {ids}, using first")
    return ids[0]


@torch.no_grad()
def eval_accuracy(model, processor, first_device, sft_by_id, eval_ids, max_new_tokens):
    correct, total = 0, 0
    for sid in eval_ids:
        d = sft_by_id[sid]
        gt_letter = d["messages"][2]["content"].strip().upper()
        inputs = build_message_inputs(processor, d, first_device)
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
        resp = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = extract_mcq_answer(resp, choices=CHOICES)
        total += 1
        if pred and pred == gt_letter:
            correct += 1
    return 100.0 * correct / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("gpu_id")
    ap.add_argument("task")
    ap.add_argument("beta", type=float)
    ap.add_argument("out_tag")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--no_projection", action="store_true")
    ap.add_argument("--eval_max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    use_opsdr = args.beta > 0
    use_projection = use_opsdr and not args.no_projection

    run_dir = os.path.join(RUNS_DIR, args.task, args.out_tag)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"=== task={args.task} beta={args.beta} projection={use_projection} out_tag={args.out_tag} gpu={args.gpu_id} ===")

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    L = text_cfg.num_hidden_layers
    layer_idx_map = {"shallow": round(0.25 * L), "mid": round(0.5 * L), "deep": round(0.9 * L)}
    layer_name = LAYER_FOR_TASK.get(args.task, "mid")
    layer_idx = layer_idx_map[layer_name]
    log(f"L={L} layer_name={layer_name} layer_idx={layer_idx}")

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0"
    )
    first_device = next(model.parameters()).device

    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                           target_modules="all-linear", bias="none")
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    P_theta = None
    if use_projection:
        P_theta = torch.nn.Linear(4096, 4096, bias=True)
        with torch.no_grad():
            P_theta.weight.copy_(torch.eye(4096) + 0.01 * torch.randn(4096, 4096))
            P_theta.bias.zero_()
        P_theta = P_theta.to(first_device, dtype=torch.float32)
        P_theta.train()

    # --- 数据:sft.jsonl(prompt构造) + hidden_states/9b/{task}.pt(h^enc,仅beta>0需要) ---
    sft_samples = load_sft_samples(args.task)
    sft_by_id = {d["id"]: d for d in sft_samples}
    log(f"sft samples for task {args.task}: {len(sft_samples)}")

    h_enc_by_id = {}
    if use_opsdr:
        # 注意:hidden_states/9b/{task}.pt里record的顺序不等于sft.jsonl的文件顺序
        # (.pt由早期多进程版本的extract_hidden_states.py生成,record顺序=多进程完成顺序/
        # dict插入顺序,不是文件顺序——2026-07-19本次dry-run现场验证过,不能假设按位置对齐,
        # 必须按id显式对齐)。get_split_indices仍然对records列表按其自身顺序做
        # stratified split(这就是train_probes.py当时训probe用的顺序/切分本身),
        # 只是之后统一用id去两边分别取值,不依赖两个列表的位置一致。
        pt_path = os.path.join(HIDDEN_STATES_DIR, f"{args.task}.pt")
        records = torch.load(pt_path, weights_only=False)
        assert len(records) == len(sft_samples), f"records={len(records)} vs sft={len(sft_samples)} mismatch for {args.task}"
        missing = [rec["id"] for rec in records if rec["id"] not in sft_by_id]
        assert not missing, f"{len(missing)} record ids not found in sft samples for {args.task}: {missing[:5]}"
        for rec in records:
            h_enc_by_id[rec["id"]] = rec[layer_name].mean(dim=0).to(torch.float32)  # (4096,)
        train_idx, test_idx = get_split_indices(records, args.task)
        id_list = [r["id"] for r in records]
    else:
        train_idx, test_idx = get_split_indices(sft_samples, args.task)
        id_list = [d["id"] for d in sft_samples]

    train_ids = [id_list[i] for i in train_idx]
    test_ids = [id_list[i] for i in test_idx]
    log(f"split: n_train={len(train_ids)} n_test={len(test_ids)}")

    params = [p for p in model.parameters() if p.requires_grad]
    if P_theta is not None:
        params += list(P_theta.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    log("=== eval before training (base + LoRA-init, i.e. ~baseline) ===")
    model.eval()
    acc_pre = eval_accuracy(model, processor, first_device, sft_by_id, test_ids, args.eval_max_new_tokens)
    log(f"acc_pre_training = {acc_pre:.2f}%")
    model.train()

    history = {"acc_pre_training": acc_pre, "epochs": [], "diag_curve": []}
    best_acc, best_epoch = -1, -1
    import random
    rng = random.Random(0)

    for epoch in range(args.epochs):
        t0 = time.time()
        order = train_ids[:]
        rng.shuffle(order)
        epoch_ce, epoch_opsdr, n_step = 0.0, 0.0, 0
        for sid in order:
            d = sft_by_id[sid]
            gt_letter = d["messages"][2]["content"].strip().upper()
            answer_token_id = get_answer_token_id(processor, gt_letter)
            inputs = build_message_inputs(processor, d, first_device)

            out = model(**inputs, output_hidden_states=use_opsdr, use_cache=False)
            logits_last = out.logits[0, -1, :].float()
            target = torch.tensor([answer_token_id], device=first_device)
            loss_ce = F.cross_entropy(logits_last.unsqueeze(0), target)

            if use_opsdr:
                h_gen = out.hidden_states[layer_idx][0, -1, :].float()
                h_gen_proj = P_theta(h_gen) if P_theta is not None else h_gen
                h_enc_target = h_enc_by_id[sid].to(first_device).detach()
                loss_opsdr = F.mse_loss(h_gen_proj, h_enc_target)
                loss = loss_ce + args.beta * loss_opsdr
                with torch.no_grad():
                    history["diag_curve"].append(float(torch.norm(h_gen_proj - h_enc_target).item()))
            else:
                loss_opsdr = torch.tensor(0.0)
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_ce += loss_ce.item()
            epoch_opsdr += float(loss_opsdr.item()) if use_opsdr else 0.0
            n_step += 1

        model.eval()
        acc_test = eval_accuracy(model, processor, first_device, sft_by_id, test_ids, args.eval_max_new_tokens)
        model.train()
        elapsed = time.time() - t0
        log(f"epoch={epoch} ce={epoch_ce/n_step:.4f} opsdr={epoch_opsdr/n_step:.4f} acc_test={acc_test:.2f}% ({elapsed:.1f}s)")
        history["epochs"].append({"epoch": epoch, "ce": epoch_ce / n_step, "opsdr": epoch_opsdr / n_step, "acc_test": acc_test})

        if acc_test > best_acc:
            best_acc, best_epoch = acc_test, epoch
            model.save_pretrained(os.path.join(run_dir, "best_adapter"))
            if P_theta is not None:
                torch.save(P_theta.state_dict(), os.path.join(run_dir, "best_p_theta.pt"))
        elif epoch > 0 and acc_test < history["epochs"][-2]["acc_test"]:
            log(f"acc_test下降(epoch{epoch}={acc_test:.2f}% < epoch{epoch-1}={history['epochs'][-2]['acc_test']:.2f}%),提前停止")
            break

    history["best_acc"] = best_acc
    history["best_epoch"] = best_epoch
    history["config"] = {
        "task": args.task, "beta": args.beta, "use_projection": use_projection,
        "layer_name": layer_name, "layer_idx": layer_idx, "lr": args.lr, "lora_r": args.lora_r,
        "n_train": len(train_ids), "n_test": len(test_ids),
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(history, f, indent=2)
    log(f"DONE. best_acc={best_acc:.2f}% at epoch {best_epoch} (pre-training={acc_pre:.2f}%). saved -> {run_dir}")


if __name__ == "__main__":
    main()
