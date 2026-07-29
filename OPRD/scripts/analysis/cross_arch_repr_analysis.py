#!/usr/bin/env python3
"""Pre-experiment 1 + 3: cross-architecture representation analysis (no training).

Computes per-layer Linear CKA, cross-layer CKA matrix, layer mapping phi,
and per-layer-pair linear probe SVD energy curves. Saves plots + JSON.

Token alignment: by default both student and teacher are forwarded with the
**student** tokenizer's input_ids (same as OPRD training). Diagnostics compare
what would happen if each model used its own tokenizer on the same text.

Example:
  PYTHONPATH=verl python scripts/analysis/cross_arch_repr_analysis.py \\
    --student-model-path /path/to/Qwen3-1.7B \\
    --teacher-model-path /path/to/Qwen3-4B \\
    --data-parquet ../datasets/dapo-math-17k.parquet \\
    --num-prompts 200 \\
    --output-dir outputs/cross_arch_preexp1 \\
    --generate-responses \\
    --max-new-tokens 4096
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LayerMapping:
    student_layers: list[int]
    teacher_layers: list[int]
    cka_scores: list[float]
    method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-architecture OPRD pre-experiment analysis")
    parser.add_argument("--student-model-path", type=str, required=True)
    parser.add_argument("--teacher-model-path", type=str, required=True)
    parser.add_argument("--data-parquet", type=str, required=True)
    parser.add_argument("--prompt-key", type=str, default="prompt")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modality", type=str, default="text", choices=["text", "video"],
        help="video: 16-frame MC-QA via a shared processor, short direct-generate, cross_arch_mm_utils forward path. "
        "Default text path is untouched.",
    )
    parser.add_argument("--stratify-key", type=str, default=None, help="Group-then-sample by this column (e.g. 'task')")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--responses-jsonl", type=str, default=None, help="Optional cached prompt/response pairs")
    parser.add_argument("--generate-responses", action="store_true", help="Generate one response per prompt with student")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--last-k", type=int, default=2000, help="Use last-k response tokens per sample")
    parser.add_argument("--max-tokens-per-layer", type=int, default=65536, help="Subsample rows for CKA/SVD")
    parser.add_argument("--ridge-lambda", type=float, default=1e-4, help="Ridge for linear probe W")
    parser.add_argument("--procrustes-rank", type=int, default=256, help="PCA rank for Procrustes alignment")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch-size", type=int, default=8, help="Max samples per forward batch")
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=131072,
        help="Forward batch token budget: batch_size * max_seq_len_in_batch <= this value",
    )
    parser.add_argument("--generate-batch-size", type=int, default=8, help="Batch size for transformers generation")
    parser.add_argument(
        "--video-student-max-batch-tokens",
        type=int,
        default=32768,
        help="Video-mode hidden-state extraction text-side token budget (B*max_len) for the student model; "
        "secondary constraint, see --video-student-max-vision-patches for the usually-binding one",
    )
    parser.add_argument(
        "--video-teacher-max-batch-tokens",
        type=int,
        default=12000,
        help="Video-mode hidden-state extraction text-side token budget (B*max_len) for the teacher model",
    )
    parser.add_argument(
        "--video-student-max-vision-patches",
        type=int,
        default=48000,
        help="Video-mode extraction: sum of raw vision-tower patches (video_grid_thw product, pre spatial-merge) "
        "allowed per batch for the student model. This is the constraint that actually protects against OOM "
        "(empirically ~3.5-3.9x the merged input_ids video-token count due to merge_size=2); calibrated from "
        "measured peak memory: 2x23808-patch samples -> 16.69GiB OK, 4x~18000-24000-patch samples -> OOM.",
    )
    parser.add_argument(
        "--video-teacher-max-vision-patches",
        type=int,
        default=19000,
        help="Same as --video-student-max-vision-patches but for the teacher model, which has far less headroom "
        "(weights alone ~17.5GiB/23.6GiB); calibrated from measured peak memory: 1x23808-patch sample -> "
        "22.21GiB OK (94% utilized), 4x4096-patch samples -> 21.00GiB OK, 6 short samples (27648 total) -> OOM.",
    )
    parser.add_argument(
        "--generate-backend",
        type=str,
        default="vllm",
        choices=["vllm", "transformers"],
        help="Student rollout backend. vllm is much faster and uses GPU better.",
    )
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=0,
        help="vLLM max_model_len; 0 = max_new_tokens + 4096 prompt budget",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--canonical-tokenizer",
        type=str,
        default="student",
        choices=["student", "teacher"],
        help=(
            "Tokenizer used to build input_ids for BOTH models. "
            "Default 'student' matches OPRD training (teacher sees student input_ids)."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def to_jsonable(value: Any) -> Any:
    """Convert parquet/numpy objects to JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def normalize_raw_prompt(prompt: Any) -> Any:
    return to_jsonable(prompt)


def _maybe_parse_json_messages(value: Any) -> Any:
    """Video-mode prompts are stored as a JSON string (see prepare_video_mcq_parquet.py)
    to avoid pyarrow unifying content-block schemas across system/user messages.
    Plain-text math prompts are ordinary (non-JSON) strings and fall through unchanged."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
        if isinstance(parsed, list):
            return parsed
    return value


def load_prompts(
    parquet_path: str, prompt_key: str, num_prompts: int, seed: int, stratify_key: str | None = None
) -> list[Any]:
    df = pd.read_parquet(parquet_path)
    if len(df) < num_prompts:
        raise ValueError(f"Dataset has {len(df)} rows, requested {num_prompts}")
    if stratify_key and stratify_key in df.columns:
        groups = list(df.groupby(stratify_key))
        per_group = max(1, num_prompts // len(groups))
        parts = [g.sample(n=min(len(g), per_group), random_state=seed) for _, g in groups]
        sampled = pd.concat(parts)
        if len(sampled) < num_prompts:
            remaining = df.drop(sampled.index)
            extra = remaining.sample(n=min(len(remaining), num_prompts - len(sampled)), random_state=seed)
            sampled = pd.concat([sampled, extra])
    else:
        sampled = df.sample(n=num_prompts, random_state=seed)
    return [normalize_raw_prompt(_maybe_parse_json_messages(p)) for p in sampled[prompt_key].tolist()]


def format_prompt(tokenizer, messages: Any, enable_thinking: bool) -> str:
    messages = normalize_raw_prompt(messages)
    if isinstance(messages, str):
        chat = [{"role": "user", "content": messages}]
    else:
        chat = messages
    kwargs = {"enable_thinking": enable_thinking} if enable_thinking else {"enable_thinking": False}
    try:
        return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)


def load_causal_lm(model_path: str, dtype: torch.dtype, device: torch.device):
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_cls = (
        AutoModelForImageTextToText
        if type(config) in AutoModelForImageTextToText._model_mapping.keys()
        else AutoModelForCausalLM
    )
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.output_hidden_states = True
    return model.to(device)


def load_processor(model_path: str):
    """Video mode only: load a processor (falls back to None if the model isn't
    multimodal, matching verl's hf_processor semantics — warns, doesn't raise)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "verl"))
    from verl.utils.tokenizer import hf_processor

    return hf_processor(model_path, trust_remote_code=True)


def tokenize_pair(
    pair: dict[str, Any],
    tokenizer,
    *,
    enable_thinking: bool,
) -> tuple[list[int], list[int]] | None:
    """Return (full_input_ids, response_mask) for prompt+response, or None if empty response."""
    prompt_text = format_prompt(tokenizer, pair["raw_prompt"], enable_thinking)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(pair["response"], add_special_tokens=False)["input_ids"]
    if not response_ids:
        return None
    full_ids = prompt_ids + response_ids
    response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return full_ids, response_mask


def estimate_pair_seq_len(pair: dict[str, Any], tokenizer, enable_thinking: bool) -> int:
    tokenized = tokenize_pair(pair, tokenizer, enable_thinking=enable_thinking)
    if tokenized is None:
        return 0
    return len(tokenized[0])


def max_token_id_in_pairs(
    pairs: list[dict[str, Any]],
    tokenizer,
    *,
    enable_thinking: bool,
) -> int:
    max_id = 0
    for pair in pairs:
        tokenized = tokenize_pair(pair, tokenizer, enable_thinking=enable_thinking)
        if tokenized is None:
            continue
        max_id = max(max_id, max(tokenized[0]))
    return max_id


def compute_token_alignment_diagnostics(
    pairs: list[dict[str, Any]],
    student_tokenizer,
    teacher_tokenizer,
    *,
    enable_thinking: bool,
) -> dict[str, Any]:
    """Compare per-sample tokenization when each model uses its own tokenizer."""
    length_diffs: list[int] = []
    response_len_diffs: list[int] = []
    exact_full_match = 0
    exact_response_match = 0
    compared = 0

    for pair in pairs:
        student_tok = tokenize_pair(pair, student_tokenizer, enable_thinking=enable_thinking)
        teacher_tok = tokenize_pair(pair, teacher_tokenizer, enable_thinking=enable_thinking)
        if student_tok is None or teacher_tok is None:
            continue
        student_ids, student_mask = student_tok
        teacher_ids, teacher_mask = teacher_tok
        compared += 1
        length_diffs.append(len(student_ids) - len(teacher_ids))
        student_resp_len = sum(student_mask)
        teacher_resp_len = sum(teacher_mask)
        response_len_diffs.append(student_resp_len - teacher_resp_len)
        if student_ids == teacher_ids:
            exact_full_match += 1
        student_resp_ids = [tid for tid, m in zip(student_ids, student_mask) if m]
        teacher_resp_ids = [tid for tid, m in zip(teacher_ids, teacher_mask) if m]
        if student_resp_ids == teacher_resp_ids:
            exact_response_match += 1

    if compared == 0:
        return {
            "num_pairs_compared": 0,
            "warning": "No valid pairs for tokenizer comparison",
        }

    abs_len_diff = np.abs(length_diffs)
    abs_resp_diff = np.abs(response_len_diffs)
    return {
        "num_pairs_compared": compared,
        "exact_full_sequence_match_rate": float(exact_full_match / compared),
        "exact_response_match_rate": float(exact_response_match / compared),
        "full_seq_len_diff_mean": float(np.mean(length_diffs)),
        "full_seq_len_diff_abs_mean": float(np.mean(abs_len_diff)),
        "full_seq_len_diff_abs_max": int(np.max(abs_len_diff)),
        "response_len_diff_mean": float(np.mean(response_len_diffs)),
        "response_len_diff_abs_mean": float(np.mean(abs_resp_diff)),
        "response_len_diff_abs_max": int(np.max(abs_resp_diff)),
        "fraction_pairs_with_len_mismatch": float(np.mean(abs_len_diff > 0)),
    }


def validate_canonical_ids_for_model(
    pairs: list[dict[str, Any]],
    canonical_tokenizer,
    model,
    *,
    enable_thinking: bool,
    role: str,
) -> dict[str, Any]:
    vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
    max_id = max_token_id_in_pairs(pairs, canonical_tokenizer, enable_thinking=enable_thinking)
    oob = vocab_size > 0 and max_id >= vocab_size
    info = {
        "role": role,
        "vocab_size": vocab_size,
        "max_token_id": int(max_id),
        "token_id_out_of_vocab": bool(oob),
    }
    if oob:
        raise ValueError(
            f"{role} vocab_size={vocab_size} but canonical tokenizer produced token id {max_id}. "
            "Cannot share student input_ids with teacher; check tokenizer compatibility."
        )
    return info


def make_dynamic_batches(
    pairs: list[dict[str, Any]],
    tokenizer,
    *,
    enable_thinking: bool,
    max_batch_size: int,
    max_batch_tokens: int,
) -> list[list[dict[str, Any]]]:
    """Pack samples into batches using length-aware token budget (B * max_seq_len)."""
    indexed_lens = []
    for idx, pair in enumerate(pairs):
        if "response" not in pair:
            indexed_lens.append((idx, 0))
            continue
        indexed_lens.append((idx, estimate_pair_seq_len(pair, tokenizer, enable_thinking)))

    indexed_lens = [(idx, seqlen) for idx, seqlen in indexed_lens if seqlen > 0]
    indexed_lens.sort(key=lambda x: x[1], reverse=True)

    batches: list[list[dict[str, Any]]] = []
    current_ids: list[int] = []
    current_max_len = 0

    for idx, seqlen in indexed_lens:
        candidate_len = max(current_max_len, seqlen)
        candidate_size = len(current_ids) + 1
        candidate_tokens = candidate_size * candidate_len
        if current_ids and (
            len(current_ids) >= max_batch_size or candidate_tokens > max_batch_tokens
        ):
            batches.append([pairs[i] for i in current_ids])
            current_ids = [idx]
            current_max_len = seqlen
        else:
            current_ids.append(idx)
            current_max_len = candidate_len

    if current_ids:
        batches.append([pairs[i] for i in current_ids])
    return batches


def load_pairs_from_jsonl(responses_jsonl: str, num_prompts: int) -> list[dict[str, Any]]:
    pairs = []
    with open(responses_jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            raw_prompt = row.get("raw_prompt", row.get("prompt"))
            pairs.append({"raw_prompt": raw_prompt, "response": row["response"]})
    if len(pairs) < num_prompts:
        raise ValueError(f"responses-jsonl has {len(pairs)} rows, need {num_prompts}")
    return pairs[:num_prompts]


def generate_pairs_with_transformers(
    prompts: list[Any],
    *,
    student_model,
    student_tokenizer,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
    generate_batch_size: int,
) -> list[dict[str, Any]]:
    pairs = []
    student_model.eval()
    student_tokenizer.padding_side = "left"
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

    prompt_chunks = [
        prompts[i : i + generate_batch_size] for i in range(0, len(prompts), generate_batch_size)
    ]
    for chunk in tqdm(prompt_chunks, desc="Generate (transformers)", unit="batch"):
        prompt_texts = [format_prompt(student_tokenizer, raw_prompt, enable_thinking) for raw_prompt in chunk]
        inputs = student_tokenizer(prompt_texts, return_tensors="pt", padding=True).to(device)
        input_width = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = student_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=student_tokenizer.pad_token_id,
            )
        for raw_prompt, row_ids in zip(chunk, output_ids):
            gen_ids = row_ids[input_width:]
            response = student_tokenizer.decode(gen_ids, skip_special_tokens=True)
            pairs.append({"raw_prompt": raw_prompt, "response": response})
    return pairs


def generate_pairs_with_vllm(
    prompts: list[Any],
    *,
    student_model_path: str,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
    vllm_tensor_parallel_size: int,
    vllm_gpu_memory_utilization: float,
    vllm_max_model_len: int,
) -> list[dict[str, Any]]:
    import gc

    from vllm import LLM, SamplingParams

    max_model_len = vllm_max_model_len if vllm_max_model_len > 0 else max_new_tokens + 4096
    print(
        f"Initializing vLLM for generation (tp={vllm_tensor_parallel_size}, "
        f"max_model_len={max_model_len}, gpu_mem={vllm_gpu_memory_utilization})"
    )
    llm = LLM(
        model=student_model_path,
        tensor_parallel_size=vllm_tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=max(temperature, 1e-5) if temperature > 0 else 0.0,
        max_tokens=max_new_tokens,
    )

    pairs: list[dict[str, Any]] = []
    chunk_size = 64
    prompt_chunks = [prompts[i : i + chunk_size] for i in range(0, len(prompts), chunk_size)]
    for chunk in tqdm(prompt_chunks, desc="Generate (vllm)", unit="batch"):
        prompt_texts = [format_prompt(tokenizer, raw_prompt, enable_thinking) for raw_prompt in chunk]
        outputs = llm.generate(prompt_texts, sampling_params)
        for raw_prompt, output in zip(chunk, outputs):
            response = output.outputs[0].text
            pairs.append({"raw_prompt": raw_prompt, "response": response})

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pairs


def generate_pairs_with_video_short(
    prompts: list[Any],
    *,
    student_model,
    processor,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> list[dict[str, Any]]:
    """Video mode generation: responses are expected to be ~1 token (bare option
    letter), so a short direct-generate loop (mirroring
    videoPerception/probe_data/direct_answer_baseline.py, the already-working
    template for this exact model+data combination) is used instead of the
    vLLM/long-CoT paths above, which are sized for multi-thousand-token math CoT."""
    import sys

    sys.path.insert(0, "/root/miniconda3/envs/lmmseval/lib/python3.12/site-packages")
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer
    from qwen_vl_utils import process_vision_info

    from cross_arch_mm_utils import extract_video_paths

    pairs: list[dict[str, Any]] = []
    for raw_prompt in tqdm(prompts, desc="Generate (video, short)", unit="sample"):
        messages = normalize_raw_prompt(raw_prompt)
        video_paths = extract_video_paths(messages)
        # options text lives in the last text block ("question\nA. ...\nB. ...\n...")
        question_block = next(b["text"] for m in messages if isinstance(m.get("content"), list)
                               for b in m["content"] if b.get("type") == "text")
        choices = [line[0] for line in question_block.splitlines() if len(line) > 1 and line[1] == "."]

        text = processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True,
        )
        video_metadata_list = None
        if video_inputs is not None:
            video_inputs, video_metadata_list = map(list, zip(*video_inputs))
        inputs = processor(
            text=text, images=image_inputs, videos=video_inputs, video_metadata=video_metadata_list,
            **video_kwargs, do_resize=False, return_tensors="pt",
        )
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        with torch.no_grad():
            gen_ids = student_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                temperature=max(temperature, 1e-5) if temperature > 0 else None,
            )
        new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
        raw_response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        letter = extract_mcq_answer(raw_response, choices=choices or None)
        pairs.append({"raw_prompt": raw_prompt, "response": letter or raw_response})
    return pairs


def load_or_build_pairs(
    prompts: list[Any],
    *,
    student_model_path: str,
    responses_jsonl: str | None,
    student_model,
    student_tokenizer,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
    generate_responses: bool,
    generate_backend: str,
    generate_batch_size: int,
    vllm_tensor_parallel_size: int,
    vllm_gpu_memory_utilization: float,
    vllm_max_model_len: int,
) -> list[dict[str, Any]]:
    if responses_jsonl:
        return load_pairs_from_jsonl(responses_jsonl, len(prompts))

    if not generate_responses:
        raise ValueError("Provide --responses-jsonl or pass --generate-responses")

    if generate_backend == "vllm":
        return generate_pairs_with_vllm(
            prompts,
            student_model_path=student_model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_max_model_len=vllm_max_model_len,
        )

    if student_model is None or student_tokenizer is None:
        raise ValueError("transformers generation requires student_model and student_tokenizer")
    return generate_pairs_with_transformers(
        prompts,
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        generate_batch_size=generate_batch_size,
    )


def save_pairs_jsonl(pairs: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in pairs:
            payload = {
                "raw_prompt": to_jsonable(row["raw_prompt"]),
                "response": str(row["response"]),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_batch_tensors(
    pairs: list[dict[str, Any]],
    tokenizer,
    *,
    last_k: int,
    enable_thinking: bool,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids_list = []
    response_mask_list = []
    attention_mask_list = []

    for pair in pairs:
        tokenized = tokenize_pair(pair, tokenizer, enable_thinking=enable_thinking)
        if tokenized is None:
            continue
        full_ids, response_mask = tokenized
        input_ids_list.append(full_ids)
        response_mask_list.append(response_mask)
        attention_mask_list.append([1] * len(full_ids))

    if not input_ids_list:
        raise ValueError("No valid prompt/response pairs after tokenization")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_len = max(len(x) for x in input_ids_list)
    batch_size = len(input_ids_list)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)

    # Left-pad batched causal LM inputs so valid tokens align on the right.
    for i, (ids, rmask, amask) in enumerate(zip(input_ids_list, response_mask_list, attention_mask_list)):
        offset = max_len - len(ids)
        input_ids[i, offset:] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, offset:] = 1
        response_mask[i, offset:] = torch.tensor(rmask, dtype=torch.float32)

    if device is not None:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        response_mask = response_mask.to(device, non_blocking=True)
    return input_ids, attention_mask, response_mask


def extract_response_hidden_rows(
    hidden_state: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    last_k: int,
) -> np.ndarray:
    """hidden_state: (B, T, D). Return stacked rows (N, D) using last-k valid response tokens."""
    rows = []
    for b in range(hidden_state.shape[0]):
        valid = response_mask[b].bool()
        if not valid.any():
            continue
        h = hidden_state[b, valid].detach().float()
        if last_k > 0 and h.shape[0] > last_k:
            h = h[-last_k:]
        rows.append(h.cpu().numpy())
    if not rows:
        return np.zeros((0, hidden_state.shape[-1]), dtype=np.float32)
    return np.concatenate(rows, axis=0)


def forward_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[list[torch.Tensor], int]:
    model.eval()
    with torch.no_grad():
        # logits_to_keep=1: same fix as forward_multimodal_hidden_states in cross_arch_mm_utils.py
        # -- default logits_to_keep=0 computes logits for the full sequence (unused here, only
        # .hidden_states is read), which was the actual cause of a Stage-1 OOM this session.
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1
        )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states; set config.output_hidden_states=True")
    # Keep layer hidden on GPU; only response slices are copied to CPU for accumulation.
    per_layer = [hs.detach().float() for hs in hidden_states[1:]]
    return per_layer, len(per_layer)


def accumulate_layer_matrices(
    model,
    tokenizer,
    pairs: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    max_batch_tokens: int,
    last_k: int,
    enable_thinking: bool,
    desc: str = "Forward hidden states",
) -> list[np.ndarray]:
    layer_rows: list[list[np.ndarray]] | None = None
    num_layers = None
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    batches = make_dynamic_batches(
        pairs,
        tokenizer,
        enable_thinking=enable_thinking,
        max_batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
    )
    batch_sizes = [len(batch) for batch in batches]
    if batch_sizes:
        model_device = next(model.parameters()).device
        tqdm.write(
            f"{desc}: model on {model_device}, {len(batches)} batches, "
            f"avg_size={np.mean(batch_sizes):.1f}, max_size={max(batch_sizes)}"
        )

    for batch_pairs in tqdm(batches, desc=desc, unit="batch"):
        input_ids, attention_mask, response_mask = build_batch_tensors(
            batch_pairs,
            tokenizer,
            last_k=last_k,
            enable_thinking=enable_thinking,
            device=device,
        )
        per_layer_hidden, n_layers = forward_hidden_states(model, input_ids, attention_mask)
        if layer_rows is None:
            num_layers = n_layers
            layer_rows = [[] for _ in range(num_layers)]

        for layer_idx, layer_hidden in enumerate(per_layer_hidden):
            rows = extract_response_hidden_rows(layer_hidden, response_mask, last_k=last_k)
            if rows.shape[0] > 0:
                layer_rows[layer_idx].append(rows)

    assert layer_rows is not None and num_layers is not None
    return [np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1)) for chunks in layer_rows]


def make_dynamic_batches_video(
    samples: list[dict[str, Any]],
    *,
    max_batch_size: int,
    max_batch_tokens: int,
    max_vision_patches: int,
) -> list[list[dict[str, Any]]]:
    """Pack already-tokenized multimodal samples into batches under TWO independent
    memory constraints, mirroring make_dynamic_batches for text mode but extended for
    video: (1) B * max_seq_len_in_batch <= max_batch_tokens, same product-based budget
    as text mode, protecting the language model's retained per-layer hidden_states;
    (2) sum of raw vision-patch counts (video_grid_thw.prod(-1).sum() per sample,
    BEFORE the merge_size**2 reduction into input_ids video-token count) across the
    batch <= max_vision_patches, protecting the vision tower's own activations.

    The vision constraint is not optional / not a nice-to-have: empirically (see
    partitioned-wandering-deer.md, 2026-07-26 batching incident), the vision tower's
    raw patch count is ~3.5-3.9x the merged input_ids video-token count (merge_size=2
    means each merged token maps to 4 raw patches spatially), so a budget based on
    input_ids length alone silently underestimates the dominant OOM driver, which
    surfaces inside the vision tower's own attention (not the language model)."""
    indexed = [
        (i, s["input_ids"].shape[0], int(s["video_grid_thw"].prod(dim=-1).sum().item()))
        for i, s in enumerate(samples)
    ]
    indexed.sort(key=lambda x: x[1], reverse=True)

    batches: list[list[int]] = []
    current_ids: list[int] = []
    current_max_len = 0
    current_patches = 0

    for idx, seqlen, patches in indexed:
        candidate_len = max(current_max_len, seqlen)
        candidate_size = len(current_ids) + 1
        candidate_tokens = candidate_size * candidate_len
        candidate_patches = current_patches + patches
        if current_ids and (
            len(current_ids) >= max_batch_size
            or candidate_tokens > max_batch_tokens
            or candidate_patches > max_vision_patches
        ):
            batches.append(current_ids)
            current_ids = [idx]
            current_max_len = seqlen
            current_patches = patches
        else:
            current_ids.append(idx)
            current_max_len = candidate_len
            current_patches = candidate_patches

    if current_ids:
        batches.append(current_ids)
    return [[samples[i] for i in ids] for ids in batches]


def accumulate_layer_matrices_video(
    model,
    processor,
    pairs: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    max_batch_tokens: int,
    max_vision_patches: int,
    last_k: int,
    enable_thinking: bool,
    tokenize_chunk_size: int = 100,
    desc: str = "Forward hidden states (video)",
) -> list[np.ndarray]:
    """Host-RAM note (2026-07-26 incident, see partitioned-wandering-deer.md): each
    tokenized sample's pixel_values_videos tensor is large (~50-150MB depending on
    video length), so materializing all ~1000 samples of a shard at once before
    batching (the original design, needed to know lengths for sorting) drove host RSS
    to ~110GB per process -- with 4 shards in parallel this exhausted host RAM/swap
    and got a process OS-OOM-killed with no Python traceback (a different failure
    mode than the CUDA OOM this function's vision-patch budget already guards
    against). Fix: tokenize + batch + forward + discard in small chunks of raw pairs,
    so at most tokenize_chunk_size samples' heavy tensors are resident at once. This
    sacrifices global length-sorting across the whole shard for local sorting within
    each chunk (make_dynamic_batches_video still runs per chunk) -- a minor batching
    efficiency loss, not a correctness issue."""
    import gc

    from cross_arch_mm_utils import build_multimodal_batch, build_multimodal_sample, forward_multimodal_hidden_states

    layer_rows: list[list[np.ndarray]] | None = None
    num_layers = None
    pad_token_id = processor.tokenizer.pad_token_id
    model_device = next(model.parameters()).device

    num_chunks = (len(pairs) + tokenize_chunk_size - 1) // tokenize_chunk_size
    for chunk_idx in range(num_chunks):
        chunk_pairs = pairs[chunk_idx * tokenize_chunk_size : (chunk_idx + 1) * tokenize_chunk_size]
        samples = []
        for pair in tqdm(
            chunk_pairs, desc=f"{desc}: tokenize chunk {chunk_idx + 1}/{num_chunks}", unit="sample", leave=False
        ):
            messages = normalize_raw_prompt(pair["raw_prompt"])
            s = build_multimodal_sample(messages, str(pair["response"]), processor, enable_thinking=enable_thinking)
            if s is not None:
                samples.append(s)

        batches = make_dynamic_batches_video(
            samples, max_batch_size=batch_size, max_batch_tokens=max_batch_tokens, max_vision_patches=max_vision_patches
        )
        batch_sizes = [len(b) for b in batches]
        if batch_sizes:
            tqdm.write(
                f"{desc}: chunk {chunk_idx + 1}/{num_chunks} on {model_device}, {len(batches)} batches, "
                f"avg_size={np.mean(batch_sizes):.1f}, max_size={max(batch_sizes)}"
            )

        for batch_samples in tqdm(
            batches, desc=f"{desc} chunk {chunk_idx + 1}/{num_chunks}", unit="batch", leave=False
        ):
            batch = build_multimodal_batch(batch_samples, pad_token_id)
            hidden_states = forward_multimodal_hidden_states(model, batch, device)
            per_layer = hidden_states[1:]  # drop embedding layer, matching accumulate_layer_matrices' convention
            if layer_rows is None:
                num_layers = len(per_layer)
                layer_rows = [[] for _ in range(num_layers)]
            response_mask = batch["response_mask"]
            for layer_idx, layer_hidden in enumerate(per_layer):
                rows = extract_response_hidden_rows(layer_hidden, response_mask, last_k=last_k)
                if rows.shape[0] > 0:
                    layer_rows[layer_idx].append(rows)
            del hidden_states, per_layer, batch
            torch.cuda.empty_cache()

        del samples, batches
        gc.collect()

    assert layer_rows is not None and num_layers is not None
    return [np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1)) for chunks in layer_rows]


def subsample_rows(matrix: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if matrix.shape[0] <= max_rows:
        return matrix
    rng = np.random.default_rng(seed)
    idx = rng.choice(matrix.shape[0], size=max_rows, replace=False)
    return matrix[idx]


def center_matrix(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def _analysis_device(device: torch.device | str | None = None) -> torch.device | None:
    """Resolve the device the CPU-bound CKA/probe/SVD math should run on. Returns
    None (meaning "stay in numpy") only if CUDA truly isn't available anywhere --
    otherwise defaults to cuda:0. This math is O(n*d^2)/O(d^3) per layer pair and
    was measured to run ~10x slower than usual under host-CPU contention on a shared
    server (2026-07-27 incident); GPUs are not subject to that same contention and
    the matrices involved are tiny relative to model-weight memory, so preferring
    GPU here is a clear, low-risk win."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return None


def _to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=torch.float32)


def linear_cka(x: np.ndarray, y: np.ndarray, *, device: torch.device | str | None = None) -> float:
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    dev = _analysis_device(device)
    if dev is None:
        x = center_matrix(x)
        y = center_matrix(y)
        xtx = x.T @ x
        yty = y.T @ y
        xty = x.T @ y
        hsic = np.linalg.norm(xty, ord="fro") ** 2
        denom = np.linalg.norm(xtx, ord="fro") * np.linalg.norm(yty, ord="fro")
        return float(hsic / denom) if denom > 0 else float("nan")
    xt = _to_torch(x, dev)
    yt = _to_torch(y, dev)
    xt = xt - xt.mean(dim=0, keepdim=True)
    yt = yt - yt.mean(dim=0, keepdim=True)
    xtx = xt.T @ xt
    yty = yt.T @ yt
    xty = xt.T @ yt
    hsic = torch.linalg.norm(xty, ord="fro") ** 2
    denom = torch.linalg.norm(xtx, ord="fro") * torch.linalg.norm(yty, ord="fro")
    if float(denom) <= 0:
        return float("nan")
    return float((hsic / denom).item())


def fit_linear_probe_w(
    h_student: np.ndarray,
    h_teacher: np.ndarray,
    *,
    ridge_lambda: float,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Return W with shape (d_teacher, d_student) so H_T ~ H_S @ W.T."""
    dev = _analysis_device(device)
    if dev is None:
        x = h_student
        y = h_teacher
        xtx = x.T @ x
        reg = ridge_lambda * np.eye(xtx.shape[0], dtype=xtx.dtype)
        w_t = np.linalg.solve(xtx + reg, x.T @ y)
        return w_t.T
    x = _to_torch(h_student, dev)
    y = _to_torch(h_teacher, dev)
    xtx = x.T @ x
    reg = ridge_lambda * torch.eye(xtx.shape[0], dtype=xtx.dtype, device=dev)
    w_t = torch.linalg.solve(xtx + reg, x.T @ y)
    return w_t.T.cpu().numpy()


def singular_value_energy_curve(
    w: np.ndarray, *, device: torch.device | str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    dev = _analysis_device(device)
    if dev is None:
        singular_values = np.linalg.svd(w, compute_uv=False)
    else:
        singular_values = torch.linalg.svdvals(_to_torch(w, dev)).cpu().numpy()
    energy = singular_values**2
    total = energy.sum()
    if total <= 0:
        return singular_values, np.zeros_like(energy)
    cumulative = np.cumsum(energy) / total
    return singular_values, cumulative


def low_rank_reconstruction(
    w: np.ndarray, ranks: list[int], *, device: torch.device | str | None = None
) -> dict[int, np.ndarray]:
    """SVD w ONCE and slice it for every requested rank, instead of re-running a full
    SVD per rank (the original code called np.linalg.svd(w) once per rank in a loop
    over the same w -- same decomposition, wastefully recomputed 4x)."""
    dev = _analysis_device(device)
    if dev is None:
        u, svals, vt = np.linalg.svd(w, full_matrices=False)
    else:
        wt = _to_torch(w, dev)
        u_t, svals_t, vt_t = torch.linalg.svd(wt, full_matrices=False)
        u, svals, vt = u_t.cpu().numpy(), svals_t.cpu().numpy(), vt_t.cpu().numpy()
    out = {}
    for r in ranks:
        r_eff = min(r, w.shape[0], w.shape[1])
        if r_eff < 1:
            continue
        out[r] = (u[:, :r_eff] * svals[:r_eff]) @ vt[:r_eff]
    return out


def proportional_layer_map(num_student_layers: int, num_teacher_layers: int) -> list[int]:
    if num_student_layers == 1:
        return [num_teacher_layers - 1]
    return [min(round(i * (num_teacher_layers - 1) / (num_student_layers - 1)), num_teacher_layers - 1) for i in range(num_student_layers)]


def cka_layer_map(cka_matrix: np.ndarray) -> LayerMapping:
    student_layers = []
    teacher_layers = []
    scores = []
    for s_idx in range(cka_matrix.shape[0]):
        t_idx = int(np.nanargmax(cka_matrix[s_idx]))
        student_layers.append(s_idx)
        teacher_layers.append(t_idx)
        scores.append(float(cka_matrix[s_idx, t_idx]))
    return LayerMapping(student_layers, teacher_layers, scores, method="cka_argmax")


def pca_basis(
    matrix: np.ndarray, rank: int, *, device: torch.device | str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    centered = center_matrix(matrix)
    if centered.shape[0] < 2:
        raise ValueError("Need at least 2 rows for PCA")
    dev = _analysis_device(device)
    if dev is None:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    else:
        _, _, vt_t = torch.linalg.svd(_to_torch(centered, dev), full_matrices=False)
        vt = vt_t.cpu().numpy()
    rank = min(rank, vt.shape[0])
    basis = vt[:rank]
    projected = centered @ basis.T
    return basis, projected


def procrustes_aligned_cosine(
    h_student: np.ndarray, h_teacher: np.ndarray, rank: int, *, device: torch.device | str | None = None
) -> float:
    rank = min(rank, h_student.shape[1], h_teacher.shape[1], h_student.shape[0] - 1, h_teacher.shape[0] - 1)
    if rank < 1:
        return float("nan")
    _, z_s = pca_basis(h_student, rank, device=device)
    _, z_t = pca_basis(h_teacher, rank, device=device)
    m = z_s.T @ z_t
    dev = _analysis_device(device)
    if dev is None:
        u, _, vt = np.linalg.svd(m, full_matrices=False)
    else:
        u_t, _, vt_t = torch.linalg.svd(_to_torch(m, dev), full_matrices=False)
        u, vt = u_t.cpu().numpy(), vt_t.cpu().numpy()
    r = u @ vt
    aligned_s = z_s @ r
    num = np.sum(aligned_s * z_t)
    den = np.linalg.norm(aligned_s) * np.linalg.norm(z_t)
    return float(num / den) if den > 0 else float("nan")


def mean_cosine_same_dim(h_student: np.ndarray, h_teacher: np.ndarray) -> float:
    if h_student.shape[1] != h_teacher.shape[1]:
        return float("nan")
    a = h_student / (np.linalg.norm(h_student, axis=1, keepdims=True) + 1e-8)
    b = h_teacher / (np.linalg.norm(h_teacher, axis=1, keepdims=True) + 1e-8)
    return float((a * b).sum(axis=1).mean())


def plot_cka_heatmap(cka_matrix: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cka_matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Teacher layer")
    ax.set_ylabel("Student layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_energy_curve(ranks: np.ndarray, curves: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, curve in curves.items():
        ax.plot(ranks, curve, label=label, linewidth=2)
    ax.axhline(0.85, color="gray", linestyle="--", linewidth=1, label="85% energy")
    ax.axhline(0.90, color="gray", linestyle=":", linewidth=1, label="90% energy")
    ax.set_xlabel("Rank r")
    ax.set_ylabel("Cumulative singular-value energy")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.01)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def elbow_rank(cumulative_energy: np.ndarray) -> int:
    for threshold in (0.85, 0.90, 0.95):
        idx = np.searchsorted(cumulative_energy, threshold)
        if idx < len(cumulative_energy):
            return int(idx + 1)
    return int(len(cumulative_energy))


def run_cka_analysis(
    student_layers: list[np.ndarray],
    teacher_layers: list[np.ndarray],
    output_dir: Path,
    args: argparse.Namespace,
    *,
    token_alignment: dict[str, Any],
    teacher_vocab_info: dict[str, Any],
    canonical_tokenizer_name: str,
) -> None:
    """Modality-agnostic: everything from here on only looks at the per-layer
    (num_rows, hidden_dim) matrices, regardless of how they were produced."""
    analysis_device = _analysis_device(None)
    if analysis_device is not None:
        print(f"Running CKA/probe/SVD math on {analysis_device} (not host CPU)")
    num_student_layers = len(student_layers)
    num_teacher_layers = len(teacher_layers)
    print(f"Student layers: {num_student_layers}, Teacher layers: {num_teacher_layers}")

    row_count_mismatch = []
    for layer_idx in range(min(num_student_layers, num_teacher_layers)):
        s_rows = student_layers[layer_idx].shape[0]
        t_rows = teacher_layers[layer_idx].shape[0]
        if s_rows != t_rows:
            row_count_mismatch.append({"layer": layer_idx, "student_rows": s_rows, "teacher_rows": t_rows})
    hidden_row_alignment = {
        "all_layers_same_row_count": len(row_count_mismatch) == 0,
        "student_rows_per_layer": [int(h.shape[0]) for h in student_layers],
        "teacher_rows_per_layer": [int(h.shape[0]) for h in teacher_layers],
        "mismatched_layers": row_count_mismatch,
    }
    if row_count_mismatch:
        print("WARNING: student/teacher hidden row counts differ (token alignment may still be wrong):")
        print(json.dumps(row_count_mismatch[:5], indent=2))
    else:
        print(f"Hidden row counts aligned across all layers ({student_layers[0].shape[0]} rows/layer).")

    max_rows = args.max_tokens_per_layer
    student_layers = [subsample_rows(h, max_rows, args.seed + i) for i, h in enumerate(student_layers)]
    teacher_layers = [subsample_rows(h, max_rows, args.seed + 1000 + i) for i, h in enumerate(teacher_layers)]

    cka_matrix = np.zeros((num_student_layers, num_teacher_layers), dtype=np.float64)
    cka_pairs = [
        (s_idx, t_idx)
        for s_idx in range(num_student_layers)
        for t_idx in range(num_teacher_layers)
    ]
    for s_idx, t_idx in tqdm(cka_pairs, desc="Compute CKA matrix", unit="pair"):
        n = min(student_layers[s_idx].shape[0], teacher_layers[t_idx].shape[0])
        if n < 2:
            cka_matrix[s_idx, t_idx] = np.nan
            continue
        cka_matrix[s_idx, t_idx] = linear_cka(
            student_layers[s_idx][:n], teacher_layers[t_idx][:n], device=analysis_device
        )

    plot_cka_heatmap(cka_matrix, output_dir / "cka_heatmap.png", "Linear CKA (student x teacher layers)")

    cka_mapping = cka_layer_map(cka_matrix)
    prop_teacher_layers = proportional_layer_map(num_student_layers, num_teacher_layers)
    prop_mapping = LayerMapping(
        list(range(num_student_layers)),
        prop_teacher_layers,
        [float(cka_matrix[i, j]) for i, j in zip(range(num_student_layers), prop_teacher_layers)],
        method="proportional",
    )

    per_pair_energy: dict[str, list[float]] = {}
    per_pair_elbow: dict[str, int] = {}
    matched_cosine_raw = []
    matched_cosine_procrustes = []
    low_rank_probe_cosine: dict[int, list[float]] = {64: [], 128: [], 256: [], 512: []}

    layer_pairs = list(zip(cka_mapping.student_layers, cka_mapping.teacher_layers))
    for s_idx, t_idx in tqdm(layer_pairs, desc="Linear probe + SVD", unit="layer"):
        h_s = student_layers[s_idx]
        h_t = teacher_layers[t_idx]
        n = min(h_s.shape[0], h_t.shape[0])
        h_s = h_s[:n]
        h_t = h_t[:n]
        pair_name = f"s{s_idx:02d}_t{t_idx:02d}"

        matched_cosine_raw.append(mean_cosine_same_dim(h_s, h_t) if h_s.shape[1] == h_t.shape[1] else float("nan"))
        matched_cosine_procrustes.append(
            procrustes_aligned_cosine(h_s, h_t, args.procrustes_rank, device=analysis_device)
        )

        w = fit_linear_probe_w(h_s, h_t, ridge_lambda=args.ridge_lambda, device=analysis_device)
        _, cumulative = singular_value_energy_curve(w, device=analysis_device)
        per_pair_energy[pair_name] = cumulative.tolist()
        per_pair_elbow[pair_name] = elbow_rank(cumulative)

        # SVD w once and reuse for every rank, instead of the original's 4 redundant
        # full SVDs of the same w (one per r in low_rank_probe_cosine).
        w_r_by_rank = low_rank_reconstruction(w, list(low_rank_probe_cosine.keys()), device=analysis_device)
        b = h_t / (np.linalg.norm(h_t, axis=1, keepdims=True) + 1e-8)
        for r, w_r in w_r_by_rank.items():
            pred = h_s @ w_r.T
            a = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
            low_rank_probe_cosine[r].append(float((a * b).sum(axis=1).mean()))

    max_rank = max(len(curve) for curve in per_pair_energy.values())
    ranks = np.arange(1, max_rank + 1)
    mean_energy = np.zeros(max_rank)
    for curve in per_pair_energy.values():
        arr = np.array(curve)
        mean_energy[: len(arr)] += arr
    mean_energy /= max(len(per_pair_energy), 1)

    plot_energy_curve(
        ranks,
        {"mean_probe_energy": mean_energy},
        output_dir / "svd_energy_mean.png",
        "Mean singular-value energy (CKA-matched layer pairs)",
    )

    # Null baseline: shuffle teacher layer assignment
    rng = np.random.default_rng(args.seed)
    shuffled_teacher = rng.permutation(num_teacher_layers)
    null_cka = []
    for s_idx in range(num_student_layers):
        t_idx = int(shuffled_teacher[s_idx])
        n = min(student_layers[s_idx].shape[0], teacher_layers[t_idx].shape[0])
        if n < 2:
            continue
        null_cka.append(linear_cka(student_layers[s_idx][:n], teacher_layers[t_idx][:n], device=analysis_device))

    summary = {
        "student_model": args.student_model_path,
        "teacher_model": args.teacher_model_path,
        "num_prompts": args.num_prompts,
        "last_k": args.last_k,
        "num_student_layers": num_student_layers,
        "num_teacher_layers": num_teacher_layers,
        "student_hidden_dim": int(student_layers[0].shape[1]),
        "teacher_hidden_dim": int(teacher_layers[0].shape[1]),
        "canonical_tokenizer": canonical_tokenizer_name,
        "token_alignment": token_alignment,
        "hidden_row_alignment": hidden_row_alignment,
        "teacher_vocab_check": teacher_vocab_info,
        "cka_matrix": cka_matrix.tolist(),
        "cka_mapping": asdict(cka_mapping),
        "proportional_mapping": asdict(prop_mapping),
        "cka_mean_matched": float(np.nanmean(cka_mapping.cka_scores)),
        "cka_mean_null_shuffled": float(np.mean(null_cka)) if null_cka else float("nan"),
        "matched_cosine_raw_mean": float(np.nanmean(matched_cosine_raw)),
        "matched_cosine_procrustes_mean": float(np.nanmean(matched_cosine_procrustes)),
        "mean_probe_energy_curve": mean_energy.tolist(),
        "mean_probe_elbow_rank_85": elbow_rank(mean_energy),
        "per_pair_probe_elbow_rank_85": per_pair_elbow,
        "low_rank_probe_cosine_mean": {str(k): float(np.mean(v)) if v else float("nan") for k, v in low_rank_probe_cosine.items()},
        "gates": {
            "low_rank_feasible_85_at_512": bool(mean_energy[min(511, len(mean_energy) - 1)] >= 0.85) if len(mean_energy) >= 1 else False,
            "cka_above_null": bool(np.nanmean(cka_mapping.cka_scores) > (np.mean(null_cka) if null_cka else 0.0) + 0.05),
            "procrustes_cosine_above_0.3": bool(np.nanmean(matched_cosine_procrustes) > 0.3),
        },
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(output_dir / "layer_mapping_cka.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cka_mapping), f, indent=2)

    with open(output_dir / "layer_mapping_proportional.json", "w", encoding="utf-8") as f:
        json.dump(asdict(prop_mapping), f, indent=2)

    print(json.dumps(summary["gates"], indent=2))
    print(f"Saved analysis to {output_dir}")


def save_layer_matrices(layers: list[np.ndarray], path: Path) -> None:
    np.savez(path, *layers)


def load_layer_matrices(path: Path) -> list[np.ndarray]:
    with np.load(path) as data:
        return [data[f"arr_{i}"] for i in range(len(data.files))]


def run_video_mode(args: argparse.Namespace) -> None:
    """Checkpointing note (2026-07-26/27 incidents): student/teacher hidden-state
    extraction is the expensive, GPU-bound part of this pipeline (~15-25 min each)
    and used to only exist in memory until run_cka_analysis finished at the very
    end -- any crash or slow-down anywhere after extraction (a later OOM, a
    contended/slow CKA analysis step, killing the wrong process) threw away all of
    it. student_layers.npz/teacher_layers.npz are now written to disk immediately
    after each extraction completes, and re-loaded instead of recomputed if already
    present, so a restart only redoes whatever step actually failed."""
    output_dir = Path(args.output_dir)
    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    student_layers_path = output_dir / "student_layers.npz"
    teacher_layers_path = output_dir / "teacher_layers.npz"

    prompts = load_prompts(
        args.data_parquet, args.prompt_key, args.num_prompts, args.seed, stratify_key=args.stratify_key
    )

    processor = load_processor(args.student_model_path)
    teacher_processor = load_processor(args.teacher_model_path)
    # Required, not just convenient (see plan): teacher/student must see identical
    # tokenization for OPRD-Bridge's premise to hold. Fail loudly, not silently, if a
    # future teacher/student swap breaks this.
    for key in ("video_token_id", "image_token_id"):
        assert getattr(processor, key, None) == getattr(teacher_processor, key, None), (
            f"student/teacher processor mismatch on {key}: "
            f"{getattr(processor, key, None)} != {getattr(teacher_processor, key, None)}"
        )
    student_vocab_size = None  # filled in after student model loads, checked against teacher below

    print(f"Loading student model: {args.student_model_path}")
    student_model = load_causal_lm(args.student_model_path, dtype, device)

    if args.responses_jsonl:
        print(f"Loading cached pairs from {args.responses_jsonl} (skipping generation)")
        pairs = load_pairs_from_jsonl(args.responses_jsonl, len(prompts))
    else:
        pairs = generate_pairs_with_video_short(
            prompts, student_model=student_model, processor=processor, device=device,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, enable_thinking=args.enable_thinking,
        )
        save_pairs_jsonl(pairs, output_dir / "on_policy_pairs.jsonl")

    if student_layers_path.exists():
        print(f"Found cached {student_layers_path}, skipping student extraction")
        student_layers = load_layer_matrices(student_layers_path)
    else:
        print("Extracting student hidden states (video)...")
        student_layers = accumulate_layer_matrices_video(
            student_model, processor, pairs, device=device, batch_size=args.batch_size,
            max_batch_tokens=args.video_student_max_batch_tokens,
            max_vision_patches=args.video_student_max_vision_patches,
            last_k=args.last_k, enable_thinking=args.enable_thinking, desc="Student forward",
        )
        save_layer_matrices(student_layers, student_layers_path)
        print(f"Checkpointed student layers to {student_layers_path}")
    student_vocab_size = int(getattr(student_model.config, "vocab_size", 0) or 0)
    del student_model
    torch.cuda.empty_cache()

    print(f"Loading teacher model: {args.teacher_model_path}")
    teacher_model = load_causal_lm(args.teacher_model_path, dtype, device)
    teacher_vocab_size = int(getattr(teacher_model.config, "vocab_size", 0) or 0)
    teacher_vocab_info = {
        "role": "teacher", "vocab_size": teacher_vocab_size, "student_vocab_size": student_vocab_size,
        "shared_processor": True,
    }

    if teacher_layers_path.exists():
        print(f"Found cached {teacher_layers_path}, skipping teacher extraction")
        teacher_layers = load_layer_matrices(teacher_layers_path)
    else:
        print("Extracting teacher hidden states (same processor/input_ids construction as student)...")
        teacher_layers = accumulate_layer_matrices_video(
            teacher_model, processor, pairs, device=device, batch_size=args.batch_size,
            max_batch_tokens=args.video_teacher_max_batch_tokens,
            max_vision_patches=args.video_teacher_max_vision_patches,
            last_k=args.last_k, enable_thinking=args.enable_thinking, desc="Teacher forward",
        )
        save_layer_matrices(teacher_layers, teacher_layers_path)
        print(f"Checkpointed teacher layers to {teacher_layers_path}")
    del teacher_model
    torch.cuda.empty_cache()

    token_alignment = {
        "note": "not applicable in video mode: a single shared processor builds both models' "
        "input_ids from the same messages, so they're identical by construction, not by comparison.",
    }
    run_cka_analysis(
        student_layers, teacher_layers, output_dir, args,
        token_alignment=token_alignment, teacher_vocab_info=teacher_vocab_info,
        canonical_tokenizer_name="shared_processor",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.modality == "video":
        run_video_mode(args)
        return

    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    prompts = load_prompts(args.data_parquet, args.prompt_key, args.num_prompts, args.seed)

    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_path, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path, trust_remote_code=True)
    student_model = None
    use_vllm_generate = (
        args.generate_responses
        and not args.responses_jsonl
        and args.generate_backend == "vllm"
    )
    if not use_vllm_generate:
        print(f"Loading student model: {args.student_model_path}")
        student_model = load_causal_lm(args.student_model_path, dtype, device)
    else:
        print(f"Student rollout via vLLM: {args.student_model_path}")

    pairs = load_or_build_pairs(
        prompts,
        student_model_path=args.student_model_path,
        responses_jsonl=args.responses_jsonl,
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        generate_responses=args.generate_responses,
        generate_backend=args.generate_backend,
        generate_batch_size=args.generate_batch_size,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
    )
    save_pairs_jsonl(pairs, output_dir / "on_policy_pairs.jsonl")

    token_alignment = compute_token_alignment_diagnostics(
        pairs,
        student_tokenizer,
        teacher_tokenizer,
        enable_thinking=args.enable_thinking,
    )
    print("Tokenizer alignment (student vs teacher, same text):")
    print(json.dumps(token_alignment, indent=2))

    canonical_tokenizer = student_tokenizer if args.canonical_tokenizer == "student" else teacher_tokenizer
    print(f"Canonical tokenizer for both forwards: {args.canonical_tokenizer}")

    if student_model is None:
        print(f"Loading student model for hidden extraction: {args.student_model_path}")
        student_model = load_causal_lm(args.student_model_path, dtype, device)

    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        print("WARNING: CUDA unavailable; student forward will run on CPU.")
    else:
        print(f"Student forward device: {next(student_model.parameters()).device}")

    validate_canonical_ids_for_model(
        pairs,
        canonical_tokenizer,
        student_model,
        enable_thinking=args.enable_thinking,
        role="student",
    )

    print("Extracting student hidden states...")
    student_layers = accumulate_layer_matrices(
        student_model,
        canonical_tokenizer,
        pairs,
        device=device,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        last_k=args.last_k,
        enable_thinking=args.enable_thinking,
        desc="Student forward",
    )
    del student_model
    torch.cuda.empty_cache()

    print(f"Loading teacher model: {args.teacher_model_path}")
    teacher_model = load_causal_lm(args.teacher_model_path, dtype, device)
    teacher_vocab_info = validate_canonical_ids_for_model(
        pairs,
        canonical_tokenizer,
        teacher_model,
        enable_thinking=args.enable_thinking,
        role="teacher",
    )
    print(f"Teacher accepts canonical token ids: {json.dumps(teacher_vocab_info)}")

    print("Extracting teacher hidden states (same input_ids as student)...")
    teacher_layers = accumulate_layer_matrices(
        teacher_model,
        canonical_tokenizer,
        pairs,
        device=device,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        last_k=args.last_k,
        enable_thinking=args.enable_thinking,
        desc="Teacher forward",
    )
    del teacher_model
    torch.cuda.empty_cache()

    run_cka_analysis(
        student_layers, teacher_layers, output_dir, args,
        token_alignment=token_alignment, teacher_vocab_info=teacher_vocab_info,
        canonical_tokenizer_name=args.canonical_tokenizer,
    )


if __name__ == "__main__":
    main()
