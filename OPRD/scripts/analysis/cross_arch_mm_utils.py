"""Shared multimodal batch-building + forward logic for Stage 0
(cross_arch_repr_analysis.py) and Stage 1 (cross_arch_preexp2_train_ps.py)
video mode. Kept as a separate module rather than imported cross-stage or
duplicated inline: the multimodal path (processor calls, mm_token_type_ids
padding, position_ids via M-RoPE) is complex enough that fixing it once in
one place matters, while Stage 0/1's existing text-only functions stay
independent by design (confirmed intentional, not an oversight).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

_VERL_ROOT = str(Path(__file__).resolve().parents[2] / "verl")
if _VERL_ROOT not in sys.path:
    sys.path.insert(0, _VERL_ROOT)

from verl.models.transformers.qwen3_vl import get_rope_index  # noqa: E402
from verl.utils.dataset.vision_utils import process_video  # noqa: E402
from verl.utils.model import extract_multi_modal_inputs  # noqa: E402


def extract_video_paths(messages: list[dict]) -> list[str]:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "video":
                return block["video"]
    raise ValueError("no video block found in messages")


def build_multimodal_sample(
    messages: list[dict],
    response_text: str,
    processor,
    *,
    enable_thinking: bool = False,
) -> dict[str, Any] | None:
    """Tokenize a (prompt messages, response text) pair into a video_mc-utils/text_mc-utils/dp_actor.py-compatible sample: full input_ids (prompt+response), a response_mask over just the response tokens, mm_token_type_ids, position_ids, and the per-sample multimodal tensors (not yet padded/concatenated — that's build_multimodal_batch's job)."""
    video_paths = extract_video_paths(messages)
    video_tensor, video_meta = process_video({"video": video_paths}, image_patch_size=16, return_video_metadata=True)

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    model_inputs = processor(
        text=[prompt_text], videos=[video_tensor],
        videos_kwargs={"video_metadata": [video_meta], "do_sample_frames": False},
        return_tensors="pt",
    )
    prompt_ids = model_inputs["input_ids"][0]
    prompt_mm_type = model_inputs["mm_token_type_ids"][0]

    response_ids_list = processor.tokenizer(response_text, add_special_tokens=False)["input_ids"]
    if not response_ids_list:
        return None
    response_ids = torch.tensor(response_ids_list, dtype=prompt_ids.dtype)

    full_ids = torch.cat([prompt_ids, response_ids])
    response_mask = torch.cat([torch.zeros_like(prompt_ids), torch.ones_like(response_ids)])
    mm_token_type_ids = torch.cat([prompt_mm_type, torch.zeros_like(response_ids)])
    attention_mask = torch.ones_like(full_ids)

    prompt_attention_mask = torch.ones_like(prompt_ids)
    vision_position_ids = get_rope_index(
        processor, input_ids=prompt_ids, video_grid_thw=model_inputs.get("video_grid_thw"),
        attention_mask=prompt_attention_mask,
    )  # (3, prompt_len)
    text_position_ids = torch.arange(prompt_ids.shape[0]).unsqueeze(0)  # (1, prompt_len)
    prompt_position_ids = torch.cat([text_position_ids, vision_position_ids], dim=0)  # (4, prompt_len)
    # response tokens are plain text: all 4 rows just keep counting up from the prompt's last position
    resp_start = prompt_position_ids[:, -1:] + 1
    response_position_ids = resp_start + torch.arange(response_ids.shape[0]).unsqueeze(0)
    position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=1)  # (4, full_len)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "position_ids": position_ids,
        "pixel_values_videos": model_inputs["pixel_values_videos"],
        "video_grid_thw": model_inputs["video_grid_thw"],
    }


def _left_pad(t: torch.Tensor, target_len: int, pad_value, last_dim: int = -1) -> torch.Tensor:
    cur_len = t.shape[last_dim]
    if cur_len == target_len:
        return t
    pad_shape = list(t.shape)
    pad_shape[last_dim] = target_len - cur_len
    pad = torch.full(pad_shape, pad_value, dtype=t.dtype)
    return torch.cat([pad, t], dim=last_dim)


def build_multimodal_batch(samples: list[dict], pad_token_id: int) -> dict[str, Any]:
    max_len = max(s["input_ids"].shape[-1] for s in samples)

    input_ids = torch.stack([_left_pad(s["input_ids"], max_len, pad_token_id) for s in samples])
    attention_mask = torch.stack([_left_pad(s["attention_mask"], max_len, 0) for s in samples])
    response_mask = torch.stack([_left_pad(s["response_mask"], max_len, 0) for s in samples])
    mm_token_type_ids = torch.stack([_left_pad(s["mm_token_type_ids"], max_len, 0) for s in samples])
    # position_ids: (4, seq_len) per sample -> pad last dim -> stack -> (bsz, 4, seq_len) -> transpose -> (4, bsz, seq_len)
    position_ids = torch.stack([_left_pad(s["position_ids"], max_len, 1, last_dim=-1) for s in samples])
    position_ids = position_ids.transpose(0, 1)

    per_sample_mm = [
        {"pixel_values_videos": s["pixel_values_videos"], "video_grid_thw": s["video_grid_thw"]} for s in samples
    ]
    multi_modal_inputs = extract_multi_modal_inputs(per_sample_mm)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "position_ids": position_ids,
        "multi_modal_inputs": multi_modal_inputs,
    }


def forward_multimodal_hidden_states(model, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    """Returns hidden_states (tuple of (bsz, seq_len, hidden) per layer, len = 1 + num_layers).

    logits_to_keep=1 (2026-07-27 OOM incident): the model's default logits_to_keep=0 means
    "keep ALL positions" (slice(-0, None) == slice(0, None)), not "keep none" as the name
    suggests -- so every call here was silently computing a (batch, full_seq_len, vocab=248320)
    logits tensor we never read (only .hidden_states is used), on top of the already-large
    hidden_states themselves. For a long single sample this alone was several GiB and was the
    actual cause of a teacher-side OOM in Stage 1 that batch-size/token-budget tuning couldn't
    fix (a lone long sample can't be split smaller). logits_to_keep=1 keeps only the last
    position's logits (computed but discarded) without touching hidden_states at all.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            mm_token_type_ids=batch["mm_token_type_ids"].to(device),
            position_ids=batch["position_ids"].to(device),
            **{k: v.to(device) for k, v in batch["multi_modal_inputs"].items()},
            use_cache=False,
            output_hidden_states=True,
            logits_to_keep=1,
        )
    return outputs.hidden_states
