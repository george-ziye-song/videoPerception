#!/usr/bin/env python3
"""Pre-experiment 2: train student projectors P_S with frozen teacher PCA bases P_T.

Frozen student + teacher weights. Only P_S is optimized:
    L = ||P_S h_S - sg(P_T h_T)||^2
in the r-dimensional subspace, using proportional layer mapping.

Subspace modes:
  full (default): P_T from teacher PCA/SVD, frozen; train P_S only.
  direct: both P_S and P_T are trainable linear layers; no teacher SVD.
  residual: freeze head bridge, PCA on teacher residual for tail P_T.

Example:
  python scripts/analysis/cross_arch_preexp2_train_ps.py \\
    --responses-jsonl outputs/cross_arch_preexp1/on_policy_pairs.jsonl \\
    --student-model-path /path/to/Qwen3-1.7B \\
    --teacher-model-path /path/to/Qwen3-4B \\
    --output-dir outputs/cross_arch_preexp2 \\
    --ranks 256 512 \\
    --epochs 20

  # Joint trainable P_S + P_T (no teacher PCA):
  python scripts/analysis/cross_arch_preexp2_train_ps.py ... --subspace-mode direct

  # MLP student bridge (hidden = 4 * rank):
  python scripts/analysis/cross_arch_preexp2_train_ps.py ... --projector mlp --mlp-hidden-mult 4
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cross_arch_repr_analysis import (  # noqa: E402
    build_batch_tensors,
    load_causal_lm,
    make_dynamic_batches,
    make_dynamic_batches_video,
    normalize_raw_prompt,
    proportional_layer_map,
    resolve_dtype,
    set_seed,
)


@dataclass
class LayerPairSpec:
    student_layer: int
    teacher_layer: int


@dataclass
class FrozenTeacherProjector:
    weight: Tensor  # (r, d_teacher), maps centered h_T -> z_T
    mean: Tensor  # (d_teacher,)


@dataclass
class TeacherPcaBasis:
    """Full PCA basis for a layer; slice top-r rows to build P_T without re-running SVD."""

    mean: np.ndarray  # (d_teacher,)
    components: np.ndarray  # (d_teacher, d_teacher), rows = principal directions


class StudentProjectorBank(nn.Module):
    """Per layer-pair student bridge P_S: linear or 2-layer MLP bottleneck."""

    def __init__(
        self,
        student_dim: int,
        rank: int,
        layer_pairs: list[LayerPairSpec],
        *,
        projector_type: str = "linear",
        mlp_hidden_mult: int = 4,
    ) -> None:
        super().__init__()
        self.layer_pairs = layer_pairs
        self.rank = rank
        self.student_dim = student_dim
        self.projector_type = projector_type
        self.mlp_hidden_mult = mlp_hidden_mult
        self.projectors = nn.ModuleDict(
            {
                f"s{s.student_layer}_t{s.teacher_layer}": self._build_projector(
                    student_dim, rank, projector_type, mlp_hidden_mult
                )
                for s in layer_pairs
            }
        )

    @staticmethod
    def _build_projector(
        student_dim: int,
        rank: int,
        projector_type: str,
        mlp_hidden_mult: int,
    ) -> nn.Module:
        if projector_type == "linear":
            return nn.Linear(student_dim, rank, bias=False)
        if projector_type == "mlp":
            hidden_dim = max(rank, mlp_hidden_mult * rank)
            return nn.Sequential(
                nn.Linear(student_dim, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, rank, bias=False),
            )
        raise ValueError(f"Unsupported projector_type={projector_type!r}; expected 'linear' or 'mlp'")

    def forward_layer(self, student_layer: int, teacher_layer: int, h_student: Tensor) -> Tensor:
        key = f"s{student_layer}_t{teacher_layer}"
        return self.projectors[key](h_student)


def build_rank_output_dir(
    output_dir: Path,
    rank: int,
    *,
    projector_type: str,
    subspace_mode: str = "full",
    head_rank: int | None = None,
) -> Path:
    suffix = "" if projector_type == "linear" else "_mlp"
    if subspace_mode == "residual":
        if head_rank is None:
            raise ValueError("head_rank is required for residual output dirs")
        return output_dir / f"residual_head{head_rank}_tail{rank}{suffix}"
    if subspace_mode == "direct":
        return output_dir / f"rank_{rank}_direct"
    return output_dir / f"rank_{rank}{suffix}"


def load_all_pairs_from_jsonl(responses_jsonl: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    with open(responses_jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            raw_prompt = row.get("raw_prompt", row.get("prompt"))
            pairs.append({"raw_prompt": raw_prompt, "response": row["response"]})
    if not pairs:
        raise ValueError(f"No pairs found in {responses_jsonl}")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-experiment 2: train P_S with frozen P_T (teacher PCA)")
    parser.add_argument("--responses-jsonl", type=str, required=True)
    parser.add_argument(
        "--modality", type=str, default="text", choices=["text", "video"],
        help="video: pairs came from cross_arch_repr_analysis.py --modality video; uses a shared "
        "processor instead of a tokenizer and cross_arch_mm_utils for batch/forward. Default text path unchanged.",
    )
    parser.add_argument("--student-model-path", type=str, required=True)
    parser.add_argument("--teacher-model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[256, 512])
    parser.add_argument(
        "--subspace-mode",
        type=str,
        default="full",
        choices=["full", "direct", "residual"],
        help="full: frozen teacher PCA P_T + trainable P_S. "
        "direct: trainable linear P_S and P_T jointly (no teacher SVD). "
        "residual: freeze head bridge, train tail on orthogonal complement.",
    )
    parser.add_argument("--head-rank", type=int, default=16, help="Head rank for residual mode.")
    parser.add_argument(
        "--head-checkpoint",
        type=str,
        default=None,
        help="Head ps_bank.pt for residual mode (e.g. outputs/.../rank_16/ps_bank.pt).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--last-k", type=int, default=2000)
    parser.add_argument(
        "--position-mode",
        type=str,
        default="last_k",
        choices=["last_k", "first_k"],
        help="Use trailing last_k or leading first_k response tokens for PCA/training.",
    )
    parser.add_argument(
        "--first-k",
        type=int,
        default=50,
        help="Leading response tokens when --position-mode=first_k (matches OPD rep_distillation_first_k).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=65536)
    parser.add_argument(
        "--max-vision-patches",
        type=int,
        default=19000,
        help="Video-mode only: sum of raw vision-tower patches (pre spatial-merge) allowed per batch. "
        "Previously hardcoded to 20000 inside _video_batches with no CLI override. Default matches "
        "Stage 0's empirically-calibrated teacher budget (see partitioned-wandering-deer.md) since "
        "student+teacher are forwarded through the SAME batch here -- the tighter (teacher) constraint "
        "binds regardless of which model's weights happen to be more memory-constrained on a given GPU.",
    )
    parser.add_argument(
        "--max-pca-rows",
        type=int,
        default=16384,
        help="Subsample teacher rows for PCA / P_T (smaller = faster, 16k is usually enough)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=5, help="Run validation every N epochs (plus first/last)")
    parser.add_argument(
        "--compute-probe-cosine",
        action="store_true",
        help="Also compute linear_probe_cosine (slow: ridge+SVD per batch per layer). Off by default.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0", help="Student model device (also default for teacher if --teacher-device is not given)")
    parser.add_argument(
        "--teacher-device",
        type=str,
        default=None,
        help="Teacher model device, if different from --device. Needed when student+teacher weights "
        "don't both fit on one GPU (e.g. Qwen3.5-4B + Qwen3.5-9B = ~26.7GB > a single 24G 4090). "
        "Safe to split across GPUs because neither model needs gradients here -- only the small "
        "StudentProjectorBank is trained, via each model's frozen forward pass values, so the only "
        "cross-device cost is moving the teacher's hidden states over before comparing (no backward "
        "pass ever crosses devices).",
    )
    parser.add_argument(
        "--extra-device-pairs",
        type=str,
        default=None,
        help="Video mode only: additional (student_device,teacher_device) pairs for data-parallel "
        "training beyond the primary --device/--teacher-device pair. Format: comma-separated "
        "'student-teacher' pairs of bare GPU indices, e.g. '2-3' or '2-3,4-5'. Each pair loads its "
        "own full student+teacher replica (frozen, no gradients -- see --teacher-device) and only "
        "the expensive forward passes are parallelized; ps_bank/optimizer stay single-copy on the "
        "primary device, so there's no gradient-sync/DDP complexity.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--layer-mode", type=str, default="all", choices=["all", "last", "mid"])
    parser.add_argument(
        "--projector",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="Student bridge P_S: linear (default) or 2-layer MLP bottleneck.",
    )
    parser.add_argument(
        "--mlp-hidden-mult",
        type=int,
        default=4,
        help="MLP hidden dim = max(rank, mlp_hidden_mult * rank). Ignored for linear.",
    )
    return parser.parse_args()


def select_layer_pairs(num_student_layers: int, num_teacher_layers: int, layer_mode: str) -> list[LayerPairSpec]:
    teacher_by_student = proportional_layer_map(num_student_layers, num_teacher_layers)
    pairs = [
        LayerPairSpec(student_layer=s_idx, teacher_layer=teacher_by_student[s_idx])
        for s_idx in range(num_student_layers)
    ]
    if layer_mode == "all":
        return pairs
    if layer_mode == "last":
        return [pairs[-1]]
    mid = num_student_layers // 2
    return [pairs[mid]]


def subsample_rows_np(matrix: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if matrix.shape[0] <= max_rows:
        return matrix
    rng = np.random.default_rng(seed)
    idx = rng.choice(matrix.shape[0], size=max_rows, replace=False)
    return matrix[idx]


def fit_teacher_pca_basis(h_teacher: np.ndarray, *, device: torch.device | None = None) -> TeacherPcaBasis:
    """Fast PCA via d×d covariance eigendecomposition (N >> d). One SVD/eigh per layer total.

    device (2026-07-27): np.linalg.eigh on the (teacher_dim, teacher_dim) covariance matrix is
    O(dim^3), independent of dataset size -- confirmed by direct measurement, ~4.5-5s/layer x 32
    layers (~2.5-3min) on this shared server's host CPU even on a 100-sample smoke test, since the
    cost is fixed by teacher_dim (e.g. 4096) alone, not sample count. Same root cause and fix as
    cross_arch_repr_analysis.py's pca_basis/linear_cka this session (see
    partitioned-wandering-deer.md): host CPU on this machine is shared with other tenants, GPUs
    are not, and this matrix is tiny relative to model-weight memory -- torch.linalg.eigh on GPU
    sidesteps the contention and is dramatically faster at this size. Falls back to the original
    numpy path when device is None or CPU-only, so behavior is unchanged if no GPU is available."""
    if h_teacher.shape[0] < 2:
        raise ValueError("Need at least 2 teacher rows to fit PCA")
    mean = h_teacher.mean(axis=0, keepdims=False).astype(np.float32)
    centered = h_teacher - mean
    n = centered.shape[0]
    if device is not None and device.type == "cuda":
        centered_t = torch.from_numpy(np.ascontiguousarray(centered)).to(device=device, dtype=torch.float32)
        cov_t = (centered_t.T @ centered_t) / max(n - 1, 1)
        eigvals_t, eigvecs_t = torch.linalg.eigh(cov_t)
        eigvals = eigvals_t.cpu().numpy()
        eigvecs = eigvecs_t.cpu().numpy()
    else:
        # Covariance (dim, dim) is much cheaper than thin-SVD on (N, dim) when N is large.
        cov = (centered.T @ centered) / max(n - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order].T.astype(np.float32)
    return TeacherPcaBasis(mean=mean, components=components)


def frozen_projector_from_basis(basis: TeacherPcaBasis, rank: int) -> FrozenTeacherProjector:
    rank = min(rank, basis.components.shape[0], basis.components.shape[1])
    weight = basis.components[:rank]
    return FrozenTeacherProjector(
        weight=torch.from_numpy(weight.copy()),
        mean=torch.from_numpy(basis.mean.copy()),
    )


def build_frozen_pts_for_rank(
    pca_bases: dict[tuple[int, int], TeacherPcaBasis],
    rank: int,
) -> dict[tuple[int, int], FrozenTeacherProjector]:
    return {key: frozen_projector_from_basis(basis, rank) for key, basis in pca_bases.items()}


def project_teacher(frozen_pt: FrozenTeacherProjector, h_teacher: Tensor) -> Tensor:
    centered = h_teacher - frozen_pt.mean.to(h_teacher.device, dtype=h_teacher.dtype)
    return centered @ frozen_pt.weight.T.to(h_teacher.device, dtype=h_teacher.dtype)


def subtract_rowspace_projection_torch(
    hidden: Tensor,
    weight: Tensor,
    *,
    mean: Tensor | None = None,
) -> Tensor:
    orig_shape = hidden.shape
    flat = hidden.reshape(-1, hidden.shape[-1])
    if mean is not None:
        flat = flat - mean.to(device=flat.device, dtype=flat.dtype).unsqueeze(0)
    weight = weight.to(device=flat.device, dtype=flat.dtype)
    coords = flat @ weight.T
    parallel = coords @ weight
    return (flat - parallel).reshape(orig_shape)


def load_head_projectors_from_checkpoint(
    checkpoint_path: str | Path,
    layer_pairs: list[LayerPairSpec],
    *,
    head_rank: int,
) -> tuple[dict[tuple[int, int], FrozenTeacherProjector], dict[tuple[int, int], Tensor]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("projector_type", "linear") != "linear":
        raise ValueError(
            f"Residual head bridge requires a linear ps_bank checkpoint, got "
            f"projector_type={checkpoint.get('projector_type')!r} in {checkpoint_path}"
        )
    state_dict = checkpoint.get("state_dict", {})
    frozen_weights = checkpoint.get("frozen_pt_weights", {})
    frozen_means = checkpoint.get("frozen_pt_means", {})
    frozen_pts: dict[tuple[int, int], FrozenTeacherProjector] = {}
    head_ps: dict[tuple[int, int], Tensor] = {}
    for spec in layer_pairs:
        key = (spec.student_layer, spec.teacher_layer)
        layer_keys = [f"s{spec.student_layer}_t{spec.teacher_layer}", f"{spec.student_layer}_{spec.teacher_layer}"]
        for lk in layer_keys:
            sk = f"projectors.{lk}.weight"
            if sk in state_dict:
                head_ps[key] = state_dict[sk][:head_rank].clone()
                break
        for lk in layer_keys:
            if lk in frozen_weights:
                frozen_pts[key] = FrozenTeacherProjector(
                    weight=frozen_weights[lk][:head_rank].clone(),
                    mean=frozen_means.get(lk, torch.zeros(frozen_weights[lk].shape[1])).clone(),
                )
                break
        if key not in frozen_pts:
            raise ValueError(f"Missing head P_T for layer pair {key} in {checkpoint_path}")
    return frozen_pts, head_ps


def accumulate_teacher_residual_rows_for_pca(
    teacher_model,
    tokenizer,
    pairs: list[dict[str, Any]],
    layer_pairs: list[LayerPairSpec],
    head_frozen_pts: dict[tuple[int, int], FrozenTeacherProjector],
    *,
    device: torch.device,
    batch_size: int,
    max_batch_tokens: int,
    position_mode: str,
    last_k: int,
    first_k: int,
    enable_thinking: bool,
    max_pca_rows: int,
    seed: int,
) -> dict[tuple[int, int], np.ndarray]:
    teacher_rows: dict[tuple[int, int], list[np.ndarray]] = {
        (spec.student_layer, spec.teacher_layer): [] for spec in layer_pairs
    }
    pos_kw = response_position_kwargs(
        position_mode=position_mode, last_k=last_k, first_k=first_k
    )
    batches = make_dynamic_batches(
        pairs,
        tokenizer,
        enable_thinking=enable_thinking,
        max_batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
    )
    for batch_pairs in tqdm(batches, desc="Collect teacher residual rows for tail P_T", unit="batch"):
        input_ids, attention_mask, response_mask = build_batch_tensors(
            batch_pairs,
            tokenizer,
            last_k=last_k,
            enable_thinking=enable_thinking,
            device=device,
        )
        teacher_layers = forward_per_layer_hidden(teacher_model, input_ids, attention_mask)
        for spec in layer_pairs:
            key = (spec.student_layer, spec.teacher_layer)
            h_t = extract_response_hidden_tensor(
                teacher_layers[spec.teacher_layer],
                response_mask,
                **pos_kw,
            )
            if h_t.shape[0] == 0:
                continue
            frozen_pt = head_frozen_pts[key]
            residual = subtract_rowspace_projection_torch(
                h_t,
                frozen_pt.weight.to(device),
                mean=frozen_pt.mean.to(device),
            )
            teacher_rows[key].append(residual.cpu().numpy())

    merged: dict[tuple[int, int], np.ndarray] = {}
    for key, chunks in teacher_rows.items():
        if not chunks:
            raise ValueError(f"No teacher residual rows collected for layer pair {key}")
        merged[key] = subsample_rows_np(np.concatenate(chunks, axis=0), max_pca_rows, seed)
    return merged


def response_position_kwargs(
    *,
    position_mode: str,
    last_k: int,
    first_k: int,
) -> dict[str, Any]:
    return {
        "position_mode": position_mode,
        "last_k": last_k,
        "first_k": first_k,
    }


def response_position_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return response_position_kwargs(
        position_mode=args.position_mode,
        last_k=args.last_k,
        first_k=args.first_k,
    )


@dataclass
class BestValTracker:
    """Track and restore projector weights with the best val subspace cosine."""

    score_key: str = "subspace_cosine_mean"
    best_score: float = float("-inf")
    best_epoch: int = 0
    best_metrics: dict[str, float] | None = None
    best_ps_state: dict[str, Tensor] | None = None
    best_pt_state: dict[str, Tensor] | None = None

    def maybe_update(
        self,
        epoch: int,
        val_metrics: dict[str, float],
        ps_bank: nn.Module,
        *,
        pt_bank: nn.Module | None = None,
    ) -> bool:
        score = val_metrics.get(self.score_key, float("nan"))
        if not np.isfinite(score) or score <= self.best_score:
            return False
        self.best_score = float(score)
        self.best_epoch = epoch
        self.best_metrics = dict(val_metrics)
        self.best_ps_state = copy.deepcopy(ps_bank.state_dict())
        self.best_pt_state = (
            copy.deepcopy(pt_bank.state_dict()) if pt_bank is not None else None
        )
        return True

    def restore_best(self, ps_bank: nn.Module, *, pt_bank: nn.Module | None = None) -> bool:
        if self.best_ps_state is None:
            return False
        ps_bank.load_state_dict(self.best_ps_state)
        if pt_bank is not None and self.best_pt_state is not None:
            pt_bank.load_state_dict(self.best_pt_state)
        return True


def slice_response_hidden_rows(
    h: Tensor,
    *,
    position_mode: str,
    last_k: int,
    first_k: int,
) -> Tensor:
    k = first_k if position_mode == "first_k" else last_k
    if k <= 0 or h.shape[0] <= k:
        return h
    if position_mode == "first_k":
        return h[:k]
    return h[-k:]


def extract_response_hidden_tensor(
    hidden_state: Tensor,
    response_mask: Tensor,
    *,
    position_mode: str = "last_k",
    last_k: int,
    first_k: int = 50,
) -> Tensor:
    rows = []
    for batch_idx in range(hidden_state.shape[0]):
        valid = response_mask[batch_idx].bool()
        if not valid.any():
            continue
        # .float(): text-mode callers already get fp32 hidden states (forward_per_layer_hidden
        # converts upfront), but video-mode's forward_multimodal_hidden_states intentionally
        # returns raw bf16 (memory efficiency, see cross_arch_mm_utils.py) -- numpy has no
        # native bfloat16 support, so .cpu().numpy() on an unconverted video-mode tensor at the
        # call site fails with "unsupported ScalarType BFloat16". Converting here (after the
        # boolean-mask indexing already narrowed to just the response rows) is a no-op for the
        # already-fp32 text-mode path and fixes video mode, matching Stage 0's established
        # extract_response_hidden_rows convention.
        h = hidden_state[batch_idx, valid].float()
        h = slice_response_hidden_rows(
            h,
            position_mode=position_mode,
            last_k=last_k,
            first_k=first_k,
        )
        rows.append(h)
    if not rows:
        return hidden_state.new_zeros((0, hidden_state.shape[-1]))
    return torch.cat(rows, dim=0)


@torch.no_grad()
def forward_per_layer_hidden(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> list[Tensor]:
    model.eval()
    # Explicit output_hidden_states=True (not just model.config.output_hidden_states set at
    # load time) -- matches forward_multimodal_hidden_states in cross_arch_mm_utils.py, which
    # already does this and is why Stage 0 never hit this. For Qwen3.5's AutoModelForImageTextToText
    # wrapper, calling with only input_ids/attention_mask (no vision inputs, as the probe step and
    # text-mode call sites do) silently drops hidden_states unless passed as a call-time kwarg.
    # logits_to_keep=1: default (0) computes logits for the FULL sequence (slice(-0,None) ==
    # slice(0,None)), a large, unused (batch,seq_len,vocab=248320) tensor we never read here --
    # see forward_multimodal_hidden_states in cross_arch_mm_utils.py for the same fix and the
    # OOM it was traced to.
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True,
        logits_to_keep=1,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states; set config.output_hidden_states=True")
    return [hs.float() for hs in hidden_states[1:]]


def _video_batches(
    pairs: list[dict[str, Any]],
    processor,
    batch_size: int,
    enable_thinking: bool,
    max_batch_tokens: int = 32768,
    max_vision_patches: int = 20000,
):
    """video modality equivalent of make_dynamic_batches: build every sample once
    (needed to know its length), then pack via the same token-budget dynamic
    batching Stage 0 uses (make_dynamic_batches_video), instead of fixed-size chunking.
    Defaults kept conservative (matching Stage 0's teacher numbers): Stage 1 co-resides
    student+teacher on the same GPU, so it has less headroom than either alone did in
    Stage 0's empirical probes, not more. Retune with real measurements before scaling
    Stage 1 up (see partitioned-wandering-deer.md's Stage 0 batching incident)."""
    from cross_arch_mm_utils import build_multimodal_sample

    built = []
    for pair in tqdm(pairs, desc="tokenize (video)", unit="sample"):
        messages = normalize_raw_prompt(pair["raw_prompt"])
        s = build_multimodal_sample(messages, str(pair["response"]), processor, enable_thinking=enable_thinking)
        if s is not None:
            built.append(s)
    return make_dynamic_batches_video(
        built, max_batch_size=batch_size, max_batch_tokens=max_batch_tokens, max_vision_patches=max_vision_patches
    )


def _forward_pair_to_primary(student_model, teacher_model, samples, processor, student_device, teacher_device, primary_device):
    """Like _forward_both_video, but for multi-pair data-parallel training: moves BOTH
    student_layers and teacher_layers to primary_device (not just onto student_device),
    since with multiple independent (student,teacher) GPU pairs, a pair's own student_device
    generally is NOT the primary device where ps_bank/frozen_pts/optimizer live (only pair 0's
    is). Called back-to-back for each pair with no synchronization in between (no .cpu()/.item()
    calls) so CUDA's per-device async dispatch lets the pairs' forward passes genuinely overlap
    in wall-clock time -- a plain sequential-looking Python loop over pairs still achieves real
    multi-GPU parallelism as long as nothing forces a sync between iterations."""
    from cross_arch_mm_utils import build_multimodal_batch, forward_multimodal_hidden_states

    batch = build_multimodal_batch(samples, processor.tokenizer.pad_token_id)
    student_layers = forward_multimodal_hidden_states(student_model, batch, student_device)[1:]
    teacher_layers = forward_multimodal_hidden_states(teacher_model, batch, teacher_device)[1:]
    student_layers = tuple(t.to(primary_device) for t in student_layers)
    teacher_layers = tuple(t.to(primary_device) for t in teacher_layers)
    response_mask = batch["response_mask"].to(primary_device)
    return student_layers, teacher_layers, response_mask


def _forward_both_video(student_model, teacher_model, samples, processor, student_device, teacher_device=None):
    """student_device/teacher_device may differ (2-GPU split when both models' weights
    don't fit on one card together). build_multimodal_batch returns plain CPU tensors;
    forward_multimodal_hidden_states does its own .to(device) per call, so the same
    CPU batch can be forwarded through each model on its own device independently.
    Teacher's returned hidden states (and response_mask, used to index both) are moved
    to student_device afterwards so callers never need to know a split happened."""
    from cross_arch_mm_utils import build_multimodal_batch, forward_multimodal_hidden_states

    if teacher_device is None:
        teacher_device = student_device
    batch = build_multimodal_batch(samples, processor.tokenizer.pad_token_id)
    student_layers = (
        forward_multimodal_hidden_states(student_model, batch, student_device)[1:] if student_model else None
    )
    teacher_layers = (
        forward_multimodal_hidden_states(teacher_model, batch, teacher_device)[1:] if teacher_model else None
    )
    if teacher_layers is not None:
        teacher_layers = tuple(t.to(student_device) for t in teacher_layers)
    return student_layers, teacher_layers, batch["response_mask"].to(student_device)


def build_hidden_cache_video(
    batches,
    all_pairs: list[tuple[Any, Any, torch.device, torch.device]],
    layer_pairs: list[LayerPairSpec],
    processor,
    *,
    position_mode: str,
    last_k: int,
    first_k: int,
    primary_device: torch.device,
    desc: str = "Precompute hidden states",
) -> list[dict[tuple[int, int], tuple[Tensor, Tensor]]]:
    """One-time forward pass over `batches` (student+teacher, frozen/no_grad), response-position
    sliced per layer pair and cached to CPU. Both models are always frozen in this script (only
    the small StudentProjectorBank is ever trained), so a given batch's (h_s, h_t) is IDENTICAL
    every time it's forwarded -- the old code re-ran this forward once for PCA-row collection and
    then AGAIN every single epoch (20x), even though nothing about the backbone output ever
    changes between those calls. Computing it here once and reading it back from CPU is
    equivalent, just without the redundant compute (2026-07-27, see partitioned-wandering-deer.md).
    Caches only the small last_k-sliced (h_s, h_t) per layer pair (not the full (batch, seq, dim)
    hidden states) -- for the full 3323-sample dataset this is ~10GB fp32 on CPU, comfortably
    within host RAM, vs. re-running two 4B/9B forward passes (incl. vision tower) 20+ times.
    Reuses the same round-robin multi-pair dispatch as training so this also benefits from
    --extra-device-pairs GPU parallelism; works unchanged for the single-pair case (num_pairs=1)."""
    pos_kw = response_position_kwargs(position_mode=position_mode, last_k=last_k, first_k=first_k)
    num_pairs = len(all_pairs)
    rounds = [batches[i : i + num_pairs] for i in range(0, len(batches), num_pairs)]
    cache: list[dict[tuple[int, int], tuple[Tensor, Tensor]]] = []
    with torch.no_grad():
        for round_batches in tqdm(rounds, desc=desc, unit="round"):
            for i, batch_pairs in enumerate(round_batches):
                s_model, t_model, s_dev, t_dev = all_pairs[i]
                student_layers, teacher_layers, response_mask = _forward_pair_to_primary(
                    s_model, t_model, batch_pairs, processor, s_dev, t_dev, primary_device
                )
                entry: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
                for spec in layer_pairs:
                    key = (spec.student_layer, spec.teacher_layer)
                    h_s = extract_response_hidden_tensor(
                        student_layers[spec.student_layer], response_mask, **pos_kw
                    ).cpu()
                    h_t = extract_response_hidden_tensor(
                        teacher_layers[spec.teacher_layer], response_mask, **pos_kw
                    ).cpu()
                    entry[key] = (h_s, h_t)
                cache.append(entry)
    return cache


def load_or_build_hidden_cache_video(
    cache_path: Path,
    batches,
    all_pairs: list[tuple[Any, Any, torch.device, torch.device]],
    layer_pairs: list[LayerPairSpec],
    processor,
    *,
    position_mode: str,
    last_k: int,
    first_k: int,
    primary_device: torch.device,
    desc: str,
) -> list[dict[tuple[int, int], tuple[Tensor, Tensor]]]:
    """Disk-persisted wrapper around build_hidden_cache_video (2026-07-27 checkpointing
    requirement: a killed/reclaimed-GPU run must not lose this ~1hr one-time precompute --
    resume_checkpoint.pt alone isn't enough since it only covers the ps_bank/optimizer state
    inside train_one_rank, which now depends on this cache existing). Atomic write (tmp + os.replace)
    matches resume_checkpoint.pt's convention so a kill mid-write can't leave a truncated cache
    that a later run would try to load."""
    if cache_path.exists():
        print(f"Loading cached hidden states from {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("num_batches") != len(batches):
            raise RuntimeError(
                f"{cache_path} has {payload.get('num_batches')} cached batches but the current "
                f"data/batching config produced {len(batches)} -- stale cache from a different "
                f"dataset/--batch-size/--max-batch-tokens/--max-vision-patches. Delete {cache_path} "
                "and rerun."
            )
        return payload["cache"]
    cache = build_hidden_cache_video(
        batches, all_pairs, layer_pairs, processor,
        position_mode=position_mode, last_k=last_k, first_k=first_k,
        primary_device=primary_device, desc=desc,
    )
    tmp_path = cache_path.with_suffix(".pt.tmp")
    torch.save({"num_batches": len(batches), "cache": cache}, tmp_path)
    os.replace(tmp_path, cache_path)
    print(f"Saved hidden-state cache to {cache_path}")
    return cache


def accumulate_teacher_rows_for_pca(
    teacher_model,
    tokenizer,
    pairs: list[dict[str, Any]],
    layer_pairs: list[LayerPairSpec],
    *,
    device: torch.device,
    batch_size: int,
    max_batch_tokens: int,
    position_mode: str,
    last_k: int,
    first_k: int,
    enable_thinking: bool,
    max_pca_rows: int,
    seed: int,
    modality: str = "text",
    processor=None,
    max_vision_patches: int = 19000,
    hidden_cache: list[dict[tuple[int, int], tuple[Tensor, Tensor]]] | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """hidden_cache (video mode only): precomputed via build_hidden_cache_video, one entry per
    train batch with response-position-sliced (h_s, h_t) per layer pair already on CPU. When
    given, skips re-running the teacher forward pass entirely -- see build_hidden_cache_video's
    docstring for why this is safe (frozen teacher => identical output, cache built from the
    exact same train batches this call would otherwise re-tokenize and re-forward)."""
    teacher_rows: dict[tuple[int, int], list[np.ndarray]] = {(
        spec.student_layer,
        spec.teacher_layer,
    ): [] for spec in layer_pairs}
    pos_kw = response_position_kwargs(
        position_mode=position_mode, last_k=last_k, first_k=first_k
    )

    if hidden_cache is not None:
        for entry in tqdm(hidden_cache, desc="Collect teacher rows for P_T (cached)", unit="batch"):
            for spec in layer_pairs:
                key = (spec.student_layer, spec.teacher_layer)
                _, h_t = entry[key]
                if h_t.shape[0] > 0:
                    teacher_rows[key].append(h_t.numpy())
        merged: dict[tuple[int, int], np.ndarray] = {}
        for key, chunks in teacher_rows.items():
            if not chunks:
                raise ValueError(f"No teacher hidden rows collected for layer pair {key}")
            merged[key] = subsample_rows_np(np.concatenate(chunks, axis=0), max_pca_rows, seed)
        return merged

    if modality == "video":
        batches = _video_batches(
            pairs, processor, batch_size, enable_thinking,
            max_batch_tokens=max_batch_tokens, max_vision_patches=max_vision_patches,
        )
    else:
        batches = make_dynamic_batches(
            pairs,
            tokenizer,
            enable_thinking=enable_thinking,
            max_batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
        )
    for batch_pairs in tqdm(batches, desc="Collect teacher rows for P_T", unit="batch"):
        if modality == "video":
            _, teacher_layers, response_mask = _forward_both_video(None, teacher_model, batch_pairs, processor, device)
        else:
            input_ids, attention_mask, response_mask = build_batch_tensors(
                batch_pairs,
                tokenizer,
                last_k=last_k,
                enable_thinking=enable_thinking,
                device=device,
            )
            teacher_layers = forward_per_layer_hidden(teacher_model, input_ids, attention_mask)
        for spec in layer_pairs:
            h_t = extract_response_hidden_tensor(
                teacher_layers[spec.teacher_layer],
                response_mask,
                **pos_kw,
            )
            if h_t.shape[0] > 0:
                teacher_rows[(spec.student_layer, spec.teacher_layer)].append(h_t.cpu().numpy())

    merged: dict[tuple[int, int], np.ndarray] = {}
    for key, chunks in teacher_rows.items():
        if not chunks:
            raise ValueError(f"No teacher hidden rows collected for layer pair {key}")
        merged[key] = subsample_rows_np(np.concatenate(chunks, axis=0), max_pca_rows, seed)
    return merged


def compute_subspace_cosine(z_student: Tensor, z_teacher: Tensor) -> float:
    if z_student.shape[0] == 0:
        return float("nan")
    a = z_student / (z_student.norm(dim=1, keepdim=True) + 1e-8)
    b = z_teacher / (z_teacher.norm(dim=1, keepdim=True) + 1e-8)
    return float((a * b).sum(dim=1).mean().item())


def compute_probe_cosine_from_rows(
    h_student: np.ndarray,
    h_teacher: np.ndarray,
    *,
    rank: int,
    max_rows: int = 4096,
    seed: int = 0,
) -> float:
    """One ridge+SVD per layer on subsampled rows (for optional comparison with pre-exp 1)."""
    from cross_arch_repr_analysis import fit_linear_probe_w

    n = min(h_student.shape[0], h_teacher.shape[0])
    if n < 2:
        return float("nan")
    h_s = subsample_rows_np(h_student[:n], max_rows, seed)
    h_t = subsample_rows_np(h_teacher[:n], max_rows, seed)
    w = fit_linear_probe_w(h_s, h_t, ridge_lambda=1e-4)
    r_eff = min(rank, w.shape[0], w.shape[1])
    if r_eff < 1:
        return float("nan")
    u, svals, vt = np.linalg.svd(w, full_matrices=False)
    w_r = (u[:, :r_eff] * svals[:r_eff]) @ vt[:r_eff]
    pred = h_s @ w_r.T
    a = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    b = h_t / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float((a * b).sum(axis=1).mean())


@torch.no_grad()
def evaluate_projectors(
    student_model,
    teacher_model,
    tokenizer,
    pairs: list[dict[str, Any]],
    layer_pairs: list[LayerPairSpec],
    frozen_pts: dict[tuple[int, int], FrozenTeacherProjector],
    ps_bank: StudentProjectorBank,
    *,
    device: torch.device,
    teacher_device: torch.device | None = None,
    batch_size: int,
    max_batch_tokens: int,
    position_mode: str,
    last_k: int,
    first_k: int,
    enable_thinking: bool,
    compute_probe_cosine: bool = False,
    modality: str = "text",
    processor=None,
    max_vision_patches: int = 19000,
    hidden_cache: list[dict[tuple[int, int], tuple[Tensor, Tensor]]] | None = None,
) -> dict[str, float]:
    """hidden_cache (video mode only): see accumulate_teacher_rows_for_pca's docstring -- same
    precomputed-forward-pass reuse, here for the (much smaller, but called up to ~8x per rank
    across before/eval-every/final/after) val set."""
    pos_kw = response_position_kwargs(
        position_mode=position_mode, last_k=last_k, first_k=first_k
    )
    subspace_cosines: list[float] = []
    subspace_mses: list[float] = []
    probe_row_buffers: dict[tuple[int, int], tuple[list[np.ndarray], list[np.ndarray]]] | None = (
        {(spec.student_layer, spec.teacher_layer): ([], []) for spec in layer_pairs}
        if compute_probe_cosine
        else None
    )

    if hidden_cache is not None:
        for entry in hidden_cache:
            for spec in layer_pairs:
                key = (spec.student_layer, spec.teacher_layer)
                h_s, h_t = entry[key]
                if h_s.shape[0] == 0:
                    continue
                h_s = h_s.to(device)
                h_t = h_t.to(device)
                z_t = project_teacher(frozen_pts[key], h_t)
                z_s = ps_bank.forward_layer(spec.student_layer, spec.teacher_layer, h_s)
                subspace_cosines.append(compute_subspace_cosine(z_s, z_t))
                subspace_mses.append(float(nn.functional.mse_loss(z_s, z_t).item()))
                if probe_row_buffers is not None:
                    s_buf, t_buf = probe_row_buffers[key]
                    s_buf.append(h_s.cpu().numpy())
                    t_buf.append(h_t.cpu().numpy())
        metrics = {
            "subspace_cosine_mean": float(np.mean(subspace_cosines)) if subspace_cosines else float("nan"),
            "subspace_mse_mean": float(np.mean(subspace_mses)) if subspace_mses else float("nan"),
        }
        if probe_row_buffers is not None:
            probe_cosines: list[float] = []
            for key in probe_row_buffers:
                s_chunks, t_chunks = probe_row_buffers[key]
                if not s_chunks:
                    continue
                probe_cosines.append(
                    compute_probe_cosine_from_rows(
                        np.concatenate(s_chunks, axis=0),
                        np.concatenate(t_chunks, axis=0),
                        rank=ps_bank.rank,
                    )
                )
            metrics["linear_probe_cosine_mean"] = float(np.mean(probe_cosines)) if probe_cosines else float("nan")
        return metrics

    if modality == "video":
        batches = _video_batches(
            pairs, processor, batch_size, enable_thinking,
            max_batch_tokens=max_batch_tokens, max_vision_patches=max_vision_patches,
        )
    else:
        batches = make_dynamic_batches(
            pairs,
            tokenizer,
            enable_thinking=enable_thinking,
            max_batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
        )
    for batch_pairs in batches:
        if modality == "video":
            student_layers, teacher_layers, response_mask = _forward_both_video(
                student_model, teacher_model, batch_pairs, processor, device, teacher_device
            )
        else:
            input_ids, attention_mask, response_mask = build_batch_tensors(
                batch_pairs,
                tokenizer,
                last_k=last_k,
                enable_thinking=enable_thinking,
                device=device,
            )
            student_layers = forward_per_layer_hidden(student_model, input_ids, attention_mask)
            teacher_layers = forward_per_layer_hidden(teacher_model, input_ids, attention_mask)

        for spec in layer_pairs:
            key = (spec.student_layer, spec.teacher_layer)
            h_s = extract_response_hidden_tensor(
                student_layers[spec.student_layer],
                response_mask,
                **pos_kw,
            )
            h_t = extract_response_hidden_tensor(
                teacher_layers[spec.teacher_layer],
                response_mask,
                **pos_kw,
            )
            if h_s.shape[0] == 0:
                continue
            z_t = project_teacher(frozen_pts[key], h_t)
            z_s = ps_bank.forward_layer(spec.student_layer, spec.teacher_layer, h_s)
            subspace_cosines.append(compute_subspace_cosine(z_s, z_t))
            subspace_mses.append(float(nn.functional.mse_loss(z_s, z_t).item()))
            if probe_row_buffers is not None:
                s_buf, t_buf = probe_row_buffers[key]
                s_buf.append(h_s.cpu().numpy())
                t_buf.append(h_t.cpu().numpy())

    metrics = {
        "subspace_cosine_mean": float(np.mean(subspace_cosines)) if subspace_cosines else float("nan"),
        "subspace_mse_mean": float(np.mean(subspace_mses)) if subspace_mses else float("nan"),
    }
    if probe_row_buffers is not None:
        probe_cosines: list[float] = []
        for key in probe_row_buffers:
            s_chunks, t_chunks = probe_row_buffers[key]
            if not s_chunks:
                continue
            probe_cosines.append(
                compute_probe_cosine_from_rows(
                    np.concatenate(s_chunks, axis=0),
                    np.concatenate(t_chunks, axis=0),
                    rank=ps_bank.rank,
                )
            )
        metrics["linear_probe_cosine_mean"] = float(np.mean(probe_cosines)) if probe_cosines else float("nan")
    return metrics


@torch.no_grad()
def evaluate_direct_projectors(
    student_model,
    teacher_model,
    tokenizer,
    pairs: list[dict[str, Any]],
    layer_pairs: list[LayerPairSpec],
    ps_bank: StudentProjectorBank,
    pt_bank: StudentProjectorBank,
    *,
    device: torch.device,
    batch_size: int,
    max_batch_tokens: int,
    position_mode: str,
    last_k: int,
    first_k: int,
    enable_thinking: bool,
    compute_probe_cosine: bool = False,
) -> dict[str, float]:
    pos_kw = response_position_kwargs(
        position_mode=position_mode, last_k=last_k, first_k=first_k
    )
    ps_bank.eval()
    pt_bank.eval()
    subspace_cosines: list[float] = []
    subspace_mses: list[float] = []
    probe_row_buffers: dict[tuple[int, int], tuple[list[np.ndarray], list[np.ndarray]]] | None = (
        {(spec.student_layer, spec.teacher_layer): ([], []) for spec in layer_pairs}
        if compute_probe_cosine
        else None
    )

    batches = make_dynamic_batches(
        pairs,
        tokenizer,
        enable_thinking=enable_thinking,
        max_batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
    )
    for batch_pairs in batches:
        input_ids, attention_mask, response_mask = build_batch_tensors(
            batch_pairs,
            tokenizer,
            last_k=last_k,
            enable_thinking=enable_thinking,
            device=device,
        )
        student_layers = forward_per_layer_hidden(student_model, input_ids, attention_mask)
        teacher_layers = forward_per_layer_hidden(teacher_model, input_ids, attention_mask)

        for spec in layer_pairs:
            h_s = extract_response_hidden_tensor(
                student_layers[spec.student_layer],
                response_mask,
                **pos_kw,
            )
            h_t = extract_response_hidden_tensor(
                teacher_layers[spec.teacher_layer],
                response_mask,
                **pos_kw,
            )
            if h_s.shape[0] == 0:
                continue
            z_s = ps_bank.forward_layer(spec.student_layer, spec.teacher_layer, h_s)
            z_t = pt_bank.forward_layer(spec.student_layer, spec.teacher_layer, h_t)
            subspace_cosines.append(compute_subspace_cosine(z_s, z_t))
            subspace_mses.append(float(nn.functional.mse_loss(z_s, z_t).item()))
            if probe_row_buffers is not None:
                s_buf, t_buf = probe_row_buffers[(spec.student_layer, spec.teacher_layer)]
                s_buf.append(h_s.cpu().numpy())
                t_buf.append(h_t.cpu().numpy())

    metrics = {
        "subspace_cosine_mean": float(np.mean(subspace_cosines)) if subspace_cosines else float("nan"),
        "subspace_mse_mean": float(np.mean(subspace_mses)) if subspace_mses else float("nan"),
    }
    if probe_row_buffers is not None:
        probe_cosines: list[float] = []
        for key in probe_row_buffers:
            s_chunks, t_chunks = probe_row_buffers[key]
            if not s_chunks:
                continue
            probe_cosines.append(
                compute_probe_cosine_from_rows(
                    np.concatenate(s_chunks, axis=0),
                    np.concatenate(t_chunks, axis=0),
                    rank=ps_bank.rank,
                )
            )
        metrics["linear_probe_cosine_mean"] = float(np.mean(probe_cosines)) if probe_cosines else float("nan")
    return metrics


def train_direct_one_rank(
    *,
    rank: int,
    student_model,
    teacher_model,
    tokenizer,
    train_pairs: list[dict[str, Any]],
    val_pairs: list[dict[str, Any]],
    layer_pairs: list[LayerPairSpec],
    student_dim: int,
    teacher_dim: int,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    rank_dir = build_rank_output_dir(
        output_dir, rank, projector_type="linear", subspace_mode="direct"
    )
    rank_dir.mkdir(parents=True, exist_ok=True)

    ps_bank = StudentProjectorBank(
        student_dim,
        rank,
        layer_pairs,
        projector_type="linear",
    ).to(device)
    pt_bank = StudentProjectorBank(
        teacher_dim,
        rank,
        layer_pairs,
        projector_type="linear",
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(ps_bank.parameters()) + list(pt_bank.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    pos_kw = response_position_kwargs_from_args(args)
    best_tracker = BestValTracker()

    before_metrics = evaluate_direct_projectors(
        student_model,
        teacher_model,
        tokenizer,
        val_pairs,
        layer_pairs,
        ps_bank,
        pt_bank,
        device=device,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        enable_thinking=args.enable_thinking,
        compute_probe_cosine=args.compute_probe_cosine,
        **pos_kw,
    )

    history: list[dict[str, float]] = []
    batches = make_dynamic_batches(
        train_pairs,
        tokenizer,
        enable_thinking=args.enable_thinking,
        max_batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
    )

    for epoch in range(1, args.epochs + 1):
        ps_bank.train()
        pt_bank.train()
        epoch_losses: list[float] = []

        for batch_pairs in tqdm(
            batches, desc=f"Train P_S+P_T direct rank={rank} epoch={epoch}", unit="batch"
        ):
            optimizer.zero_grad(set_to_none=True)
            input_ids, attention_mask, response_mask = build_batch_tensors(
                batch_pairs,
                tokenizer,
                last_k=args.last_k,
                enable_thinking=args.enable_thinking,
                device=device,
            )

            with torch.no_grad():
                student_layers = forward_per_layer_hidden(student_model, input_ids, attention_mask)
                teacher_layers = forward_per_layer_hidden(teacher_model, input_ids, attention_mask)

            batch_losses = []
            for spec in layer_pairs:
                h_s = extract_response_hidden_tensor(
                    student_layers[spec.student_layer],
                    response_mask,
                    **pos_kw,
                )
                h_t = extract_response_hidden_tensor(
                    teacher_layers[spec.teacher_layer],
                    response_mask,
                    **pos_kw,
                )
                if h_s.shape[0] == 0:
                    continue
                z_s = ps_bank.forward_layer(spec.student_layer, spec.teacher_layer, h_s)
                z_t = pt_bank.forward_layer(spec.student_layer, spec.teacher_layer, h_t)
                batch_losses.append(nn.functional.mse_loss(z_s, z_t))

            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        should_eval = (
            epoch == 1
            or epoch == args.epochs
            or epoch % max(args.eval_every, 1) == 0
        )
        if should_eval:
            val_metrics = evaluate_direct_projectors(
                student_model,
                teacher_model,
                tokenizer,
                val_pairs,
                layer_pairs,
                ps_bank,
                pt_bank,
                device=device,
                batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
                enable_thinking=args.enable_thinking,
                compute_probe_cosine=False,
                **pos_kw,
            )
            row = {
                "epoch": epoch,
                "train_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)
            if best_tracker.maybe_update(epoch, val_metrics, ps_bank, pt_bank=pt_bank):
                tqdm.write(
                    f"[direct rank={rank} epoch={epoch}] new best val_subspace_cosine="
                    f"{best_tracker.best_score:.4f}"
                )
            tqdm.write(
                f"[direct rank={rank} epoch={epoch}] train_loss={row['train_loss_mean']:.6f} "
                f"val_subspace_cosine={row['val_subspace_cosine_mean']:.4f}"
            )
        else:
            tqdm.write(
                f"[direct rank={rank} epoch={epoch}] train_loss="
                f"{float(np.mean(epoch_losses)) if epoch_losses else float('nan'):.6f} (skip val)"
            )

    final_metrics = evaluate_direct_projectors(
        student_model,
        teacher_model,
        tokenizer,
        val_pairs,
        layer_pairs,
        ps_bank,
        pt_bank,
        device=device,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        enable_thinking=args.enable_thinking,
        compute_probe_cosine=args.compute_probe_cosine,
        **pos_kw,
    )
    saved_from = "final"
    if best_tracker.restore_best(ps_bank, pt_bank=pt_bank):
        saved_from = "best_val"
        after_metrics = evaluate_direct_projectors(
            student_model,
            teacher_model,
            tokenizer,
            val_pairs,
            layer_pairs,
            ps_bank,
            pt_bank,
            device=device,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            enable_thinking=args.enable_thinking,
            compute_probe_cosine=args.compute_probe_cosine,
            **pos_kw,
        )
        best_epoch = best_tracker.best_epoch
    else:
        after_metrics = final_metrics
        best_epoch = args.epochs

    pt_weights = {
        f"s{s}_t{t}": pt_bank.projectors[f"s{s}_t{t}"].weight.detach().cpu()
        for s, t in {(spec.student_layer, spec.teacher_layer) for spec in layer_pairs}
    }
    ps_norms = [
        float(w.float().norm().item()) for w in ps_bank.state_dict().values() if w.ndim == 2
    ]
    pt_norms = [float(w.float().norm().item()) for w in pt_weights.values()]
    print(
        f"Saving direct_bank.pt from {saved_from} epoch={best_epoch}: P_S layers={len(ps_bank.projectors)}, "
        f"P_T layers={len(pt_weights)}, "
        f"P_S norm mean={float(np.mean(ps_norms)):.4f}, P_T norm mean={float(np.mean(pt_norms)):.4f}"
    )
    torch.save(
        {
            "subspace_mode": "direct",
            "rank": rank,
            "best_epoch": best_epoch,
            "saved_from": saved_from,
            "projector_type": "linear",
            "layer_pairs": [asdict(spec) for spec in layer_pairs],
            "state_dict": ps_bank.state_dict(),
            "pt_state_dict": pt_bank.state_dict(),
            "pt_weights": pt_weights,
        },
        rank_dir / "direct_bank.pt",
    )

    plot_training_curves(history, rank_dir / "training_curves.png", rank=rank)

    summary = {
        "rank": rank,
        "subspace_mode": "direct",
        "projector_type": "linear",
        "best_epoch": best_epoch,
        "saved_from": saved_from,
        "epochs": args.epochs,
        "layer_pairs": [asdict(spec) for spec in layer_pairs],
        "metrics_before": before_metrics,
        "metrics_final": final_metrics,
        "metrics_after": after_metrics,
        "history": history,
        "gate_pass_subspace_cosine_0p8": bool(after_metrics["subspace_cosine_mean"] > 0.8),
    }
    with open(rank_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def train_one_rank(
    *,
    rank: int,
    student_model,
    teacher_model,
    tokenizer,
    train_pairs: list[dict[str, Any]],
    val_pairs: list[dict[str, Any]],
    layer_pairs: list[LayerPairSpec],
    frozen_pts: dict[tuple[int, int], FrozenTeacherProjector],
    student_dim: int,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
    modality: str = "text",
    processor=None,
    teacher_device: torch.device | None = None,
    extra_pairs: list[tuple[Any, Any, torch.device, torch.device]] | None = None,
    train_hidden_cache: list[dict[tuple[int, int], tuple[Tensor, Tensor]]] | None = None,
    val_hidden_cache: list[dict[tuple[int, int], tuple[Tensor, Tensor]]] | None = None,
) -> dict[str, Any]:
    """extra_pairs (2026-07-27, explicit user requirement -- data-parallel training across
    more GPUs, not just the one (student,teacher) pair): each entry is
    (student_model_i, teacher_model_i, student_device_i, teacher_device_i) for an ADDITIONAL,
    independent replica of the frozen backbones on its own GPUs. ps_bank/optimizer/frozen_pts
    still live only on the primary pair's device (the plain student_model/teacher_model/device/
    teacher_device args above). Video-mode only; ignored for modality="text".

    train_hidden_cache/val_hidden_cache (2026-07-27, see build_hidden_cache_video): precomputed
    once in main() via all_pairs (so they already reflect any --extra-device-pairs parallelism)
    and reused for every epoch here instead of re-running the frozen student+teacher forward each
    time -- see build_hidden_cache_video's docstring for why this is correctness-safe. Video mode
    only; None for modality="text", which keeps its original per-epoch forward path unchanged."""
    rank_dir = build_rank_output_dir(
        output_dir, rank, projector_type=args.projector, subspace_mode="full"
    )
    rank_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoint-resume (2026-07-27, explicit user requirement: "GPUs may need to be handed
    # back to someone else at any time"). A 20-epoch run over the full dataset can take many
    # hours; without this, killing the process for any reason (reclaimed GPU, external-tenant
    # OOM, crash) loses every epoch trained so far, since the only prior output was the FINAL
    # best-checkpoint written after all epochs complete. resume_checkpoint.pt is now
    # (over)written after every epoch (atomically, via a temp-file + rename, so a kill mid-write
    # can never leave a corrupt checkpoint) and transparently resumed from on the next run with
    # identical arguments -- optimizer state included, so resuming isn't just "reload weights and
    # keep training with a cold Adam" but a true continuation.
    resume_ckpt_path = rank_dir / "resume_checkpoint.pt"

    for frozen_pt in frozen_pts.values():
        frozen_pt.weight = frozen_pt.weight.to(device)
        frozen_pt.mean = frozen_pt.mean.to(device)

    ps_bank = StudentProjectorBank(
        student_dim,
        rank,
        layer_pairs,
        projector_type=args.projector,
        mlp_hidden_mult=args.mlp_hidden_mult,
    ).to(device)
    optimizer = torch.optim.AdamW(ps_bank.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_kw = response_position_kwargs_from_args(args)
    best_tracker = BestValTracker()

    start_epoch = 1
    history: list[dict[str, float]] = []
    if resume_ckpt_path.exists():
        print(f"Found {resume_ckpt_path}, resuming training instead of starting fresh")
        ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        ps_bank.load_state_dict(ckpt["ps_bank_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_tracker.best_score = ckpt["best_score"]
        best_tracker.best_epoch = ckpt["best_epoch"]
        best_tracker.best_metrics = ckpt["best_metrics"]
        best_tracker.best_ps_state = ckpt["best_ps_state"]
        history = ckpt["history"]
        before_metrics = ckpt["before_metrics"]
        start_epoch = ckpt["epoch"] + 1
        print(
            f"Resuming rank={rank} at epoch {start_epoch}/{args.epochs}; "
            f"best so far: epoch={best_tracker.best_epoch} score={best_tracker.best_score:.4f}"
        )
    else:
        before_metrics = evaluate_projectors(
            student_model,
            teacher_model,
            tokenizer,
            val_pairs,
            layer_pairs,
            frozen_pts,
            ps_bank,
            device=device,
            teacher_device=teacher_device,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            max_vision_patches=args.max_vision_patches,
            enable_thinking=args.enable_thinking,
            compute_probe_cosine=args.compute_probe_cosine,
            modality=modality,
            processor=processor,
            hidden_cache=val_hidden_cache,
            **pos_kw,
        )

    if modality == "video":
        # No `batches` needed here: train_hidden_cache (built once in main() via
        # build_hidden_cache_video, using the exact same train_pairs/batch config) already
        # defines both the per-batch grouping and the forward-pass output for every epoch below.
        batches = None
    else:
        batches = make_dynamic_batches(
            train_pairs,
            tokenizer,
            enable_thinking=args.enable_thinking,
            max_batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        ps_bank.train()
        epoch_losses: list[float] = []

        if modality == "video":
            # Cache-driven: no forward pass here at all (see build_hidden_cache_video) -- just
            # the lightweight trainable projector + loss + backward + step per cached batch.
            for entry in tqdm(train_hidden_cache, desc=f"Train P_S rank={rank} epoch={epoch}", unit="batch"):
                optimizer.zero_grad(set_to_none=True)
                batch_losses = []
                for spec in layer_pairs:
                    key = (spec.student_layer, spec.teacher_layer)
                    h_s, h_t = entry[key]
                    if h_s.shape[0] == 0:
                        continue
                    h_s = h_s.to(device)
                    h_t = h_t.to(device)
                    z_t = project_teacher(frozen_pts[key], h_t).detach()
                    z_s = ps_bank.forward_layer(spec.student_layer, spec.teacher_layer, h_s)
                    batch_losses.append(nn.functional.mse_loss(z_s, z_t))

                if not batch_losses:
                    continue
                loss = torch.stack(batch_losses).mean()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
        else:
            for batch_pairs in tqdm(batches, desc=f"Train P_S rank={rank} epoch={epoch}", unit="batch"):
                optimizer.zero_grad(set_to_none=True)

                input_ids, attention_mask, response_mask = build_batch_tensors(
                    batch_pairs,
                    tokenizer,
                    last_k=args.last_k,
                    enable_thinking=args.enable_thinking,
                    device=device,
                )
                with torch.no_grad():
                    student_layers = forward_per_layer_hidden(student_model, input_ids, attention_mask)
                    teacher_layers = forward_per_layer_hidden(teacher_model, input_ids, attention_mask)

                batch_losses = []
                for spec in layer_pairs:
                    key = (spec.student_layer, spec.teacher_layer)
                    h_s = extract_response_hidden_tensor(
                        student_layers[spec.student_layer],
                        response_mask,
                        **pos_kw,
                    )
                    h_t = extract_response_hidden_tensor(
                        teacher_layers[spec.teacher_layer],
                        response_mask,
                        **pos_kw,
                    )
                    if h_s.shape[0] == 0:
                        continue
                    z_t = project_teacher(frozen_pts[key], h_t).detach()
                    z_s = ps_bank.forward_layer(spec.student_layer, spec.teacher_layer, h_s)
                    batch_losses.append(nn.functional.mse_loss(z_s, z_t))

                if not batch_losses:
                    continue
                loss = torch.stack(batch_losses).mean()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

        should_eval = (
            epoch == 1
            or epoch == args.epochs
            or epoch % max(args.eval_every, 1) == 0
        )
        if should_eval:
            val_metrics = evaluate_projectors(
                student_model,
                teacher_model,
                tokenizer,
                val_pairs,
                layer_pairs,
                frozen_pts,
                ps_bank,
                device=device,
                teacher_device=teacher_device,
                batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
                max_vision_patches=args.max_vision_patches,
                enable_thinking=args.enable_thinking,
                compute_probe_cosine=False,
                modality=modality,
                processor=processor,
                hidden_cache=val_hidden_cache,
                **pos_kw,
            )
            row = {
                "epoch": epoch,
                "train_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)
            if best_tracker.maybe_update(epoch, val_metrics, ps_bank):
                tqdm.write(
                    f"[rank={rank} epoch={epoch}] new best val_subspace_cosine="
                    f"{best_tracker.best_score:.4f}"
                )
            tqdm.write(
                f"[rank={rank} epoch={epoch}] train_loss={row['train_loss_mean']:.6f} "
                f"val_subspace_cosine={row['val_subspace_cosine_mean']:.4f}"
            )
        else:
            tqdm.write(
                f"[rank={rank} epoch={epoch}] train_loss={float(np.mean(epoch_losses)) if epoch_losses else float('nan'):.6f} "
                "(skip val)"
            )

        # Checkpoint every epoch (not just eval epochs) so a kill loses at most one epoch of
        # training. Atomic write: build the full state on disk under a temp name first, then
        # rename over the real path -- os.replace is atomic on POSIX, so a kill mid-write can
        # never leave resume_checkpoint.pt truncated/corrupt for the next run to trip over.
        tmp_ckpt_path = resume_ckpt_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "epoch": epoch,
                "ps_bank_state": ps_bank.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_score": best_tracker.best_score,
                "best_epoch": best_tracker.best_epoch,
                "best_metrics": best_tracker.best_metrics,
                "best_ps_state": best_tracker.best_ps_state,
                "history": history,
                "before_metrics": before_metrics,
            },
            tmp_ckpt_path,
        )
        os.replace(tmp_ckpt_path, resume_ckpt_path)

    final_metrics = evaluate_projectors(
        student_model,
        teacher_model,
        tokenizer,
        val_pairs,
        layer_pairs,
        frozen_pts,
        ps_bank,
        device=device,
        teacher_device=teacher_device,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_vision_patches=args.max_vision_patches,
        enable_thinking=args.enable_thinking,
        compute_probe_cosine=args.compute_probe_cosine,
        modality=modality,
        processor=processor,
        hidden_cache=val_hidden_cache,
        **pos_kw,
    )
    saved_from = "final"
    if best_tracker.restore_best(ps_bank):
        saved_from = "best_val"
        after_metrics = evaluate_projectors(
            student_model,
            teacher_model,
            tokenizer,
            val_pairs,
            layer_pairs,
            frozen_pts,
            ps_bank,
            device=device,
            teacher_device=teacher_device,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            max_vision_patches=args.max_vision_patches,
            enable_thinking=args.enable_thinking,
            compute_probe_cosine=args.compute_probe_cosine,
            modality=modality,
            processor=processor,
            hidden_cache=val_hidden_cache,
            **pos_kw,
        )
        best_epoch = best_tracker.best_epoch
    else:
        after_metrics = final_metrics
        best_epoch = args.epochs

    frozen_pt_weights = {
        f"s{s}_t{t}": frozen_pts[(s, t)].weight.cpu() for s, t in frozen_pts
    }
    frozen_pt_means = {
        f"s{s}_t{t}": frozen_pts[(s, t)].mean.cpu() for s, t in frozen_pts
    }
    if not frozen_pt_weights:
        raise RuntimeError("frozen_pt_weights is empty; refusing to save checkpoint without P_T")
    pt_norms = [float(w.float().norm().item()) for w in frozen_pt_weights.values()]
    print(
        f"Saving ps_bank.pt from {saved_from} epoch={best_epoch}: P_S layers={len(ps_bank.projectors)}, "
        f"P_T layers={len(frozen_pt_weights)}, P_T norm mean={float(np.mean(pt_norms)):.4f}"
    )
    torch.save(
        {
            "rank": rank,
            "best_epoch": best_epoch,
            "saved_from": saved_from,
            "projector_type": args.projector,
            "mlp_hidden_mult": args.mlp_hidden_mult if args.projector == "mlp" else None,
            "layer_pairs": [asdict(spec) for spec in layer_pairs],
            "state_dict": ps_bank.state_dict(),
            "frozen_pt_weights": frozen_pt_weights,
            "frozen_pt_means": frozen_pt_means,
        },
        rank_dir / "ps_bank.pt",
    )

    plot_training_curves(history, rank_dir / "training_curves.png", rank=rank)

    summary = {
        "rank": rank,
        "projector_type": args.projector,
        "mlp_hidden_mult": args.mlp_hidden_mult if args.projector == "mlp" else None,
        "best_epoch": best_epoch,
        "saved_from": saved_from,
        "epochs": args.epochs,
        "layer_pairs": [asdict(spec) for spec in layer_pairs],
        "metrics_before": before_metrics,
        "metrics_final": final_metrics,
        "metrics_after": after_metrics,
        "history": history,
        "gate_pass_subspace_cosine_0p8": bool(after_metrics["subspace_cosine_mean"] > 0.8),
    }
    with open(rank_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def train_residual_one_rank(
    *,
    tail_rank: int,
    head_rank: int,
    head_checkpoint: str,
    student_model,
    teacher_model,
    tokenizer,
    train_pairs: list[dict[str, Any]],
    val_pairs: list[dict[str, Any]],
    layer_pairs: list[LayerPairSpec],
    head_frozen_pts: dict[tuple[int, int], FrozenTeacherProjector],
    head_ps: dict[tuple[int, int], Tensor],
    frozen_tail_pts: dict[tuple[int, int], FrozenTeacherProjector],
    student_dim: int,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    rank_dir = build_rank_output_dir(
        output_dir,
        tail_rank,
        projector_type=args.projector,
        subspace_mode="residual",
        head_rank=head_rank,
    )
    rank_dir.mkdir(parents=True, exist_ok=True)

    for frozen_pt in head_frozen_pts.values():
        frozen_pt.weight = frozen_pt.weight.to(device)
        frozen_pt.mean = frozen_pt.mean.to(device)
    for frozen_pt in frozen_tail_pts.values():
        frozen_pt.weight = frozen_pt.weight.to(device)
        frozen_pt.mean = frozen_pt.mean.to(device)
    head_ps = {k: v.to(device) for k, v in head_ps.items()}

    ps_bank = StudentProjectorBank(
        student_dim,
        tail_rank,
        layer_pairs,
        projector_type=args.projector,
        mlp_hidden_mult=args.mlp_hidden_mult,
    ).to(device)
    optimizer = torch.optim.AdamW(ps_bank.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_kw = response_position_kwargs_from_args(args)

    def eval_residual_metrics(pairs: list[dict[str, Any]]) -> dict[str, float]:
        ps_bank.eval()
        head_cosines: list[float] = []
        tail_cosines: list[float] = []
        tail_mses: list[float] = []
        batches = make_dynamic_batches(
            pairs,
            tokenizer,
            enable_thinking=args.enable_thinking,
            max_batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
        )
        with torch.no_grad():
            for batch_pairs in batches:
                input_ids, attention_mask, response_mask = build_batch_tensors(
                    batch_pairs,
                    tokenizer,
                    last_k=args.last_k,
                    enable_thinking=args.enable_thinking,
                    device=device,
                )
                student_layers = forward_per_layer_hidden(student_model, input_ids, attention_mask)
                teacher_layers = forward_per_layer_hidden(teacher_model, input_ids, attention_mask)
                for spec in layer_pairs:
                    key = (spec.student_layer, spec.teacher_layer)
                    h_s = extract_response_hidden_tensor(
                        student_layers[spec.student_layer], response_mask, **pos_kw
                    )
                    h_t = extract_response_hidden_tensor(
                        teacher_layers[spec.teacher_layer], response_mask, **pos_kw
                    )
                    if h_s.shape[0] == 0:
                        continue
                    z_h_s = h_s @ head_ps[key].T
                    h_t_c = h_t - head_frozen_pts[key].mean.to(h_t.dtype)
                    z_h_t = h_t_c @ head_frozen_pts[key].weight.T.to(h_t.dtype)
                    head_cosines.append(compute_subspace_cosine(z_h_s, z_h_t))

                    h_s_res = subtract_rowspace_projection_torch(h_s, head_ps[key])
                    h_t_res = subtract_rowspace_projection_torch(
                        h_t, head_frozen_pts[key].weight, mean=head_frozen_pts[key].mean
                    )
                    z_t = project_teacher(frozen_tail_pts[key], h_t_res)
                    z_s = ps_bank.forward_layer(spec.student_layer, spec.teacher_layer, h_s_res)
                    tail_cosines.append(compute_subspace_cosine(z_s, z_t))
                    tail_mses.append(float(nn.functional.mse_loss(z_s, z_t).item()))
        return {
            "head_subspace_cosine_mean": float(np.mean(head_cosines)) if head_cosines else float("nan"),
            "subspace_cosine_mean": float(np.mean(tail_cosines)) if tail_cosines else float("nan"),
            "subspace_mse_mean": float(np.mean(tail_mses)) if tail_mses else float("nan"),
        }

    before_metrics = eval_residual_metrics(val_pairs)
    best_tracker = BestValTracker()
    history: list[dict[str, float]] = []
    batches = make_dynamic_batches(
        train_pairs,
        tokenizer,
        enable_thinking=args.enable_thinking,
        max_batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
    )

    for epoch in range(1, args.epochs + 1):
        ps_bank.train()
        epoch_losses: list[float] = []
        for batch_pairs in tqdm(
            batches, desc=f"Train P_S tail rank={tail_rank} epoch={epoch}", unit="batch"
        ):
            optimizer.zero_grad(set_to_none=True)
            input_ids, attention_mask, response_mask = build_batch_tensors(
                batch_pairs,
                tokenizer,
                last_k=args.last_k,
                enable_thinking=args.enable_thinking,
                device=device,
            )
            with torch.no_grad():
                student_layers = forward_per_layer_hidden(student_model, input_ids, attention_mask)
                teacher_layers = forward_per_layer_hidden(teacher_model, input_ids, attention_mask)

            batch_losses = []
            for spec in layer_pairs:
                key = (spec.student_layer, spec.teacher_layer)
                h_s = extract_response_hidden_tensor(
                    student_layers[spec.student_layer], response_mask, **pos_kw
                )
                h_t = extract_response_hidden_tensor(
                    teacher_layers[spec.teacher_layer], response_mask, **pos_kw
                )
                if h_s.shape[0] == 0:
                    continue
                h_s_res = subtract_rowspace_projection_torch(h_s, head_ps[key])
                h_t_res = subtract_rowspace_projection_torch(
                    h_t, head_frozen_pts[key].weight, mean=head_frozen_pts[key].mean
                )
                z_t = project_teacher(frozen_tail_pts[key], h_t_res).detach()
                z_s = ps_bank.forward_layer(spec.student_layer, spec.teacher_layer, h_s_res)
                batch_losses.append(nn.functional.mse_loss(z_s, z_t))

            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        should_eval = epoch == 1 or epoch == args.epochs or epoch % max(args.eval_every, 1) == 0
        if should_eval:
            val_metrics = eval_residual_metrics(val_pairs)
            row = {
                "epoch": epoch,
                "train_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)
            if best_tracker.maybe_update(epoch, val_metrics, ps_bank):
                tqdm.write(
                    f"[tail={tail_rank} epoch={epoch}] new best val_tail_cos="
                    f"{best_tracker.best_score:.4f}"
                )
            tqdm.write(
                f"[tail={tail_rank} epoch={epoch}] train_loss={row['train_loss_mean']:.6f} "
                f"val_head_cos={row['val_head_subspace_cosine_mean']:.4f} "
                f"val_tail_cos={row['val_subspace_cosine_mean']:.4f}"
            )

    final_metrics = eval_residual_metrics(val_pairs)
    saved_from = "final"
    if best_tracker.restore_best(ps_bank):
        saved_from = "best_val"
        after_metrics = eval_residual_metrics(val_pairs)
        best_epoch = best_tracker.best_epoch
    else:
        after_metrics = final_metrics
        best_epoch = args.epochs

    frozen_tail_weights = {f"s{s}_t{t}": frozen_tail_pts[(s, t)].weight.cpu() for s, t in frozen_tail_pts}
    frozen_tail_means = {f"s{s}_t{t}": frozen_tail_pts[(s, t)].mean.cpu() for s, t in frozen_tail_pts}
    torch.save(
        {
            "subspace_mode": "residual",
            "head_rank": head_rank,
            "tail_rank": tail_rank,
            "head_checkpoint": head_checkpoint,
            "best_epoch": best_epoch,
            "saved_from": saved_from,
            "projector_type": args.projector,
            "mlp_hidden_mult": args.mlp_hidden_mult if args.projector == "mlp" else None,
            "rank": tail_rank,
            "layer_pairs": [asdict(spec) for spec in layer_pairs],
            "state_dict": ps_bank.state_dict(),
            "frozen_tail_pt_weights": frozen_tail_weights,
            "frozen_tail_pt_means": frozen_tail_means,
            "frozen_pt_weights": frozen_tail_weights,
            "frozen_pt_means": frozen_tail_means,
        },
        rank_dir / "ps_tail_bank.pt",
    )
    with open(rank_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tail_rank": tail_rank,
                "head_rank": head_rank,
                "projector_type": args.projector,
                "mlp_hidden_mult": args.mlp_hidden_mult if args.projector == "mlp" else None,
                "best_epoch": best_epoch,
                "saved_from": saved_from,
                "metrics_before": before_metrics,
                "metrics_final": final_metrics,
                "metrics_after": after_metrics,
                "history": history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return {
        "tail_rank": tail_rank,
        "head_rank": head_rank,
        "best_epoch": best_epoch,
        "saved_from": saved_from,
        "metrics_after": after_metrics,
        "gate_pass_subspace_cosine_0p8": bool(after_metrics["subspace_cosine_mean"] > 0.8),
    }


def plot_training_curves(history: list[dict[str, float]], output_path: Path, *, rank: int) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss_mean"] for row in history]
    val_cos = [row["val_subspace_cosine_mean"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train subspace MSE loss")
    axes[0].set_title(f"Rank {rank}: train loss")

    axes[1].plot(epochs, val_cos, marker="o", color="tab:green")
    axes[1].axhline(0.8, color="gray", linestyle="--", linewidth=1, label="gate 0.8")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val subspace cosine")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].legend()
    axes[1].set_title(f"Rank {rank}: val subspace cosine")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def split_pairs(pairs: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_fraction <= 0 or len(pairs) < 2:
        return pairs, []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)
    val_size = max(1, int(round(len(pairs) * val_fraction)))
    val_idx = set(indices[:val_size].tolist())
    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val_pairs = [pairs[i] for i in range(len(pairs)) if i in val_idx]
    return train_pairs, val_pairs


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teacher_device = torch.device(args.teacher_device) if args.teacher_device else device
    if teacher_device != device:
        print(f"Student device: {device}, teacher device: {teacher_device} (2-GPU split)")

    pairs = load_all_pairs_from_jsonl(args.responses_jsonl)
    train_pairs, val_pairs = split_pairs(pairs, args.val_fraction, args.seed)
    print(f"Loaded {len(pairs)} pairs: train={len(train_pairs)}, val={len(val_pairs)}")

    processor = None
    if args.modality == "video":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from cross_arch_repr_analysis import load_processor

        processor = load_processor(args.student_model_path)
        tokenizer = processor.tokenizer
        # Required, not just convenient (see plan): a single shared processor must build both
        # models' input_ids identically for OPRD-Bridge's premise to hold.
        teacher_processor = load_processor(args.teacher_model_path)
        for key in ("video_token_id", "image_token_id"):
            assert getattr(processor, key, None) == getattr(teacher_processor, key, None), (
                f"student/teacher processor mismatch on {key}"
            )
        del teacher_processor
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.student_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading student model: {args.student_model_path}")
    student_model = load_causal_lm(args.student_model_path, dtype, device)
    student_model.eval()
    for param in student_model.parameters():
        param.requires_grad = False

    print(f"Loading teacher model: {args.teacher_model_path}")
    teacher_model = load_causal_lm(args.teacher_model_path, dtype, teacher_device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    extra_pairs: list[tuple[Any, Any, torch.device, torch.device]] = []
    if args.extra_device_pairs:
        for pair_str in args.extra_device_pairs.split(","):
            s_idx, t_idx = pair_str.strip().split("-")
            s_dev = torch.device(f"cuda:{s_idx}")
            t_dev = torch.device(f"cuda:{t_idx}")
            print(f"Loading extra data-parallel pair: student on {s_dev}, teacher on {t_dev}")
            s_model = load_causal_lm(args.student_model_path, dtype, s_dev)
            s_model.eval()
            for param in s_model.parameters():
                param.requires_grad = False
            t_model = load_causal_lm(args.teacher_model_path, dtype, t_dev)
            t_model.eval()
            for param in t_model.parameters():
                param.requires_grad = False
            extra_pairs.append((s_model, t_model, s_dev, t_dev))
        print(f"Total data-parallel pairs: {1 + len(extra_pairs)} (1 primary + {len(extra_pairs)} extra)")

    with torch.no_grad():
        probe_ids = torch.tensor([[tokenizer.eos_token_id or 0]], device=device)
        probe_attn = torch.ones_like(probe_ids)
        student_probe = forward_per_layer_hidden(student_model, probe_ids, probe_attn)
        teacher_probe_ids = probe_ids.to(teacher_device)
        teacher_probe_attn = probe_attn.to(teacher_device)
        teacher_probe = forward_per_layer_hidden(teacher_model, teacher_probe_ids, teacher_probe_attn)
    num_student_layers = len(student_probe)
    num_teacher_layers = len(teacher_probe)
    student_dim = student_probe[0].shape[-1]
    teacher_dim = teacher_probe[0].shape[-1]
    del probe_ids, probe_attn, student_probe, teacher_probe

    layer_pairs = select_layer_pairs(num_student_layers, num_teacher_layers, args.layer_mode)
    pos_kw = response_position_kwargs_from_args(args)
    position_k = args.first_k if args.position_mode == "first_k" else args.last_k
    if args.subspace_mode == "direct" and args.projector != "linear":
        raise ValueError(
            f"--subspace-mode=direct only supports --projector=linear, got {args.projector!r}"
        )
    print(
        f"Layers: student={num_student_layers}, teacher={num_teacher_layers}, "
        f"pairs={len(layer_pairs)} ({args.layer_mode}), subspace_mode={args.subspace_mode}, "
        f"projector={args.projector}, position_mode={args.position_mode}, k={position_k}"
    )

    all_summaries: list[dict[str, Any]] = []
    if args.subspace_mode == "direct":
        for rank in args.ranks:
            print(f"\n=== Training direct P_S+P_T rank={rank} (no teacher SVD) ===")
            summary = train_direct_one_rank(
                rank=rank,
                student_model=student_model,
                teacher_model=teacher_model,
                tokenizer=tokenizer,
                train_pairs=train_pairs,
                val_pairs=val_pairs if val_pairs else train_pairs,
                layer_pairs=layer_pairs,
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                device=device,
                args=args,
                output_dir=output_dir,
            )
            all_summaries.append(summary)
    elif args.subspace_mode == "residual":
        if not args.head_checkpoint:
            raise ValueError("--head-checkpoint is required when --subspace-mode=residual")
        head_frozen_pts, head_ps = load_head_projectors_from_checkpoint(
            args.head_checkpoint,
            layer_pairs,
            head_rank=args.head_rank,
        )
        print(f"Loaded head bridge from {args.head_checkpoint} (rank={args.head_rank})")
        residual_rows = accumulate_teacher_residual_rows_for_pca(
            teacher_model,
            tokenizer,
            train_pairs,
            layer_pairs,
            head_frozen_pts,
            device=device,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            enable_thinking=args.enable_thinking,
            max_pca_rows=args.max_pca_rows,
            seed=args.seed,
            **pos_kw,
        )
        tail_pca_bases = {
            key: fit_teacher_pca_basis(rows, device=device)
            for key, rows in tqdm(residual_rows.items(), desc="Tail PCA / P_T", unit="layer")
        }
        for tail_rank in args.ranks:
            print(f"\n=== Training residual tail rank={tail_rank} (head rank={args.head_rank}) ===")
            frozen_tail_pts = build_frozen_pts_for_rank(tail_pca_bases, tail_rank)
            summary = train_residual_one_rank(
                tail_rank=tail_rank,
                head_rank=args.head_rank,
                head_checkpoint=args.head_checkpoint,
                student_model=student_model,
                teacher_model=teacher_model,
                tokenizer=tokenizer,
                train_pairs=train_pairs,
                val_pairs=val_pairs if val_pairs else train_pairs,
                layer_pairs=layer_pairs,
                head_frozen_pts=head_frozen_pts,
                head_ps=head_ps,
                frozen_tail_pts=frozen_tail_pts,
                student_dim=student_dim,
                device=device,
                args=args,
                output_dir=output_dir,
            )
            all_summaries.append(summary)
    else:
        # 2026-07-27: precompute + cache hidden states ONCE here (video mode only) instead of
        # letting accumulate_teacher_rows_for_pca and every epoch of train_one_rank each re-run
        # the frozen student/teacher forward pass independently. Both models never change after
        # this point (frozen, no_grad), so a batch's (h_s, h_t) computed here is valid for PCA
        # fitting AND for all args.epochs training epochs AND for every eval-every validation
        # call -- see build_hidden_cache_video's docstring. Persisted to output_dir so a killed/
        # resumed run doesn't redo this ~1-pass cost either (output_dir, not rank_dir: the raw
        # hidden states are rank-independent, shared across all --ranks values).
        train_hidden_cache = None
        val_hidden_cache = None
        if args.modality == "video":
            cache_all_pairs = [(student_model, teacher_model, device, teacher_device)] + extra_pairs
            if len(cache_all_pairs) > 1:
                print(
                    f"Precomputing hidden states with {len(cache_all_pairs)} (student,teacher) "
                    f"pairs, primary device={device}"
                )
            train_video_batches = _video_batches(
                train_pairs, processor, args.batch_size, args.enable_thinking,
                max_batch_tokens=args.max_batch_tokens, max_vision_patches=args.max_vision_patches,
            )
            train_hidden_cache = load_or_build_hidden_cache_video(
                output_dir / "hidden_cache_train.pt",
                train_video_batches, cache_all_pairs, layer_pairs, processor,
                position_mode=args.position_mode, last_k=args.last_k, first_k=args.first_k,
                primary_device=device, desc="Precompute train hidden states",
            )
            eval_pairs = val_pairs if val_pairs else train_pairs
            val_video_batches = _video_batches(
                eval_pairs, processor, args.batch_size, args.enable_thinking,
                max_batch_tokens=args.max_batch_tokens, max_vision_patches=args.max_vision_patches,
            )
            val_hidden_cache = load_or_build_hidden_cache_video(
                output_dir / "hidden_cache_val.pt",
                val_video_batches, cache_all_pairs, layer_pairs, processor,
                position_mode=args.position_mode, last_k=args.last_k, first_k=args.first_k,
                primary_device=device, desc="Precompute val hidden states",
            )

        teacher_rows = accumulate_teacher_rows_for_pca(
            teacher_model,
            tokenizer,
            train_pairs,
            layer_pairs,
            device=teacher_device,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            max_vision_patches=args.max_vision_patches,
            enable_thinking=args.enable_thinking,
            max_pca_rows=args.max_pca_rows,
            seed=args.seed,
            modality=args.modality,
            processor=processor,
            hidden_cache=train_hidden_cache,
            **pos_kw,
        )

        print("Fitting teacher PCA bases (covariance eigh, once per layer)...")
        pca_bases = {
            key: fit_teacher_pca_basis(rows, device=device)
            for key, rows in tqdm(teacher_rows.items(), desc="PCA / P_T", unit="layer")
        }
        for rank in args.ranks:
            print(f"\n=== Training rank={rank} ===")
            frozen_pts = build_frozen_pts_for_rank(pca_bases, rank)
            summary = train_one_rank(
                rank=rank,
                student_model=student_model,
                teacher_model=teacher_model,
                tokenizer=tokenizer,
                train_pairs=train_pairs,
                val_pairs=val_pairs if val_pairs else train_pairs,
                layer_pairs=layer_pairs,
                frozen_pts=frozen_pts,
                student_dim=student_dim,
                device=device,
                teacher_device=teacher_device,
                args=args,
                output_dir=output_dir,
                modality=args.modality,
                processor=processor,
                extra_pairs=extra_pairs,
                train_hidden_cache=train_hidden_cache,
                val_hidden_cache=val_hidden_cache,
            )
            all_summaries.append(summary)

    final_summary = {
        "student_model": args.student_model_path,
        "teacher_model": args.teacher_model_path,
        "responses_jsonl": args.responses_jsonl,
        "num_pairs": len(pairs),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "student_dim": student_dim,
        "teacher_dim": teacher_dim,
        "layer_mode": args.layer_mode,
        "position_mode": args.position_mode,
        "last_k": args.last_k,
        "first_k": args.first_k,
        "projector_type": args.projector,
        "mlp_hidden_mult": args.mlp_hidden_mult if args.projector == "mlp" else None,
        "layer_pairs": [asdict(spec) for spec in layer_pairs],
        "ranks": args.ranks,
        "subspace_mode": args.subspace_mode,
        "head_rank": args.head_rank if args.subspace_mode == "residual" else None,
        "head_checkpoint": args.head_checkpoint,
        "rank_summaries": all_summaries,
        "gates": {
            (
                f"subspace_cosine_0p8_at_tail_{s['tail_rank']}"
                if "tail_rank" in s
                else f"subspace_cosine_0p8_at_rank_{s['rank']}"
            ): s["gate_pass_subspace_cosine_0p8"]
            for s in all_summaries
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(final_summary["gates"], indent=2))
    print(f"Saved pre-experiment 2 outputs to {output_dir}")


if __name__ == "__main__":
    main()
