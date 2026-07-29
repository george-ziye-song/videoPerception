"""Reward function for multiple-choice video QA (bare option-letter answers,
enable_thinking=False, no CoT expected)."""

import re
import traceback

_SHORT_LETTER_RE = re.compile(r"^([A-Za-z])\b")
_ANY_LETTER_RE = re.compile(r"\b([A-Z])\b")
_SHORT_RESPONSE_MAX_LEN = 6


def extract_letter(solution_str: str) -> str | None:
    if not solution_str:
        return None
    text = solution_str.strip()

    if len(text) <= _SHORT_RESPONSE_MAX_LEN:
        m = _SHORT_LETTER_RE.match(text)
        return m.group(1).upper() if m else None

    # Longer/rambling response despite the instruction: "I" is almost always the
    # pronoun here, not the answer, so exclude it and take the LAST remaining
    # standalone capital letter (closer to "final answer" than the first word).
    matches = [c for c in _ANY_LETTER_RE.findall(text) if c != "I"]
    return matches[-1] if matches else None


def compute_score(solution_str: str, ground_truth: str) -> dict:
    pred = extract_letter(solution_str)
    gt = str(ground_truth).strip().upper()
    is_correct = pred is not None and pred == gt
    return {
        "score": 1.0 if is_correct else 0.0,
        "acc": is_correct,
        "pred": pred or "",
    }


def reward_func(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    try:
        return compute_score(solution_str, ground_truth)
    except Exception as e:
        print(f"[ERROR] Error in video_mc reward_func: {e}")
        traceback.print_exc()
        raise
