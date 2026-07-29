"""Convert FINAL_TRAINING_DATASET_passrate_filtered.json (16-frame video MC-QA)
into a parquet whose `prompt` column is self-contained chat messages
(video block embedded directly in the user turn) — the format Stage 0/1
(`cross_arch_repr_analysis.py`/`cross_arch_preexp2_train_ps.py`) expect, since
they call `processor.apply_chat_template()` directly and never go through
verl's `RLHFDataset._build_messages()` placeholder-substitution path.

Usage: python prepare_video_mcq_parquet.py [--num-per-task N]
Writes: /remote-home/ziyesong/OPRD/datasets/video_mcq_oprd_bridge.parquet
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

SRC = "/remote-home/ziyesong/videoPerception/FINAL_TRAINING_DATASET_final.json"
OUT = Path("/remote-home/ziyesong/OPRD/datasets/video_mcq_oprd_bridge.parquet")
SYSTEM_PROMPT = "You are a helpful assistant."
INSTRUCTION = "Answer with only the option's letter (e.g. A), no explanation."


def build_record(row: dict) -> dict:
    question_text = "\n".join([row["question"], *row["options"], INSTRUCTION])
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": row["frames"]},
            {"type": "text", "text": question_text},
        ]},
    ]
    return {
        # Stored as a JSON string, not a nested struct: pyarrow unifies content-block
        # schemas across the system (text-only) and user (video+text) messages in the
        # same column, silently adding e.g. a `"video": None` key to the text block —
        # harmless-looking, but Qwen3.5's chat template rejects any `video` key
        # (even null) on a system message. A JSON string sidesteps pyarrow's struct
        # inference entirely; load_prompts() does a json.loads() before use.
        "prompt": json.dumps(messages),
        "id": row["id"],
        "task": row["task"],
        "layer": row["layer"],
        "source": row["source"],
        "answer": row["answer"],
        "options": row["options"],
        "num_frames": row["num_frames"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-per-task", type=int, default=None, help="cap rows per task (omit for all rows)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(SRC) as f:
        data = json.load(f)

    if args.num_per_task is not None:
        rng = random.Random(args.seed)
        by_task = {}
        for row in data:
            by_task.setdefault(row["task"], []).append(row)
        selected = []
        for task, rows in by_task.items():
            rows = rows[:]
            rng.shuffle(rows)
            selected.extend(rows[: args.num_per_task])
        data = selected

    records = [build_record(r) for r in data]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(OUT)
    print(f"wrote {len(records)} rows to {OUT}")


if __name__ == "__main__":
    main()
