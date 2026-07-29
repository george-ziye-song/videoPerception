"""Convert FINAL_TRAINING_DATASET_final.json (16-frame video MC QA, gap-recalibrated
2026-07-27) into the parquet schema verl's RLHFDataset expects for GRPO/OPD training.

Usage: python video_qa_mc.py
Reads:  /remote-home/ziyesong/videoPerception/FINAL_TRAINING_DATASET_final.json
Writes: /remote-home/ziyesong/videoPerception/OPRD/datasets/video_qa_mc_train.parquet
        /remote-home/ziyesong/videoPerception/OPRD/datasets/video_qa_mc_val.parquet
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

SRC = "/remote-home/ziyesong/videoPerception/FINAL_TRAINING_DATASET_final.json"
OUT_DIR = Path("/remote-home/ziyesong/videoPerception/OPRD/datasets")
VAL_PER_TASK = 30
SEED = 0

SYSTEM_PROMPT = "You are a helpful assistant."
INSTRUCTION = "Respond with only the letter of the correct option."

ANSWER_RE = re.compile(r"^([A-Za-z])\.\s")


def build_record(row: dict) -> dict:
    frames = row["frames"]
    image_tags = "<image>" * len(frames)
    question_block = "\n".join([row["question"], *row["options"], INSTRUCTION])
    user_content = image_tags + "\n" + question_block

    answer_letter = ANSWER_RE.match(row["answer"]).group(1).upper()

    return {
        "data_source": "video_qa_mc",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "images": [{"image": p} for p in frames],
        "reward_model": {"ground_truth": answer_letter, "style": "rule"},
        "extra_info": {
            "id": row["id"],
            "task": row["task"],
            "layer": row["layer"],
            "num_options": len(row["options"]),
        },
    }


def main():
    with open(SRC) as f:
        data = json.load(f)

    by_task = defaultdict(list)
    for row in data:
        by_task[row["task"]].append(row)

    rng = random.Random(SEED)
    train_rows, val_rows = [], []
    for task, rows in by_task.items():
        rows = rows[:]
        rng.shuffle(rows)
        n_val = min(VAL_PER_TASK, len(rows) // 10)  # never take more than 10% of a small task as val
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])
        print(f"{task}: {len(rows)} total -> {len(rows) - n_val} train / {n_val} val")

    train_records = [build_record(r) for r in train_rows]
    val_records = [build_record(r) for r in val_rows]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_records).to_parquet(OUT_DIR / "video_qa_mc_train.parquet")
    pd.DataFrame(val_records).to_parquet(OUT_DIR / "video_qa_mc_val.parquet")

    print(f"\ntotal: {len(train_records)} train / {len(val_records)} val")
    print(f"wrote {OUT_DIR / 'video_qa_mc_train.parquet'}")
    print(f"wrote {OUT_DIR / 'video_qa_mc_val.parquet'}")


if __name__ == "__main__":
    main()
