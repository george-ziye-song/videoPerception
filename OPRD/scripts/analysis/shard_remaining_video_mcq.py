"""Second batch: shard whatever wasn't already sampled into the first batch
(datasets/shards/shard_*.parquet), so the two batches together cover the
full 8248-row dataset with no overlap.
"""

import argparse
from pathlib import Path

import pandas as pd

FULL = Path("/remote-home/ziyesong/OPRD/datasets/video_mcq_oprd_bridge.parquet")
FIRST_BATCH_DIR = Path("/remote-home/ziyesong/OPRD/datasets/shards")
OUT_DIR = Path("/remote-home/ziyesong/OPRD/datasets/shards_batch2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    full_df = pd.read_parquet(FULL)
    first_batch_ids = set()
    for shard_path in sorted(FIRST_BATCH_DIR.glob("shard_*.parquet")):
        first_batch_ids.update(pd.read_parquet(shard_path)["id"].tolist())
    print(f"first batch covered {len(first_batch_ids)} rows")

    remaining = full_df[~full_df["id"].isin(first_batch_ids)].reset_index(drop=True)
    print(f"remaining: {len(remaining)} / {len(full_df)} rows")
    print(remaining["task"].value_counts())

    remaining = remaining.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_ids = [i % args.num_shards for i in range(len(remaining))]
    remaining["_shard"] = shard_ids
    for shard_idx in range(args.num_shards):
        shard_df = remaining[remaining["_shard"] == shard_idx].drop(columns=["_shard"])
        out_path = OUT_DIR / f"shard_{shard_idx}.parquet"
        shard_df.to_parquet(out_path)
        print(f"shard {shard_idx}: {len(shard_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
