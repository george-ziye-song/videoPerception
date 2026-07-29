"""One-off: stratified-sample a target fraction of video_mcq_oprd_bridge.parquet,
then split into N disjoint, task-balanced shards for data-parallel Stage 0 runs
(one shard per GPU). Each shard is a self-contained parquet Stage 0 can read directly.
"""

import argparse
from pathlib import Path

import pandas as pd

SRC = Path("/remote-home/ziyesong/OPRD/datasets/video_mcq_oprd_bridge.parquet")
OUT_DIR = Path("/remote-home/ziyesong/OPRD/datasets/shards")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.3, help="Fraction of each task to sample")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_parquet(SRC)
    sampled_parts = []
    for task, group in df.groupby("task"):
        n = max(1, int(round(len(group) * args.fraction)))
        sampled_parts.append(group.sample(n=n, random_state=args.seed))
    sampled = pd.concat(sampled_parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"sampled {len(sampled)} / {len(df)} rows ({args.fraction*100:.0f}% per task)")
    print(sampled["task"].value_counts())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_ids = [i % args.num_shards for i in range(len(sampled))]
    sampled["_shard"] = shard_ids
    for shard_idx in range(args.num_shards):
        shard_df = sampled[sampled["_shard"] == shard_idx].drop(columns=["_shard"])
        out_path = OUT_DIR / f"shard_{shard_idx}.parquet"
        shard_df.to_parquet(out_path)
        print(f"shard {shard_idx}: {len(shard_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
