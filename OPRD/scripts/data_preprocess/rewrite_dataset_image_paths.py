#!/usr/bin/env python3
"""Rewrite the absolute frame-image path prefix baked into a video-QA parquet.

video_qa_mc_*.parquet's `images` column holds rows like
[{"image": "/remote-home/ziyesong/videoPerception/data/<source>/.../frame_01.jpg"}, ...] --
an absolute path on the machine the dataset was built on. After downloading the
parquet + frame-image folders from the HuggingFace dataset repo onto a different
machine (e.g. the A100 training server), run this once per parquet so the paths
point at wherever the images actually landed.

If you clone/download everything to the exact same absolute path
(/remote-home/ziyesong/videoPerception/data/...), this step is unnecessary.

Usage:
    python rewrite_dataset_image_paths.py IN.parquet OUT.parquet --new-root /abs/path/to/videoPerception/data
"""

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_parquet")
    ap.add_argument("output_parquet")
    ap.add_argument(
        "--old-root",
        default="/remote-home/ziyesong/videoPerception/data",
        help="Path prefix currently baked into the parquet (default: this server's original path).",
    )
    ap.add_argument("--new-root", required=True, help="Path prefix where the image folders actually live now.")
    args = ap.parse_args()

    df = pd.read_parquet(args.input_parquet)

    n_rewritten = 0

    def rewrite(images):
        nonlocal n_rewritten
        out = []
        for im in images:
            im = dict(im)
            if im["image"].startswith(args.old_root):
                im["image"] = args.new_root + im["image"][len(args.old_root) :]
                n_rewritten += 1
            out.append(im)
        return out

    df["images"] = df["images"].apply(rewrite)
    df.to_parquet(args.output_parquet)
    print(f"{args.input_parquet}: {len(df)} rows, {n_rewritten} image paths rewritten -> {args.output_parquet}")


if __name__ == "__main__":
    main()
