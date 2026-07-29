#!/usr/bin/env python3
"""Probe实验Phase1:生成SynRL合成数据(pilot规模,samples_per_type=50)。
纯CPU,不需要GPU。只跑probe-experiment.md §3表格里列出的task type,
不是生成器自带GENERATORS字典里的全部key。

用法: python3 gen_pilot_data.py [samples_per_type]
"""
import sys
import importlib.util
import os

REPO_DIR = "/remote-home/ziyesong/videoPerception/repos/Synthetic-Video"
SAMPLES_PER_TYPE = int(sys.argv[1]) if len(sys.argv) > 1 else 50

# probe-experiment.md §3: 只要这几个task type,不是生成器全部的GENERATORS keys
WANTED_01 = [
    "Complex_Direction_Identification",
    "Rotation_Direction",
    "Rotation_Count",
    "Bouncing_Counting",
    "Directional_Event_Counting",
    "Acceleration_Identification",
]
WANTED_02 = ["Event_Sequence"]


def load_module(filename, modname):
    path = os.path.join(REPO_DIR, filename)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def run_generator(filename, modname, wanted_keys):
    print(f"\n=== {filename}: 生成 {wanted_keys} (each {SAMPLES_PER_TYPE} samples) ===")
    mod = load_module(filename, modname)
    full = mod.GENERATORS
    missing = [k for k in wanted_keys if k not in full]
    if missing:
        raise KeyError(f"{filename} 的 GENERATORS 里没有这些key: {missing}(现有: {list(full.keys())})")
    mod.GENERATORS = {k: full[k] for k in wanted_keys}
    mod.generate_dataset(samples_per_type=SAMPLES_PER_TYPE, max_workers=64)


if __name__ == "__main__":
    run_generator("01_atomic_motion.py", "atomic_motion_pilot", WANTED_01)
    run_generator("02_atomic2_extended_motion.py", "atomic2_extended_motion_pilot", WANTED_02)
    print("\n全部生成完成。")
