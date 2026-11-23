"""Compute the estimated answer counts (summed-area table) of every stream once and store them next to the stream.
Run: python export_est.py <project_dir> <dataset> <n_use>"""
import sys

from replay_core import *  # noqa: F401,F403

for s in ["valid", "mixed", "train_full", "train_half", "insert", "drift"] + [f"dist_{d}" for d in ["uniform", "data", "hotspot", "zipf", "trajectory"]]:
    if not (W / f"{s}.csv").exists():
        continue
    df, _ = load(s)
    np.save(W / f"est_{s}.npy", est_answers(df, s))
    print(DS, s, "ok", flush=True)
