"""Choose the parameters of OASIS (2026-11-13 revision) on the validation streams of all datasets.
Rule (fixed before looking at test streams, as for the earlier version): minimise the mean, over datasets, of
total time / best-fixed-per-type time. Writes results2/oasis2_config.json.
Run: python choose2.py <project_dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd

P = Path(sys.argv[1])
KEYS = ["tau", "r", "T0", "b", "N0", "cf"]
df = pd.concat([pd.read_csv(f) for f in (P / "results2").glob("*/tune2.csv")])
df["rel"] = df.total_us / df.bestfixed_us
g = df.groupby(KEYS).rel.mean().sort_values()
print(g.head(12).to_string())
best = dict(zip(KEYS, g.index[0]))
cfg = {"tau": float(best["tau"]), "r": int(best["r"]), "T0": float(best["T0"]), "b": float(best["b"]), "N0": float(best["N0"]),
       "cf": int(best["cf"])}
(P / "results2" / "oasis2_config.json").write_text(json.dumps(cfg))
print("chosen", cfg)
