"""Choose the OASIS parameters on the validation streams of all datasets.
Rule (fixed before looking at test streams): minimise the mean, over datasets, of total time / best-fixed-per-type time.
Run: python choose.py <project_dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd

P = Path(sys.argv[1])
df = pd.concat([pd.read_csv(f) for f in (P / "results").glob("*/tune.csv")])
df["rel"] = df.total_us / df.bestfixed_us
g = df.groupby(["alpha", "bound", "pace", "window"]).rel.mean().sort_values()
print(g.head(12).to_string())
alpha, bound, pace, window = g.index[0]
cfg = {"rule": "ctx", "alpha": float(alpha), "bound": float(bound), "pace": int(pace), "window": int(window), "revive": 1000}
(P / "results" / "oasis_config.json").write_text(json.dumps(cfg))
print("chosen", cfg)
