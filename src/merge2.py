"""Merge the OASIS rows of results2/ (2026-11-13 revision) into a copy of results/ (all other selectors unchanged).
Writes results_merged/<dataset>/: the old CSV files with every OASIS row replaced, plus ablation2.csv, params2.csv,
info2.csv, tune2.csv and the summary of the new OASIS.
Run: python merge2.py <project_dir>
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

P = Path(sys.argv[1])
OLD, NEW, OUT = P / "results", P / "results2", P / "results_merged"
KEY = {"total_time.csv": "method", "selected.csv": "method", "distributions.csv": "method", "stream.csv": "method",
       "selectors.csv": "selector", "overhead.csv": "method"}
for d in ["Uber", "Dragon", "GeoNames", "Porto"]:
    o, n, m = OLD / d, NEW / d, OUT / d
    m.mkdir(parents=True, exist_ok=True)
    for f in o.iterdir():
        shutil.copy(f, m / f.name)
    for name, col in KEY.items():
        a, b = pd.read_csv(o / name), pd.read_csv(n / name)
        a = a[a[col] != "OASIS"]
        pd.concat([a, b[a.columns]], ignore_index=True).to_csv(m / name, index=False)
    a, b = pd.read_csv(o / "summary.csv"), pd.read_csv(n / "summary.csv")
    pd.concat([a[a.method != "OASIS"], b[a.columns]], ignore_index=True).to_csv(m / "summary.csv", index=False)
    a, b = pd.read_csv(o / "scalability.csv"), pd.read_csv(n / "scalability.csv")
    pd.concat([a[~a.series.str.startswith("OASIS")], b], ignore_index=True).to_csv(m / "scalability.csv", index=False)
    a, b = pd.read_csv(o / "memory.csv"), pd.read_csv(n / "memory.csv")
    a["oasis_state_kb"] = float(b.oasis_state_kb.iloc[0]); a.to_csv(m / "memory.csv", index=False)
    c = json.load(open(o / "cost_us.json")); c["OASIS"] = json.load(open(n / "cost_us.json"))["OASIS"]
    json.dump(c, open(m / "cost_us.json", "w"), indent=1)
    for name in ["ablation2.csv", "params2.csv", "info2.csv", "tune2.csv", "cost2.json"]:
        if (n / name).exists():
            shutil.copy(n / name, m / name)
    (m / "summary_new_oasis.csv").write_bytes((n / "summary.csv").read_bytes())
print("merged into", OUT)
