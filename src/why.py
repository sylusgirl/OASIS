"""Per-strategy mean times on the mixed test stream, per (type, size class). Run: python why.py <project_dir> <n_use>"""
import sys

import numpy as np

sys.argv = [sys.argv[0], sys.argv[1], "Uber", sys.argv[2]]
import replay_core as rc  # noqa: E402

for ds in ["Uber", "Dragon", "GeoNames", "Porto"]:
    rc.DS = ds; rc.W = rc.P / "work8" / f"{ds}_{rc.N}"
    df, T = rc.load("mixed")
    ty, size = rc.ctx_size(df, df.est.values)
    print("==", ds, "R-BF KD-BF DFS-MBR BFS-MBR DFS-Ball BFS-Ball Grid")
    for t in range(3):
        for s in np.unique(size[ty == t]):
            m = (ty == t) & (size == s)
            print(f"  {rc.TYPES[t]:6s} {s} n={m.sum():5d} " + " ".join(f"{v:8.1f}" for v in T[m].mean(0)))
    # range queries: fraction where Grid is fastest, by density
    m = ty == 2
    g = T[m].argmin(1) == 6
    print("  range: Grid fastest on %.0f%% of range queries" % (g.mean() * 100))
