"""Figure 1 of the OASIS paper (MDPI width): (a) the best search plan differs by query kind, (b) offline selectors show a
long tail under query drift while OASIS stays close to the per-query optimum. Real measurements (GeoNames, 1M points).
Run: python plot_intro.py <results_dir> <out_dir>
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.0, "axes.titlesize": 8.4, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.4, "axes.linewidth": 0.6,
    "hatch.linewidth": 0.35, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2,
    "ytick.major.size": 2.2, "axes.titlepad": 3.0, "axes.labelpad": 1.6, "pdf.fonttype": 42, "ps.fonttype": 42})
EDGE = "#222222"

# (a) mean search time (us) of four plans on the mixed test stream of GeoNames (exp_code/why.py output, 2026-11-12)
groups = ["kNN, $k$=1", "kNN, $k$=1000"]
plans = ["KD-DFS-MBR", "KD-BF", "KD-BFS-MBR", "Grid"]
PLAB = {"KD-DFS-MBR": "A: KD-tree, depth-first", "KD-BF": "B: KD-tree, best-first", "KD-BFS-MBR": "C: KD-tree, breadth-first", "Grid": "D: uniform grid"}
times = {"KD-DFS-MBR": [0.4, 84.3], "KD-BF": [0.6, 40.4], "KD-BFS-MBR": [1.0, 6234.4],
         "Grid": [89.4, 486.9]}
fill = {"KD-DFS-MBR": "#BCD2EF", "KD-BF": "#B5E1E2", "KD-BFS-MBR": "#F6E7A1", "Grid": "#DFC9B8"}
hatch = {"KD-DFS-MBR": "\\\\\\\\", "KD-BF": "....", "KD-BFS-MBR": "||||", "Grid": "ooo"}

# (b) normalized query time per window of the drift stream on GeoNames
data = defaultdict(dict)
for r in csv.DictReader(open(RES / "GeoNames" / "stream.csv")):
    if r["workload"] == "Drift":
        data[r["method"]][float(r["progress_pct"])] = float(r["total_us"])
orc = data["Oracle"]
lines = [("AI+R-tree", "AI+R", "#24959A", "h", 0.9), ("UnIS", "UnIS", "#3B6FB6", "D", 0.9), ("OASIS", "OASIS", "#D62728", "*", 1.4)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.46, 2.05), gridspec_kw={"width_ratios": [1.25, 1.0]})
n = len(plans); wd = 0.8 / n
for j, p in enumerate(plans):
    xs = [k + (j - (n - 1) / 2) * wd for k in range(len(groups))]
    ax1.bar(xs, times[p], width=wd * 0.94, color=fill[p], edgecolor=EDGE, hatch=hatch[p], lw=0.4, zorder=3)
for k in range(len(groups)):
    best = min(plans, key=lambda p: times[p][k])
    j = plans.index(best); x = k + (j - (n - 1) / 2) * wd
    ax1.plot(x, times[best][k] * 2.1, marker="*", ms=7, color="#D62728", mec="#D62728", zorder=5, clip_on=False)
ax1.set_yscale("log")
ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=7)); ax1.yaxis.set_minor_formatter(NullFormatter())
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
ax1.set_ylim(0.2, 3e6); ax1.set_yticks([1, 10, 100, 1000, 10000]); ax1.minorticks_off()
ax1.set_xticks(range(len(groups))); ax1.set_xticklabels(groups); ax1.tick_params(axis="x", length=0)
ax1.set_ylabel(r"Search time ($\mu$s)")
ax1.set_title("(a) The fastest plan differs by query")
ax1.yaxis.grid(True, color="#C8C8C8", linestyle=(0, (1, 1.6)), linewidth=0.5, zorder=0); ax1.set_axisbelow(True)
ax1.legend(handles=[Patch(facecolor=fill[p], edgecolor=EDGE, hatch=hatch[p], lw=0.4, label=PLAB[p]) for p in plans]
           + [Line2D([], [], marker="*", ls="none", ms=7, color="#D62728", label="fastest plan")],
           loc="upper left", ncol=2, frameon=False, handlelength=1.3, columnspacing=0.5, handletextpad=0.4,
           borderaxespad=0.2, fontsize=6.6)

for x in (25, 50, 75):
    ax2.axvline(x, color="#BBBBBB", lw=0.6, zorder=1)
for lab, key, col, mk, lw in lines:
    pts = sorted(data[key].items())
    ax2.plot([p for p, _ in pts], [t / orc[p] for p, t in pts], color=col, lw=lw, marker=mk, markevery=6,
             ms=4.2 if key == "OASIS" else 2.8, mfc="white", mec=col, mew=0.6, zorder=5 if key == "OASIS" else 3, label=lab)
ax2.set_yscale("log")
ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=4)); ax2.yaxis.set_minor_formatter(NullFormatter())
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax2.set_ylim(0.8, 60)
ax2.set_xlim(0, 100); ax2.set_xticks([0, 25, 50, 75, 100])
ax2.set_xlabel("Query stream (%)"); ax2.set_ylabel("Time / fastest")
ax2.set_title("(b) Selectors under query drift")
ax2.yaxis.grid(True, color="#C8C8C8", linestyle=(0, (1, 1.6)), linewidth=0.5, zorder=0); ax2.set_axisbelow(True)
ax2.legend(loc="upper left", ncol=3, frameon=False, handlelength=1.5, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.2, fontsize=6.8)
fig.tight_layout(pad=0.15, w_pad=1.0)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "intro_problem.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "intro_problem.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
print("wrote intro_problem")
