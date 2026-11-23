"""Section 5 figures of the OASIS paper: DejaVu Sans, one dataset per panel, OASIS last and red, plain axis
labels that only name the quantity.
Run: python plot_paper.py <results_dir> <out_dir>
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

import os
MDPI = os.environ.get("FIGMODE", "col") == "mdpi"   # mdpi: Applied Sciences text width, one row of four panels
W = 5.46 if MDPI else 3.33                 # text width of the MDPI template / \columnwidth of acmart sigconf, inches
FT, FL, FK, FG = (8.2, 8.0, 7.2, 7.4) if MDPI else (6.8, 6.6, 6.0, 6.0)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FL, "axes.titlesize": FT, "axes.labelsize": FL,
    "xtick.labelsize": FK, "ytick.labelsize": FK, "legend.fontsize": FG,
    "axes.linewidth": 0.6, "hatch.linewidth": 0.35,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.4, "ytick.major.pad": 1.2, "axes.titlepad": 2.4, "axes.labelpad": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white"})

DATASETS = ["Uber", "GeoNames", "Porto", "Dragon"]
QT = ["kNN", "Radius", "Range"]
METHODS = ["Random", "Frozen", "AI+R", "UnIS", "UnIS-R", "LQP", "OASIS"]
LAB = {"AI+R": "AI+R-tree", "LQP": "LQP"}
LINE = {"Random": "#8C5A3C", "Frozen": "#C9A000", "AI+R": "#24959A", "UnIS": "#3B6FB6", "UnIS-R": "#7B5EA7",
        "LQP": "#4E9A4F", "OASIS": "#D62728"}
FILL = {"Random": "#DFC9B8", "Frozen": "#F6E7A1", "AI+R": "#B5E1E2", "UnIS": "#BCD2EF", "UnIS-R": "#D9CCEB",
        "LQP": "#C6E5C3", "OASIS": "#F08A8A"}
HATCH = {"Random": "ooo", "Frozen": "||||", "AI+R": "....", "UnIS": "\\\\\\\\", "UnIS-R": "////", "LQP": "----",
         "OASIS": "xxxx"}
MARK = {"Random": "o", "Frozen": "v", "AI+R": "h", "UnIS": "D", "UnIS-R": "P", "LQP": "^", "OASIS": "*"}
EDGE = "#222222"
REF = {"BestFixed": dict(color="#555555", ls=(0, (3, 1.5)), lw=0.8), "Oracle": dict(color="#111111", ls=(0, (1, 1.1)), lw=0.9)}
REFLAB = {"BestFixed": "Best fixed per type", "Oracle": "Oracle"}
STRATS = ["R-BF", "KD-BF", "KD-DFS-MBR", "KD-BFS-MBR", "KD-DFS-Ball", "KD-BFS-Ball", "Grid"]
SFILL = ["#D9D9D9", "#8F8F8F", "#BCD2EF", "#3B6FB6", "#F9CFAE", "#E07A2B", "#C6E5C3"]


def quad(h_col, h_mdpi, **kw):
    """Four dataset panels: 2x2 at column width, 1x4 at MDPI text width."""
    if MDPI:
        fig, axes = plt.subplots(1, 4, figsize=(W, h_mdpi), **kw)
        return fig, list(axes.flat)
    fig, axes = plt.subplots(2, 2, figsize=(W, h_col), **kw)
    return fig, list(axes.flat)


def load(name):
    rows = []
    for d in DATASETS:
        f = RES / d / name
        if f.exists():
            rows += list(csv.DictReader(open(f)))
    return rows


def grid(ax):
    ax.yaxis.grid(True, which="major", color="#C8C8C8", linestyle=(0, (1, 1.6)), linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def logy(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))


def patch(fill, hatch, label):
    return Patch(facecolor=fill, edgecolor=EDGE, hatch=hatch, lw=0.35, label=label)


def legend(fig, handles, ncol, hl=1.6):
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False, handlelength=hl,
               columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.3)


def save(fig, name, top_inches):
    H = fig.get_size_inches()[1]
    fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.35, rect=(0, 0, 1, 1 - top_inches / H))
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", name)


def method_handles(refs=True):
    hs = [patch(FILL[m], HATCH[m], LAB.get(m, m)) for m in METHODS]
    if refs:
        hs += [Line2D([], [], label=REFLAB[k], **REF[k]) for k in REF]
    return hs


def bars(ax, groups, value, refs=()):
    n = len(METHODS); wd = 0.84 / n
    for j, m in enumerate(METHODS):
        xs = [k + (j - (n - 1) / 2) * wd for k in range(len(groups))]
        ax.bar(xs, [value(g, m) for g in groups], width=wd, color=FILL[m], edgecolor=EDGE, hatch=HATCH[m], lw=0.35,
               zorder=4 if m == "OASIS" else 3)
    for r in refs:
        for k, g in enumerate(groups):
            ax.plot([k - 0.44, k + 0.44], [value(g, r)] * 2, zorder=6, **REF[r])


# ---------------------------------------------------------------- 5.2 overall query time (mixed stream)
def fig_total():
    v = {(r["dataset"], r["qtype"], r["method"]): float(r["total_us"]) for r in load("total_time.csv")}
    fig, axes = quad(2.45, 2.15)
    for c, d in enumerate(DATASETS):
        ax = axes[c]; grid(ax)
        bars(ax, QT, lambda q, m: v[(d, q, m)], refs=("BestFixed", "Oracle"))
        logy(ax); ax.set_title(d)
        ax.set_xticks(range(3)); ax.set_xticklabels(["kNN", "Radius", "Range"], fontsize=6.4 if MDPI else FK); ax.tick_params(axis="x", length=0)
        if (c == 0) if MDPI else (c % 2 == 0):
            ax.set_ylabel(r"Query time ($\mu$s)")
    legend(fig, method_handles(), 5)
    save(fig, "exp_total", 0.30)


# ---------------------------------------------------------------- 5.2 selected strategies
def fig_selected():
    v = {(r["dataset"], r["qtype"], r["method"], r["strategy"]): float(r["share_pct"]) for r in load("selected.csv")}
    fig, axes = quad(2.45, 2.15, sharey=True)
    for c, d in enumerate(DATASETS):
        ax = axes[c]; grid(ax)
        for k, q in enumerate(QT):
            for j, m in enumerate(["Oracle", "OASIS"]):
                x = k + (j - 0.5) * 0.42; b = 0
                for s, col in zip(STRATS, SFILL):
                    h = v[(d, q, m, s)]
                    ax.bar(x, h, bottom=b, width=0.38, color=col, edgecolor=EDGE, lw=0.3, zorder=3); b += h
                ax.text(x, 102, "Or" if m == "Oracle" else "OA", ha="center", va="bottom", fontsize=4.6,
                        color=LINE["OASIS"] if m == "OASIS" else "black")
        ax.set_ylim(0, 114); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_title(d)
        ax.set_xticks(range(3)); ax.set_xticklabels(["kNN", "Radius", "Range"], fontsize=6.4 if MDPI else FK); ax.tick_params(axis="x", length=0)
        if (c == 0) if MDPI else (c % 2 == 0):
            ax.set_ylabel("Share (%)")
    legend(fig, [patch(col, "", s) for s, col in zip(STRATS, SFILL)], 4)
    save(fig, "exp_selected", 0.30)


# ---------------------------------------------------------------- 5.2 selector models
def fig_selectors():
    rows = load("selectors.csv")
    v = {(r["dataset"], r["workload"], r["selector"]): float(r["total_us"]) for r in rows}
    fam = {r["selector"]: r["family"] for r in rows}
    sels = list(dict.fromkeys(r["selector"] for r in rows))
    off = [s for s in sels if fam[s] == "offline"]; on = [s for s in sels if fam[s] == "online" and s != "OASIS"]
    order = off + on + ["OASIS"]
    offc = ["#EFEFEF", "#D9D9D9", "#BDBDBD", "#A3A3A3", "#878787"]
    onc = ["#DCE8F6", "#BCD2EF", "#8DB3E2", "#5E8FCB", "#3B6FB6"]
    color = {s: offc[i] for i, s in enumerate(off)}; color.update({s: onc[i] for i, s in enumerate(on)}); color["OASIS"] = FILL["OASIS"]
    hatch = {s: "" for s in off}; hatch.update({s: "////" for s in on}); hatch["OASIS"] = HATCH["OASIS"]
    lab = {"eps-greedy": r"$\epsilon$-greedy", "kNN": "kNN-C"}
    fig, axes = quad(2.55, 2.3)
    n = len(order); wd = 0.86 / n
    for c, d in enumerate(DATASETS):
        ax = axes[c]; grid(ax)
        for j, s in enumerate(order):
            xs = [k + (j - (n - 1) / 2) * wd for k in range(2)]
            ax.bar(xs, [v[(d, wl, s)] for wl in ["Static", "Changing"]], width=wd, color=color[s], edgecolor=EDGE,
                   hatch=hatch[s], lw=0.3, zorder=4 if s == "OASIS" else 3)
        logy(ax); ax.set_title(d)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Static", "Changing"]); ax.tick_params(axis="x", length=0)
        if (c == 0) if MDPI else (c % 2 == 0):
            ax.set_ylabel(r"Query time ($\mu$s)")
    legend(fig, [patch(color[s], hatch[s], lab.get(s, s)) for s in order], 6)
    save(fig, "exp_selectors", 0.30)


# ---------------------------------------------------------------- 5.2 query distributions
def fig_distributions():
    dists = ["Uniform", "Data", "Hotspot", "Zipf", "Trajectory"]
    v = {(r["dataset"], r["distribution"], r["method"]): float(r["total_us"]) for r in load("distributions.csv")}
    fig, axes = plt.subplots(2, 2, figsize=(W, 3.5 if MDPI else 2.45))
    for i, d in enumerate(DATASETS):
        ax = axes.flat[i]; grid(ax)
        bars(ax, dists, lambda g, m: v[(d, g, m)], refs=("BestFixed", "Oracle"))
        logy(ax); ax.set_title(d)
        ax.set_xticks(range(5)); ax.set_xticklabels(["Unif.", "Data", "Hotsp.", "Zipf", "Traj."]); ax.tick_params(axis="x", length=0)
        if i % 2 == 0:
            ax.set_ylabel(r"Query time ($\mu$s)")
    legend(fig, method_handles(), 5)
    save(fig, "exp_distributions", 0.30)


# ---------------------------------------------------------------- 5.2 insertions and query drift (time / Oracle)
def fig_stream():
    data = defaultdict(dict)
    for r in load("stream.csv"):
        data[(r["dataset"], r["workload"], r["method"])][float(r["progress_pct"])] = float(r["total_us"])
    show = [m for m in METHODS if m != "Random"]
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.1 if MDPI else 2.3), sharex=True)
    for r_, wl in enumerate(["Insertion", "Drift"]):
        for c, d in enumerate(DATASETS):
            ax = axes[r_][c]; grid(ax)
            if wl == "Drift":
                for x in (25, 50, 75):
                    ax.axvline(x, color="#BBBBBB", lw=0.5, zorder=1)
            orc = data[(d, wl, "Oracle")]
            for m in show:
                pts = sorted(data[(d, wl, m)].items())
                ax.plot([p for p, _ in pts], [t / orc[p] for p, t in pts], color=LINE[m], marker=MARK[m], markevery=6,
                        ms=3.6 if m == "OASIS" else 2.4, mfc="white", mec=LINE[m], mew=0.6,
                        lw=1.2 if m == "OASIS" else 0.75, zorder=5 if m == "OASIS" else 3)
            logy(ax)
            ax.set_title(d)
            ax.set_xticks([0, 50, 100])
            if c == 0:
                ax.set_ylabel(f"{wl}\nNormalized time", linespacing=1.1)
            if r_ == 1:
                ax.set_xlabel("Stream (%)")
    legend(fig, [Line2D([], [], color=LINE[m], marker=MARK[m], mfc="white", mec=LINE[m], mew=0.6,
                        ms=3.6 if m == "OASIS" else 2.6, lw=1.2 if m == "OASIS" else 0.8, label=LAB.get(m, m)) for m in show], 6, hl=2.2)
    save(fig, "exp_stream", 0.18)


# ---------------------------------------------------------------- 5.3 selection overhead
def fig_overhead():
    v = {(r["dataset"], r["qtype"], r["method"], r["component"]): float(r["us"]) for r in load("overhead.csv")}
    comps = [("Features", "#E3E3E3", ""), ("Prediction", FILL["UnIS"], HATCH["UnIS"]), ("Selection", FILL["OASIS"], HATCH["OASIS"]),
             ("Search", "white", "")]
    labs = {"Features": "Features (UnIS)", "Prediction": "Prediction (UnIS)", "Selection": "Selection (OASIS)", "Search": "Search"}
    fig, axes = quad(2.35, 2.0)
    for c, d in enumerate(DATASETS):
        ax = axes[c]; grid(ax); top = 0
        for k, q in enumerate(QT):
            for j, m in enumerate(["UnIS", "OASIS"]):
                x = k + (j - 0.5) * 0.42; b = 0
                for comp, col, h in comps:
                    val = v.get((d, q, m, comp))
                    if val is None:
                        continue
                    ax.bar(x, val, bottom=b, width=0.38, color=col, edgecolor=EDGE, hatch=h, lw=0.3, zorder=3); b += val
                top = max(top, b)
                ax.text(x, b * 1.02, "U" if m == "UnIS" else "O", ha="center", va="bottom", fontsize=4.6,
                        color=LINE["OASIS"] if m == "OASIS" else "black")
        ax.set_ylim(0, top * 1.15); ax.set_title(d)
        ax.set_xticks(range(3)); ax.set_xticklabels(["kNN", "Radius", "Range"], fontsize=6.4 if MDPI else FK); ax.tick_params(axis="x", length=0)
        if (c == 0) if MDPI else (c % 2 == 0):
            ax.set_ylabel(r"Query time ($\mu$s)")
    legend(fig, [patch(col, h, labs[comp]) for comp, col, h in comps], 4)
    save(fig, "exp_overhead", 0.18)


# ---------------------------------------------------------------- 5.3 scalability
def fig_scalability():
    data = defaultdict(list)
    for r in load("scalability.csv"):
        data[(r["dataset"], r["series"])].append((float(r["n_million"]), float(r["us"])))
    st = {"OASIS search": dict(color=LINE["OASIS"], marker="*", ls="-", lw=1.2, ms=3.6),
          "OASIS overhead": dict(color=LINE["OASIS"], marker="*", ls=(0, (3, 1.5)), lw=1.0, ms=3.6),
          "UnIS overhead": dict(color=LINE["UnIS"], marker="D", ls=(0, (3, 1.5)), lw=0.8, ms=2.4)}
    lab = {"OASIS search": "Search (OASIS)", "OASIS overhead": "Selection (OASIS)", "UnIS overhead": "Features + prediction (UnIS)"}
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.95 if MDPI else 1.45), sharey=True)
    for c, d in enumerate(DATASETS):
        ax = axes[c]; grid(ax)
        for s, k in st.items():
            pts = sorted(data[(d, s)])
            ax.plot(range(len(pts)), [p[1] for p in pts], mfc="white", mec=k["color"], mew=0.6, **k)
        ax.set_xticks(range(len(pts))); ax.set_xticklabels([f"{p[0]:g}" for p in pts])
        logy(ax); ax.set_title(d); ax.set_xlabel("Points (M)")
        if c == 0:
            ax.set_ylabel(r"Time ($\mu$s)")
    legend(fig, [Line2D([], [], mfc="white", mec=k["color"], mew=0.6, label=lab[s], **k) for s, k in st.items()], 3, hl=2.2)
    save(fig, "exp_scalability", 0.15)


# ---------------------------------------------------------------- 5.4 ablation
def fig_ablation():
    var = [("w/o estimate", "#D9CCEB", "////"), ("w/o bounded probes", "#BCD2EF", "\\\\\\\\"), ("w/o warm start", "#B5E1E2", "...."),
           ("w/o window", "#C6E5C3", "----"), ("OASIS", FILL["OASIS"], HATCH["OASIS"])]
    v = {(r["dataset"], r["workload"], r["variant"]): float(r["total_us"]) for r in load("ablation.csv")}
    fig, axes = quad(2.2, 1.95, sharey=True)
    n = len(var); wd = 0.84 / n
    for c, d in enumerate(DATASETS):
        ax = axes[c]; grid(ax)
        for j, (name, col, h) in enumerate(var):
            xs = [k + (j - (n - 1) / 2) * wd for k in range(2)]
            ax.bar(xs, [v[(d, wl, name)] / v[(d, wl, "OASIS")] for wl in ["Static", "Changing"]], width=wd, color=col,
                   edgecolor=EDGE, hatch=h, lw=0.35, zorder=3)
        ax.set_ylim(0.9, 1.12); ax.set_title(d)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Static", "Changing"]); ax.tick_params(axis="x", length=0)
        if (c == 0) if MDPI else (c % 2 == 0):
            ax.set_ylabel("Normalized time")
    legend(fig, [patch(col, h, name) for name, col, h in var], 3)
    save(fig, "exp_ablation", 0.30)


# ---------------------------------------------------------------- 5.4 parameters
def fig_params():
    rows = load("params.csv")
    params = [("alpha", r"$\alpha$"), ("b", r"$b$"), ("w", r"$w$")]
    vlab = {"100": ".1k", "500": ".5k", "2000": "2k", "all": "all", "0.05": ".05", "0.1": ".1", "0.25": ".25", "0.5": ".5",
            "1.5": "1.5", "2.0": "2", "3.0": "3", "5.0": "5"}
    shades = ["#BCD2EF", "#8DB3E2", "#5E8FCB", "#3B6FB6"]
    fig, axes = plt.subplots(len(params), 4, figsize=(W, 3.2 if MDPI else 2.2), sharey=True)
    for r_, (p, sym) in enumerate(params):
        sub = [x for x in rows if x["param"] == p]
        vals = list(dict.fromkeys(x["value"] for x in sub))
        for c, d in enumerate(DATASETS):
            ax = axes[r_][c]; grid(ax)
            cur = {x["value"]: x for x in sub if x["dataset"] == d}
            dflt = float(next(x["total_us"] for x in cur.values() if x["is_default"] == "1"))
            for j, val in enumerate(vals):
                e = cur[val]["is_default"] == "1"
                ax.bar(j, float(cur[val]["total_us"]) / dflt, width=0.7, color=FILL["OASIS"] if e else shades[j],
                       edgecolor=EDGE, hatch=HATCH["OASIS"] if e else "", lw=0.35, zorder=3)
            ax.set_ylim(0.9, 1.06)
            ax.set_xticks(range(len(vals))); ax.set_xticklabels([vlab.get(x, x) for x in vals])
            ax.tick_params(axis="x", length=0)
            if r_ == 0:
                ax.set_title(d)
            if c == 0:
                ax.set_ylabel(sym)
    fig.text(0.0, 0.5, "Normalized time", rotation=90, va="center", ha="left")
    fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.35, rect=(0.05, 0, 1, 1))
    fig.savefig(OUT / "exp_params.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / "exp_params.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote exp_params")


if __name__ == "__main__":
    fig_total(); fig_selected(); fig_selectors(); fig_distributions(); fig_stream()
    fig_overhead(); fig_scalability(); fig_ablation(); fig_params()
