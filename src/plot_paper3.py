"""Section 5 figures, compact version (2026-11-13): lower figures, OASIS last in every legend and bar group, UnIS-R
removed, and no figure of the selected plans. Same style and data as plot_paper2.py.
Run: FIGMODE=mdpi python plot_paper3.py <results_merged_dir> <out_dir>
"""
import matplotlib.pyplot as plt

import plot_paper as PP
from plot_paper2 import *  # noqa: F401,F403  (style, load, figures of plot_paper.py, fig_ablation2, fig_params2)

PP.METHODS.remove("UnIS-R")          # the same list object is used by every figure function


def method_handles(refs=True):       # OASIS last, after the references
    hs = [patch(FILL[m], HATCH[m], LAB.get(m, m)) for m in PP.METHODS if m != "OASIS"]
    if refs:
        hs += [Line2D([], [], label=REFLAB[k], **REF[k]) for k in REF]
    hs.append(patch(FILL["OASIS"], HATCH["OASIS"], "OASIS"))
    return hs


PP.method_handles = method_handles

_subplots = plt.subplots
SCALE = {"f": 1.0}


def _scaled_subplots(*a, **kw):       # every figure keeps its width and gets SCALE["f"] of its height
    if "figsize" in kw:
        w, h = kw["figsize"]
        kw["figsize"] = (w, h * SCALE["f"])
    return _subplots(*a, **kw)


plt.subplots = _scaled_subplots


def fig_scalability3():               # UnIS first, OASIS last
    data = defaultdict(list)
    for r in load("scalability.csv"):
        data[(r["dataset"], r["series"])].append((float(r["n_million"]), float(r["us"])))
    st = {"UnIS overhead": dict(color=LINE["UnIS"], marker="D", ls=(0, (3, 1.5)), lw=0.8, ms=2.4),
          "OASIS search": dict(color=LINE["OASIS"], marker="*", ls="-", lw=1.2, ms=3.6),
          "OASIS overhead": dict(color=LINE["OASIS"], marker="*", ls=(0, (3, 1.5)), lw=1.0, ms=3.6)}
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


if __name__ == "__main__":
    for f, scale in [(fig_total, 0.72), (fig_selectors, 0.70), (fig_distributions, 0.70), (fig_stream, 0.75),
                     (fig_overhead, 0.78), (fig_scalability3, 0.75), (fig_ablation2, 0.73), (fig_params2, 0.75)]:
        SCALE["f"] = scale
        f()
