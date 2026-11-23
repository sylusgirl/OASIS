"""Section 5 figures with the OASIS rows of the 2026-11-13 revision (two modules: two-level feature extraction and limited
trials with a time limit; recency weighting dropped). Every figure of plot_paper.py in the same style, with the ablation
replaced by a leave-one-out ablation and the parameter study by the new parameters.
Run: FIGMODE=mdpi python plot_paper2.py <results_merged_dir> <out_dir>
"""
from plot_paper import *  # noqa: F401,F403  (RES, OUT, style, load, quad, grid, legend, save, patch, FILL, HATCH, EDGE)

ABL = [("Base", "Without both", "#E3E3E3", ""), ("w/o Features", "Without two-level features", "#BCD2EF", "////"),
       ("w/o Trials", "Without time-limited trials", "#8DB3E2", "\\\\\\\\"), ("OASIS", "OASIS", FILL["OASIS"], HATCH["OASIS"])]


def fig_ablation2():
    rows = load("ablation2.csv")
    acc, tot = defaultdict(list), defaultdict(list)
    for r in rows:
        acc[(r["dataset"], r["variant"])].append(float(r["pred_accuracy_pct"]))
        tot[(r["dataset"], r["variant"])].append(float(r["total_us"]))
    mean = lambda v: sum(v) / len(v)
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.0 if MDPI else 2.3))
    for c, d in enumerate(DATASETS):
        full = mean(tot[(d, "OASIS")])
        for r_, (val, ylab) in enumerate([(lambda v: mean(acc[(d, v)]), "Accuracy (%)"),
                                          (lambda v: mean(tot[(d, v)]) / full, "Normalized time")]):
            ax = axes[r_][c]; grid(ax)
            ys = [val(v) for v, _, _, _ in ABL]
            for j, (v, lab, col, h) in enumerate(ABL):
                ax.bar(j, ys[j], width=0.72, color=col, edgecolor=EDGE, hatch=h, lw=0.35, zorder=3)
                if r_ == 1 and v != "OASIS":
                    ax.text(j, ys[j], f"{ys[j]:.2f}", ha="center", va="bottom", fontsize=5.4)
            if r_ == 0:
                ax.set_ylim(0, max(ys) * 1.15); ax.set_title(d)
            else:
                ax.set_ylim(0.8, max(ys) * 1.08); ax.axhline(1.0, color="#555555", lw=0.6, ls=(0, (3, 1.5)), zorder=2)
            ax.set_xticks([]); ax.tick_params(axis="x", length=0)
            if c == 0:
                ax.set_ylabel(ylab)
    legend(fig, [patch(col, h, lab) for _, lab, col, h in ABL], 2)
    save(fig, "exp_ablation", 0.36)


def fig_params2():
    rows = load("params2.csv")
    params = [("b", r"$b$"), ("T0", r"$\theta_0$"), ("r", r"$h$"), ("tau", r"$\tau$")]
    shades = ["#BCD2EF", "#8DB3E2", "#5E8FCB", "#3B6FB6"]
    fig, axes = plt.subplots(len(params), 4, figsize=(W, 4.0 if MDPI else 3.0), sharey=True)
    for r_, (p, sym) in enumerate(params):
        sub = [x for x in rows if x["param"] == p]
        vals = list(dict.fromkeys(x["value"] for x in sub))
        for c, d in enumerate(DATASETS):
            ax = axes[r_][c]; grid(ax)
            cur = {x["value"]: x for x in sub if x["dataset"] == d}
            dflt = float(next(x["total_us"] for x in cur.values() if x["is_default"] == "1"))
            for j, val in enumerate(vals):
                e = cur[val]["is_default"] == "1"
                ax.bar(j, float(cur[val]["total_us"]) / dflt, width=0.7, color=FILL["OASIS"] if e else shades[j % 4],
                       edgecolor=EDGE, hatch=HATCH["OASIS"] if e else "", lw=0.35, zorder=3)
            ax.set_ylim(0.9, 1.12)
            ax.set_xticks(range(len(vals))); ax.set_xticklabels(vals, fontsize=5.8)
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
    fig_overhead(); fig_scalability(); fig_ablation2(); fig_params2()
