"""Experiment driver: replays all selectors on the measured streams of one dataset and writes the figure CSVs.
Run: python replay.py <project_dir> <dataset> <n_use> [--tune | --scal]
"""
from replay_core import *  # noqa: F401,F403  (P, DS, N, R, CFG, loaders, selectors, costs)

MAIN = ["Random", "Frozen", "AI+R", "UnIS", "UnIS-R", "LQP", "OASIS"]
OFFSEL = {"RF": ("rf", f_sel), "GBDT": ("gbr", f_sel), "kNN": ("knn", f_sel), "MLP": ("mlp", f_sel), "Ridge": ("ridge", f_sel)}
LINEAR = {"LinUCB": "lcb", "TS": "ts", "eps-greedy": "eps", "SquareCB": "squarecb", "UCB1": "ucb1"}
NAMES = ["R-BF", "KD-BF", "KD-DFS-MBR", "KD-BFS-MBR", "KD-DFS-Ball", "KD-BFS-Ball", "Grid"]
B = dict(alpha=CFG.get("alpha", 0.25), bound=CFG.get("bound", 2.0), pace=CFG.get("pace", 4), window=CFG.get("window", 500),
         revive=CFG.get("revive", 1000))


def write(name, cols, rows):
    pd.DataFrame(rows, columns=cols).to_csv(R / name, index=False)


def oasis(df, T, **kw):
    """Returns the picks and the extra time of aborted probes (probe until abort + re-run by the incumbent)."""
    a = dict(B); a.update(kw)
    ty, size = ctx_size(df, df.est.values, a.get("use_est", True))
    p, c = oasis_ctx(T, ty, size, alpha=a["alpha"], bound=a["bound"], pace=a["pace"], window=a["window"], revive=a["revive"],
                     warm=a.get("warm", True))
    return p, c - T[np.arange(len(T)), p]


def oasis_cost(df, T, tag, **kw):
    a = dict(B); a.update(kw)
    return ctx_cost_us(df, T, df.est.values, a["alpha"], a["bound"], a["pace"], a["window"], a["revive"], tag)


def linear(df, T, rule):
    return online(f_lin(df), T, df.qtype.values, rule, 0.5, 0.1, 2000, target="log")


def main():
    trF, TtrF = load("train_full"); trH, TtrH = load("train_half")
    S = {k: load(k) for k in ["mixed", "insert", "drift"] + [f"dist_{d}" for d in ["uniform", "data", "hotspot", "zipf", "trajectory"]]}
    mixed_df, mixed_T = S["mixed"]
    feat_us = mixed_df.feat_ns.mean() / 1000
    path_us = mixed_df.path_ns.mean() / 1000

    # offline models: static streams use the full index, the insertion stream the initial half
    models = {}
    for tag, (df, T) in [("full", (trF, TtrF)), ("half", (trH, TtrH))]:
        models[tag] = {"UnIS": Offline("rf", f_unis).fit(df, T), "AI+R": Offline("dt", f_air).fit(df, T),
                       "LQP": Offline("gbr", f_lqp).fit(df, T)}
        for k, (kind, feat) in OFFSEL.items():
            models[tag][k] = Offline(kind, feat).fit(df, T)
    full = models["full"]

    # per-query selection cost, measured in C++ on the mixed stream
    cost = {"Random": 0.0, "Frozen": 0.0, "BestFixed": 0.0, "Oracle": 0.0}
    cost["UnIS"] = path_us + full["UnIS"].cost_us(mixed_df, "unis")
    cost["UnIS-R"] = cost["UnIS"]
    cost["AI+R"] = full["AI+R"].cost_us(mixed_df, "air")
    cost["LQP"] = feat_us + full["LQP"].cost_us(mixed_df, "lqp")
    for k in OFFSEL:
        cost[k] = feat_us + full[k].cost_us(mixed_df, k)
    for k, rule in LINEAR.items():
        cost[k] = feat_us + online_cost_us(mixed_df, mixed_T, rule, 0.5, 0.1, 2000, rule, "log", feats=f_lin(mixed_df))
    cost["OASIS"] = oasis_cost(mixed_df, mixed_T, "oasis")
    json.dump(cost, open(R / "cost_us.json", "w"), indent=1)

    def picks(df, T, tag):
        ty = df.qtype.values
        m = models[tag]
        out = {"Random": random_pick(len(T)), "Frozen": frozen_pick(T, ty), "AI+R": m["AI+R"].predict(df),
               "UnIS": m["UnIS"].predict(df), "LQP": m["LQP"].predict(df),
               "BestFixed": best_fixed(T, ty), "Oracle": T.argmin(1)}
        out["OASIS"], ex_o = oasis(df, T)
        out["UnIS-R"], ex_u = unis_r(df, T, m["UnIS"])
        return out, {"UnIS-R": ex_u, "OASIS": ex_o}

    def totals(T, pk, extra):
        return {k: gather(T, p, cost[k], extra.get(k)) for k, p in pk.items()}

    # 5.2 Overall query time and selected strategies (mixed stream)
    pk, ex = picks(mixed_df, mixed_T, "full")
    tot = totals(mixed_T, pk, ex)
    rows = []
    for t, tn in enumerate(TYPES):
        m = mixed_df.qtype.values == t
        rows += [["Server", DS, tn, k, tot[k][m].mean()] for k in MAIN + ["BestFixed", "Oracle"]]
    write("total_time.csv", ["platform", "dataset", "qtype", "method", "total_us"], rows)
    rows = []
    for t, tn in enumerate(TYPES):
        m = mixed_df.qtype.values == t
        for k in ["Oracle", "OASIS"]:
            share = np.bincount(pk[k][m], minlength=M) / m.sum() * 100
            rows += [[DS, tn, k, NAMES[a], share[a]] for a in range(M)]
    write("selected.csv", ["dataset", "qtype", "method", "strategy", "share_pct"], rows)

    # 5.2 Query distributions
    rows = []
    for d in ["uniform", "data", "hotspot", "zipf", "trajectory"]:
        df, T = S[f"dist_{d}"]
        p, e = picks(df, T, "full"); tt = totals(T, p, e)
        rows += [[DS, d.capitalize(), k, tt[k].mean()] for k in MAIN + ["BestFixed", "Oracle"]]
    write("distributions.csv", ["dataset", "distribution", "method", "total_us"], rows)

    # 5.2 Insertions and query drift (windows of 2% of the stream)
    rows, summ = [], []
    summ += [[DS, "mixed", k, tot[k].mean(), acc(mixed_T, pk[k]).mean() * 100] for k in MAIN + ["BestFixed", "Oracle"]]
    for wl, key, tag in [("Insertion", "insert", "half"), ("Drift", "drift", "full")]:
        df, T = S[key]
        p, e = picks(df, T, tag); tt = totals(T, p, e)
        w = len(T) // 50
        for i in range(50):
            sl = slice(i * w, (i + 1) * w)
            rows += [[DS, wl, (i + 1) * 2, k, acc(T[sl], p[k][sl]).mean() * 100, tt[k][sl].mean()] for k in MAIN + ["BestFixed", "Oracle"]]
        summ += [[DS, key, k, tt[k].mean(), acc(T, p[k]).mean() * 100] for k in MAIN + ["BestFixed", "Oracle"]]
    for d in ["uniform", "data", "hotspot", "zipf", "trajectory"]:
        df, T = S[f"dist_{d}"]
        p, e = picks(df, T, "full"); tt = totals(T, p, e)
        summ += [[DS, f"dist_{d}", k, tt[k].mean(), acc(T, p[k]).mean() * 100] for k in MAIN + ["BestFixed", "Oracle"]]
    write("stream.csv", ["dataset", "workload", "progress_pct", "method", "accuracy_pct", "total_us"], rows)
    write("unisr_cost.csv", ["at_query", "relabel_us", "train_us"], UNISR_LOG)
    write("summary.csv", ["dataset", "stream", "method", "total_us", "accuracy_pct"], summ)

    # 5.2 Selector models: same context, different selectors (static = mixed, changing = insert + drift)
    rows = []
    for wl, keys in [("Static", ["mixed"]), ("Changing", ["insert", "drift"])]:
        for k in list(OFFSEL) + list(LINEAR) + ["OASIS"]:
            vals = []
            for key in keys:
                df, T = S[key]; tag = "half" if key == "insert" else "full"
                if k in OFFSEL:
                    p = models[tag][k].predict(df)
                elif k in LINEAR:
                    p = linear(df, T, LINEAR[k])
                else:
                    p, e = oasis(df, T)
                vals.append(gather(T, p, cost[k], e if k == "OASIS" else None).mean())
            rows.append([DS, wl, k, "offline" if k in OFFSEL else "online", float(np.mean(vals))])
    write("selectors.csv", ["dataset", "workload", "selector", "family", "total_us"], rows)

    # 5.3 Overhead per component (mixed stream)
    rows = []
    idx = np.arange(len(mixed_T))
    for t, tn in enumerate(TYPES):
        m = mixed_df.qtype.values == t
        rows += [[DS, tn, "UnIS", "Features", mixed_df.path_ns[m].mean() / 1000],
                 [DS, tn, "UnIS", "Prediction", cost["UnIS"] - path_us],
                 [DS, tn, "UnIS", "Search", mixed_T[idx, pk["UnIS"]][m].mean()],
                 [DS, tn, "OASIS", "Selection", cost["OASIS"]],
                 [DS, tn, "OASIS", "Search", (mixed_T[idx, pk["OASIS"]] + ex["OASIS"])[m].mean()]]
    write("overhead.csv", ["dataset", "qtype", "method", "component", "us"], rows)

    # 5.3 Memory and one-time cost of the offline selector
    rf = full["UnIS"].m
    nodes = sum(e.tree_.node_count for e in rf.estimators_)
    ty_m, sz_m = ctx_size(mixed_df, mixed_df.est.values)
    ctx_used = len(np.unique(ty_m * 8 + sz_m))
    g_cells = (65 ** 2) if mixed_df.z2.abs().max() == 0 else 25 ** 3
    oasis_kb = (ctx_used * M * (2 * 8 + 8 + 4) + ctx_used * B["window"] * 12 + g_cells * 8) / 1024
    write("memory.csv", ["dataset", "unis_label_s", "unis_train_s", "unis_model_kb", "oasis_state_kb"],
          [[DS, TtrF.sum() / 1e6, full["UnIS"].fit_s, nodes * (4 * 4 + 8 + M * 8) / 1024, oasis_kb]])

    # 5.4 Ablation (static = mixed, changing = insert + drift)
    rows = []
    variants = {"w/o estimate": dict(use_est=False), "w/o bounded probes": dict(bound=1e12),
                "w/o warm start": dict(warm=False), "w/o window": dict(window=0), "OASIS": {}}
    for wl, keys in [("Static", ["mixed"]), ("Changing", ["insert", "drift"])]:
        for v, kw in variants.items():
            vals = []
            for k in keys:
                p, e = oasis(S[k][0], S[k][1], **kw)
                vals.append(gather(S[k][1], p, cost["OASIS"], e).mean())
            rows.append([DS, wl, v, float(np.mean(vals))])
    write("ablation.csv", ["dataset", "workload", "variant", "total_us"], rows)

    # 5.4 Parameter study (mean over mixed, insert and drift)
    grid = {"alpha": [0.05, 0.1, 0.25, 0.5], "b": [1.5, 2.0, 3.0, 5.0], "w": [100, 500, 2000, 0], "rule": ["OASIS", "LinUCB", "TS", "eps-greedy"]}
    key_of = {"alpha": "alpha", "b": "bound", "w": "window"}
    rows = []
    for pname, vals in grid.items():
        for v in vals:
            res = []
            for k in ["mixed", "insert", "drift"]:
                df, T = S[k]
                if pname == "rule":
                    c = cost[v]
                    p, e = oasis(df, T) if v == "OASIS" else (linear(df, T, LINEAR[v]), None)
                else:
                    kw = {key_of[pname]: v}; c = cost["OASIS"]
                    p, e = oasis(df, T, **kw)
                res.append(gather(T, p, c, e).mean())
            dflt = (v == "OASIS") if pname == "rule" else (v == B[key_of[pname]])
            lab = ("all" if v == 0 else v) if pname == "w" else v
            rows.append([DS, pname, lab, int(dflt), float(np.mean(res))])
    write("params.csv", ["dataset", "param", "value", "is_default", "total_us"], rows)
    print(DS, "done", flush=True)


def scalability(sizes):
    rows = []
    rf = Offline("rf", f_unis).fit(*load("train_full"))
    for n in sizes:
        f = P / os.environ.get("OASIS_WORK", "work8") / f"{DS}_{n}" / "mixed.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f); T = df[[f"t{s}" for s in range(M)]].values / 1000.0
        df["est"] = np.load(f.parent / "est_mixed.npy")
        p, e = oasis(df, T)
        on = oasis_cost(df, T, f"s{n}")
        un = df.path_ns.mean() / 1000 + rf.cost_us(df, f"s{n}")
        rows += [[DS, n / 1e6, "OASIS search", (T[np.arange(len(T)), p] + e).mean()], [DS, n / 1e6, "OASIS overhead", on],
                 [DS, n / 1e6, "UnIS overhead", un]]
    write("scalability.csv", ["dataset", "n_million", "series", "us"], rows)


def tune():
    df, T = load("valid")
    ty = df.qtype.values
    bf = T[np.arange(len(T)), best_fixed(T, ty)].mean()
    rows = []
    c = oasis_cost(df, T, "tune")
    for a in [0.1, 0.25, 0.5]:
        for bd in [1.5, 2.0, 3.0]:
            for pc in [1, 2, 4, 8]:
                for w in [500, 2000]:
                    p, e = oasis(df, T, alpha=a, bound=bd, pace=pc, window=w)
                    rows.append([DS, "ctx", a, bd, pc, w, gather(T, p, c, e).mean(), bf, c])
    write("tune.csv", ["dataset", "rule", "alpha", "bound", "pace", "window", "total_us", "bestfixed_us", "cost_us"], rows)


if __name__ == "__main__":
    if "--tune" in sys.argv:
        tune()
    elif "--scal" in sys.argv:
        scalability([250000, 500000, 1000000, 2000000])
    else:
        main()
