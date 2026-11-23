"""Replay OASIS (2026-11-13 revision, oasis2.py) on the measured streams of one dataset. The other selectors are not
rerun: their rows stay as in results/<dataset>/. Writes results2/<dataset>/*.csv with the same columns as the old CSV
files (OASIS rows only), plus ablation2.csv (incremental ablation), params2.csv, tune2.csv and info2.csv.
The replays run in parallel (env OASIS_PROCS); the selection cost is measured in C++ in a separate, sequential step.
Run: python replay2.py <project_dir> <dataset> <n_use> [--tune | --cost | --scal | --smoke]
"""
from multiprocessing import Pool

from replay_core import *  # noqa: F401,F403  (P, DS, N, W, TMP, M, TYPES, load, gather, acc, best_fixed, wmat)
import oasis2 as O

NAMES = ["R-BF", "KD-BF", "KD-DFS-MBR", "KD-BFS-MBR", "KD-DFS-Ball", "KD-BFS-Ball", "Grid"]
R2 = P / "results2" / DS
R2.mkdir(parents=True, exist_ok=True)
SB2 = str(Path(__file__).resolve().parent / "selbench2")
_cfg = P / "results2" / "oasis2_config.json"
CFG2 = json.loads(_cfg.read_text()) if _cfg.exists() else {}
B2 = dict(O.DEFAULT); B2.update({k: v for k, v in CFG2.items() if k in O.DEFAULT})
DISTS = ["uniform", "data", "hotspot", "zipf", "trajectory"]
STREAMS = ["mixed", "insert", "drift"] + [f"dist_{d}" for d in DISTS]
ABL = {"Base": dict(features=False, limited=False, recency=False),            # leave-one-out with two modules
       "w/o Trials": dict(features=True, limited=False, recency=False),
       "w/o Features": dict(features=False, limited=True, recency=False),
       "OASIS": dict(features=True, limited=True, recency=False)}
PARAMS = {"b": [1.5, 2.0, 3.0, 5.0], "T0": [0.02, 0.05, 0.1, 0.25], "r": [2, 3, 4, 7], "tau": [0.25, 0.5, 1.0, 2.0]}
SIZES = [250000, 500000, 1000000, 2000000]
PROCS = int(os.environ.get("OASIS_PROCS", "8"))
G = {}   # data shared with the worker processes (fork)


def write(name, cols, rows):
    pd.DataFrame(rows, columns=cols).to_csv(R2 / name, index=False)


def feats(name, df, wdir=None):
    wdir = wdir or W
    (wdir / "tmp").mkdir(exist_ok=True)
    f = wdir / "tmp" / f"feat3_{name}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    d = O.query_features(wdir, name, df)
    np.savez(f, **d)
    return d


def run(T, F, **kw):
    a = dict(B2); a.update(kw)
    return O.oasis2(T, F["ty"], F["phi"], O.coarse_cells(F, int(a["cf"])), **a)


def _task(job):                      # worker: (stream key, overrides) -> results
    key, kw = job
    T, F = G["S"][key][1], G["F"][key]
    p, e, a, info = run(T, F, **kw)
    return key, kw, p, e, a, info


def pmap(jobs):
    with Pool(min(PROCS, len(jobs))) as pool:
        return pool.map(_task, jobs, chunksize=1)


def sb2(*args):
    out = subprocess.run([SB2, *map(str, args)], capture_output=True, text=True, check=True).stdout
    return float(out.strip()) / 1000.0   # microseconds


def cost(F, T, tag, **kw):
    a = dict(B2); a.update(kw)
    ft, fy, fz = TMP / f"t2_{tag}.bin", TMP / f"y2_{tag}.bin", TMP / f"z2_{tag}.bin"
    wmat(ft, T * 1000.0); wmat(fy, F["ty"].astype(float)[:, None]); wmat(fz, np.c_[F["z"], F["param"], F["half"]])
    return sb2("ctx2", ft, fy, fz, int(F["d"]), a["tau"], a["K"], a["r"], a["N0"], a["T0"], a["gamma"], a["b"], a["lam"],
               a["w"], int(a["cf"]), int(a["features"]), int(a["limited"]), int(a["recency"]))


def predicted(p, astar):
    return np.where(astar >= 0, astar, p)


def load_at(wdir):
    df = pd.read_csv(wdir / "mixed.csv"); T = df[[f"t{s}" for s in range(M)]].values / 1000.0
    df["est"] = np.load(wdir / "est_mixed.npy")
    return df, T


def costs():
    """Selection cost of OASIS and of the ablation variants (mixed stream), and of OASIS at every data size.
    Run alone, one measurement at a time, so that no other replay disturbs the timing."""
    df, T = load("mixed"); Fm = feats("mixed", df)
    out = {"OASIS": cost(Fm, T, "oasis2")}
    for v, kw in ABL.items():
        out[f"abl:{v}"] = cost(Fm, T, "abl_" + "".join(ch for ch in v if ch.isalnum()), **kw)
    for n in SIZES:
        wdir = P / os.environ.get("OASIS_WORK", "work8") / f"{DS}_{n}"
        if (wdir / "mixed.csv").exists():
            d2, T2 = load_at(wdir)
            out[f"n:{n}"] = cost(feats("mixed", d2, wdir), T2, f"s{n}")
    json.dump(out, open(R2 / "cost2.json", "w"), indent=1)
    json.dump({"OASIS": out["OASIS"]}, open(R2 / "cost_us.json", "w"), indent=1)
    print(DS, "costs", out, flush=True)


def main():
    S = {k: load(k) for k in STREAMS}
    F = {k: feats(k, S[k][0]) for k in S}
    G["S"], G["F"] = S, F
    C = json.load(open(R2 / "cost2.json")); c = C["OASIS"]
    jobs = [(k, {}) for k in STREAMS]
    jobs += [(k, kw) for v, kw in ABL.items() if v != "OASIS" for k in ["mixed", "insert", "drift"]]
    jobs += [(k, {pn: v}) for pn, vals in PARAMS.items() for v in vals if v != B2[pn] for k in ["mixed", "insert", "drift"]]
    out = {}
    for key, kw, p, e, a, info in pmap(jobs):
        out[(key, tuple(sorted(kw.items())))] = (p, e, a, info)
    res = {k: out[(k, ())] for k in STREAMS}
    info_rows = [[DS, k, r[3]["classes"], r[3]["cells"], r[3]["trials"] * 100, r[3]["stopped"] * 100] for k, r in res.items()]
    write("info2.csv", ["dataset", "stream", "classes", "cells", "trials_pct", "stopped_pct"], info_rows)
    tot = {k: gather(S[k][1], res[k][0], c, res[k][1]) for k in STREAMS}

    # 5.2 Overall query time and selected plans (mixed stream)
    mdf, mT = S["mixed"]; p = res["mixed"][0]
    rows, sel = [], []
    for t, tn in enumerate(TYPES):
        m = mdf.qtype.values == t
        rows.append(["Server", DS, tn, "OASIS", tot["mixed"][m].mean()])
        share = np.bincount(p[m], minlength=M) / m.sum() * 100
        sel += [[DS, tn, "OASIS", NAMES[s], share[s]] for s in range(M)]
    write("total_time.csv", ["platform", "dataset", "qtype", "method", "total_us"], rows)
    write("selected.csv", ["dataset", "qtype", "method", "strategy", "share_pct"], sel)

    # 5.2 Query distributions
    write("distributions.csv", ["dataset", "distribution", "method", "total_us"],
          [[DS, d.capitalize(), "OASIS", tot[f"dist_{d}"].mean()] for d in DISTS])

    # 5.2 Insertions and query drift (windows of 2% of the stream) and per-stream summary
    rows, summ = [], []
    for wl, key in [("Insertion", "insert"), ("Drift", "drift")]:
        T = S[key][1]; p = res[key][0]; w = len(T) // 50
        for i in range(50):
            sl = slice(i * w, (i + 1) * w)
            rows.append([DS, wl, (i + 1) * 2, "OASIS", acc(T[sl], p[sl]).mean() * 100, tot[key][sl].mean()])
    for key in STREAMS:
        T = S[key][1]; p, e, a, info = res[key]
        summ.append([DS, key, "OASIS", tot[key].mean(), acc(T, p).mean() * 100, acc(T, predicted(p, a)).mean() * 100])
    write("stream.csv", ["dataset", "workload", "progress_pct", "method", "accuracy_pct", "total_us"], rows)
    write("summary.csv", ["dataset", "stream", "method", "total_us", "accuracy_pct", "pred_accuracy_pct"], summ)

    # 5.2 Selector models (OASIS row only)
    write("selectors.csv", ["dataset", "workload", "selector", "family", "total_us"],
          [[DS, "Static", "OASIS", "online", tot["mixed"].mean()],
           [DS, "Changing", "OASIS", "online", float(np.mean([tot["insert"].mean(), tot["drift"].mean()]))]])

    # 5.3 Overhead (mixed stream) and memory
    p, e = res["mixed"][0], res["mixed"][1]; idx = np.arange(len(mT)); rows = []
    for t, tn in enumerate(TYPES):
        m = mdf.qtype.values == t
        rows += [[DS, tn, "OASIS", "Selection", c], [DS, tn, "OASIS", "Search", (mT[idx, p] + e)[m].mean()]]
    write("overhead.csv", ["dataset", "qtype", "method", "component", "us"], rows)
    inf = res["mixed"][3]; d = int(F["mixed"]["d"]); g = 64 if d == 2 else 24
    win = B2["w"] if B2["recency"] else 0
    kb = (inf["classes"] * (M * (2 * 8 + 4) + 3 * 8 + win * 24) + inf["cells"] * (M * (2 * 8 + 4) + 16)
          + ((g + 1) ** d + g ** d) * 8) / 1024
    write("memory.csv", ["dataset", "oasis_state_kb", "classes", "cells"], [[DS, kb, inf["classes"], inf["cells"]]])

    # 5.4 Incremental ablation: the three modules added one by one (mixed, insertion and drift streams)
    rows = []
    for v, kw in ABL.items():
        cv = C[f"abl:{v}"]
        for key in ["mixed", "insert", "drift"]:
            T = S[key][1]
            p, e, a, info = res[key] if v == "OASIS" else out[(key, tuple(sorted(kw.items())))]
            rows.append([DS, key, v, gather(T, p, cv, e).mean(), acc(T, predicted(p, a)).mean() * 100, acc(T, p).mean() * 100,
                         info["trials"] * 100, cv])
    write("ablation2.csv", ["dataset", "stream", "variant", "total_us", "pred_accuracy_pct", "exec_accuracy_pct", "trials_pct",
                            "selection_us"], rows)

    # 5.4 Parameter study (mean over the mixed, insertion and drift streams)
    rows = []
    for pn, vals in PARAMS.items():
        for v in vals:
            o = []
            for key in ["mixed", "insert", "drift"]:
                T = S[key][1]
                p, e, a, info = res[key] if v == B2[pn] else out[(key, ((pn, v),))]
                o.append(gather(T, p, c, e).mean())
            rows.append([DS, pn, "all" if (pn == "w" and v == 0) else v, int(v == B2[pn]), float(np.mean(o))])
    write("params2.csv", ["dataset", "param", "value", "is_default", "total_us"], rows)
    print(DS, "done", flush=True)


def scalability():
    C = json.load(open(R2 / "cost2.json"))
    rows = []
    for n in SIZES:
        wdir = P / os.environ.get("OASIS_WORK", "work8") / f"{DS}_{n}"
        if not (wdir / "mixed.csv").exists():
            continue
        df, T = load_at(wdir); Fm = feats("mixed", df, wdir)
        p, e, a, info = run(T, Fm)
        rows += [[DS, n / 1e6, "OASIS search", (T[np.arange(len(T)), p] + e).mean()], [DS, n / 1e6, "OASIS overhead", C[f"n:{n}"]]]
    write("scalability.csv", ["dataset", "n_million", "series", "us"], rows)


def _tune_task(cfg):
    T, Fv = G["T"], G["F"]
    p, e, a, info = run(T, Fv, **cfg)
    return cfg, p, e


def tune():
    df, T = load("valid")
    Fv = feats("valid", df)
    G["T"], G["F"] = T, Fv
    bf = T[np.arange(len(T)), best_fixed(T, df.qtype.values)].mean()
    cfgs = [dict(tau=tau, r=r, T0=T0, b=b, N0=N0, cf=cf) for tau in [0.5, 1.0, 2.0] for r in [2, 3] for T0 in [0.02, 0.05, 0.1]
            for b in [2.0, 3.0] for N0 in [8.0, 32.0, 1e9] for cf in [4, 8]]
    with Pool(min(PROCS, len(cfgs))) as pool:
        res = pool.map(_tune_task, cfgs, chunksize=2)
    rows = [[DS, c["tau"], c["r"], c["T0"], c["b"], c["N0"], c["cf"], gather(T, p, 0.0, e).mean(), bf, T.min(1).mean()]
            for c, p, e in res]
    write("tune2.csv", ["dataset", "tau", "r", "T0", "b", "N0", "cf", "total_us", "bestfixed_us", "oracle_us"], rows)
    print(DS, "tuned", flush=True)


if __name__ == "__main__":
    if "--tune" in sys.argv:
        tune()
    elif "--cost" in sys.argv:
        costs()
    elif "--scal" in sys.argv:
        scalability()
    elif "--smoke" in sys.argv:            # quick check on the first 2,000 queries of the mixed stream
        df, T = load("mixed"); Fm = feats("mixed", df)
        T, Fm = T[:2000], {k: (v[:2000] if getattr(v, "ndim", 0) and len(v) == len(df) else v) for k, v in Fm.items()}
        for v, kw in ABL.items():
            p, e, a, info = run(T, Fm, **kw)
            print(v, round(gather(T, p, 0.0, e).mean(), 2), "us", round(acc(T, predicted(p, a)).mean() * 100, 1), "%", info, flush=True)
        print("best fixed", round(T[np.arange(len(T)), best_fixed(T, df.qtype.values[:2000])].mean(), 2), "oracle", round(T.min(1).mean(), 2))
        print("cost_us", cost(Fm, T, "smoke"), flush=True)
    else:
        main()
