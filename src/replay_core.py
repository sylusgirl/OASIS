"""Replay every selector on the measured streams of one dataset and write the CSV files for the figures.

For every query the harness (bench) already measured the search time of all seven strategies. A selector picks one
strategy per query using only earlier feedback; its total time is the measured search time of that strategy plus its
own selection cost, which is measured in C++ (selbench) on the same feature stream, plus, for UnIS-R, the time to
relabel (run all strategies on recent queries) and retrain.
Run: python replay.py <project_dir> <dataset> <n_use> [--tune]
"""
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

P, DS, N = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
import os
W = P / os.environ.get("OASIS_WORK", "work8") / f"{DS}_{N}"
R = P / "results" / DS
R.mkdir(parents=True, exist_ok=True)
TMP = W / "tmp"
TMP.mkdir(exist_ok=True)
SB = str(P / "src" / "selbench")
M, TYPES = 7, ["kNN", "Radius", "Range"]
CFG = json.loads((P / "results" / "oasis_config.json").read_text()) if (P / "results" / "oasis_config.json").exists() else \
    {"rule": "lcb", "alpha": 1.0, "lam": 1.0, "window": 2000, "target": "raw"}
ACC_TOL = 1.05


def load(name):
    df = pd.read_csv(W / f"{name}.csv")
    T = df[[f"t{s}" for s in range(M)]].values / 1000.0          # microseconds
    f = W / f"est_{name}.npy"
    if f.exists():
        df["est"] = np.load(f)
    return df, T


def onehot(df):
    return np.eye(3)[df.qtype.values]


def f_oasis(df):
    return df[[f"x{i}" for i in range(8)]].values.astype(float)


def f_unis(df):   # query coordinates + index-path statistics (UnIS-style)
    return np.c_[df[["z0", "z1", "z2"]].values, onehot(df), df.ell.values, df.depth.values, np.log(df.leafsz.values),
                 np.log(df.leafvol.values + 1e-18), df.leafdist.values]


def f_air(df):    # query coordinates and size (AI+R-tree-style classifier)
    return np.c_[df[["z0", "z1", "z2"]].values, onehot(df), df.ell.values]


def f_lqp(df):    # dataset and query statistics (learned query planner), with the same answer estimate as OASIS
    return np.c_[np.log(df.ninserted.values), np.log1p(df.cellcnt.values), np.log1p(df.est.values), onehot(df),
                 df.ell.values, (df.x7.values > 0).astype(float)]


def f_sel(df):    # common features for the selector-model comparison
    return np.c_[f_lin(df), onehot(df)]


def wmat(path, A):
    A = np.ascontiguousarray(A, dtype=np.float64)
    with open(path, "wb") as f:
        f.write(struct.pack("<qq", *A.shape)); f.write(A.tobytes())


def sb(*args):
    out = subprocess.run([SB, *map(str, args)], capture_output=True, text=True, check=True).stdout
    return float(out.strip()) / 1000.0   # microseconds


# ---------------------------------------------------------------- online selectors (same algorithms as selbench)
def bucket(x):
    """Context bucket of the tabular rule: size (4) x density (4) x outside (2)."""
    return min(3, int(x[1] * 4)) * 8 + min(3, int(x[5] * 4)) * 2 + (1 if x[7] > 0 else 0)


def online(X, T, types, rule="lcb", alpha=1.0, lam=1.0, window=2000, mask=None, seed=0, target="raw"):
    rng = np.random.default_rng(seed)
    X = X.copy()
    if mask is not None:
        X[:, mask] = 0.0
    C = np.log(T) if target == "log" else T / 1000.0          # learning target per (query, strategy)
    if rule == "tab":
        cnt = np.zeros((3, 32, M)); sm = np.zeros((3, 32, M)); win = [[], [], []]; pick = np.zeros(len(X), int)
        for q in range(len(X)):
            t, k = types[q], bucket(X[q])
            zero = np.flatnonzero(cnt[t, k] == 0)
            a = int(zero[0]) if len(zero) else int(np.argmin(sm[t, k] / cnt[t, k] - alpha * .5 / np.sqrt(cnt[t, k])))
            pick[q] = a; cnt[t, k, a] += 1; sm[t, k, a] += C[q, a]
            if window > 0:
                win[t].append((q, k, a))
                if len(win[t]) > window:
                    r, k2, a2 = win[t].pop(0); cnt[t, k2, a2] -= 1; sm[t, k2, a2] -= C[r, a2]
        return pick
    d = X.shape[1]
    Ai = np.tile(np.eye(d) / lam, (3, M, 1, 1)); b = np.zeros((3, M, d)); th = np.zeros((3, M, d))
    cnt = np.zeros((3, M)); csum = np.zeros((3, M)); seen = np.zeros(3, int)
    win = [[], [], []]; pick = np.zeros(len(X), int)

    def upd(t, a, x, c, s):
        u = Ai[t, a] @ x
        Ai[t, a] -= s * np.outer(u, u) / (1 + s * x @ u)
        b[t, a] += s * c * x; th[t, a] = Ai[t, a] @ b[t, a]
        cnt[t, a] += s; csum[t, a] += s * c

    for q in range(len(X)):
        t, x = types[q], X[q]
        seen[t] += 1
        if seen[t] <= 4 * M:
            a = (seen[t] - 1) % M
        elif rule == "lcb":
            w = np.sqrt(np.maximum(0, np.einsum("i,aij,j->a", x, Ai[t], x)))
            a = int(np.argmin(th[t] @ x - alpha * w))
        elif rule == "ts":
            s = [rng.multivariate_normal(th[t, k], alpha ** 2 * Ai[t, k], method="cholesky") @ x for k in range(M)]
            a = int(np.argmin(s))
        elif rule == "eps":
            a = int(rng.integers(M)) if rng.random() < .05 else int(np.argmin(th[t] @ x))
        elif rule == "squarecb":
            yh = th[t] @ x; am = int(np.argmin(yh)); p = 1.0 / (M + 200 * (yh - yh[am])); p[am] = 0; p[am] = 1 - p.sum()
            a = int(rng.choice(M, p=p))
        else:  # ucb1 on mean cost
            mean = csum[t] / np.maximum(1, cnt[t])
            a = int(np.argmin(mean - alpha * np.sqrt(2 * np.log(seen[t]) / np.maximum(1, cnt[t]))))
        pick[q] = a
        upd(t, a, x, C[q, a], +1)
        if window > 0:
            win[t].append((q, a))
            if len(win[t]) > window:
                r, a2 = win[t].pop(0)
                upd(t, a2, X[r], C[r, a2], -1)
    return pick


def online_lazy(X, T, types, alpha=0.5, lam=0.1, window=2000, r=8, expire=256, mask=None, target="log", stats=None):
    """OASIS with lazy selection: every query computes only its context bucket; a bucket reuses its cached greedy
    decision for `expire` queries of its type. Every r-th query of a type is a feedback query: it computes the full
    features, selects by the lower confidence bound (exploration happens only here), is timed, and updates the model."""
    X = X.copy()
    if mask is not None:
        X[:, mask] = 0.0
    C = np.log(T) if target == "log" else T / 1000.0
    d = X.shape[1]
    Ai = np.tile(np.eye(d) / lam, (3, M, 1, 1)); b = np.zeros((3, M, d)); th = np.zeros((3, M, d))
    seen = np.zeros(3, int); win = [[], [], []]; pick = np.zeros(len(X), int)
    cache = {}; full = np.zeros(len(X), bool)

    def upd(t, a, x, c, s):
        u = Ai[t, a] @ x
        Ai[t, a] -= s * np.outer(u, u) / (1 + s * x @ u)
        b[t, a] += s * c * x; th[t, a] = Ai[t, a] @ b[t, a]

    for q in range(len(X)):
        t, x = types[q], X[q]
        seen[t] += 1; n = seen[t]
        if n <= M:                                    # cold start: each strategy once
            a = n - 1; fb = True
        elif n % r == 0:                              # feedback query: explore by the lower confidence bound
            w = np.sqrt(np.maximum(0, np.einsum("i,aij,j->a", x, Ai[t], x)))
            a = int(np.argmin(th[t] @ x - alpha * w)); fb = True
        else:                                         # other queries: cached greedy decision of the bucket
            k = (t, bucket(x)); c = cache.get(k)
            if c is not None and n - c[0] < expire:
                a = c[1]
            else:
                a = int(np.argmin(th[t] @ x)); cache[k] = (n, a); full[q] = True
            fb = False
        pick[q] = a
        if fb:
            full[q] = True
            upd(t, a, x, C[q, a], +1)
            if window > 0:
                win[t].append((q, a))
                if len(win[t]) > window:
                    r0, a2 = win[t].pop(0); upd(t, a2, X[r0], C[r0, a2], -1)
    if stats is not None:
        stats["full_frac"] = float(full.mean())
    return pick


NS = 6   # size buckets per query type


def keys_of(X, types, est=None):
    sb = np.minimum(3, (X[:, 1] * 4).astype(int))
    if est is not None:   # radius and range queries: size = decade of the estimated number of answers
        sb = np.where(types == 0, sb, np.minimum(NS - 1, np.floor(np.log10(1 + est)).astype(int)))
    db = np.minimum(3, (X[:, 5] * 4).astype(int)); of = (X[:, 7] > 0).astype(int)
    return types * NS + sb, ((types * NS + sb) * 4 + db) * 2 + of, types


REBUILD = 1000


def est_answers(df, stream):
    """Estimated number of answers of each radius/range query from a summed-area table of the count grid.
    The grid counts the stored points; insertions are replayed in order and the table is rebuilt every REBUILD insertions."""
    import struct as _st
    pts_f = W / "points.bin"
    with open(pts_f, "rb") as f:
        n, d = _st.unpack("<qi", f.read(12)); PTS = np.frombuffer(f.read(), dtype=np.float64).reshape(n, d)
    EV = np.dtype([("kind", "<i4"), ("qtype", "<i4"), ("pid", "<i8"), ("z", "<f8", (3,)), ("param", "<f8"), ("half", "<f8", (3,))])
    with open(W / f"{stream}.ev", "rb") as f:
        ne, n0, nuse, dd = _st.unpack("<qqqi", f.read(28)); ev = np.frombuffer(f.read(), dtype=EV)
    g = 64 if d == 2 else 24
    lo, hi = PTS.min(0), PTS.max(0) + 1e-12
    w = (hi - lo) / g
    cell = np.minimum(g - 1, ((PTS - lo) / w).astype(int))
    flat = np.ravel_multi_index(tuple(cell.T), (g,) * d)
    counts = np.bincount(flat[:n0], minlength=g ** d).astype(float)
    qev = ev[ev["kind"] == 1]
    ins = np.cumsum(ev["kind"] == 0)[ev["kind"] == 1]           # insertions applied before each query
    est = np.zeros(len(qev)); done = 0; sat = None
    for i, e in enumerate(qev):
        if ins[i] - done >= REBUILD or sat is None:      # the table is rebuilt after every REBUILD insertions
            if ins[i] != done:
                np.add.at(counts, flat[n0 + done:n0 + ins[i]], 1); done = ins[i]
            sat = counts.reshape((g,) * d)
            for ax in range(d):
                sat = sat.cumsum(ax)
            sat = np.pad(sat, [(1, 0)] * d)
        if e["qtype"] == 0:
            est[i] = e["param"]; continue
        half = np.full(d, e["param"]) if e["qtype"] == 1 else e["half"][:d]
        a = np.clip(((e["z"][:d] - half - lo) / w), 0, g); b = np.clip(((e["z"][:d] + half - lo) / w), 0, g)
        ai, bi = np.floor(a).astype(int), np.ceil(b).astype(int)
        tot = 0.0
        for corner in range(2 ** d):                                # inclusion-exclusion over the box corners
            idx, sgn = [], 1
            for ax in range(d):
                if corner >> ax & 1:
                    idx.append(bi[ax])
                else:
                    idx.append(ai[ax]); sgn = -sgn
            tot += sgn * sat[tuple(idx)]
        vol_cells = np.prod(np.maximum(bi - ai, 1)); frac = np.prod(np.maximum(b - a, 1e-9)) / vol_cells
        est[i] = tot * min(1.0, frac) * ((np.pi / 4 if d == 2 else np.pi / 6) if e["qtype"] == 1 else 1.0)
    assert len(est) == len(df)
    return est


def online_bucket(X, T, types, alpha=0.5, r=8, expire=256, elim=4.0, window=2000, stats=None, use_density=True,
                  use_location=True, nmin=5, margin=0.1, bound=2.0, est=None):
    """OASIS with bucketed lower confidence bounds.
    Coarse bucket = (type, size); fine bucket = (type, size, density, outside). Each bucket keeps the count and the sum of
    the observed log search times of every strategy. Every query only looks up its fine bucket and reuses the cached
    decision (greedy on the bucket means, falling back to the coarse bucket); every r-th query of a type is a feedback
    query: it tries untried strategies of its coarse bucket first, otherwise explores by the lower confidence bound, is
    timed, and updates both buckets. A strategy that is `elim` times slower than the best one seen in a coarse bucket is
    eliminated there and in the larger size buckets of the same type."""
    C = T.copy()          # arithmetic mean search time (us): the objective is the mean total query time
    nC, nF = 3 * NS, 3 * NS * 4 * 2
    cc = np.zeros((nC, M)); sc = np.zeros((nC, M)); cf = np.zeros((nF, M)); sf = np.zeros((nF, M))
    dead = np.zeros((nC, M), bool)
    kc, kf, _ = keys_of(X, types, est)
    if not use_density:
        kf = (kf // 8) * 8 + (kf % 2)
    if not use_location:
        kf = kf - (kf % 2)
    seen = np.zeros(3, int); cache = {}; win = [[], [], []]; pick = np.zeros(len(X), int); full = np.zeros(len(X), bool)
    extra = np.zeros(len(X))
    for q in range(len(X)):
        t, c0, f0 = types[q], kc[q], kf[q]
        seen[t] += 1; n = seen[t]
        live = ~dead[c0]
        if not live.any():
            live = np.ones(M, bool)
        if n % r == 0 or n <= M:
            means = np.where(live & (cc[c0] > 0), sc[c0] / np.maximum(cc[c0], 1), np.inf)
            wa = int(np.argmin(means)) if np.isfinite(means).any() else -1     # current winner of the coarse bucket
            untried = np.flatnonzero(live & (cc[c0] == 0))
            if len(untried):
                a = int(untried[0])
            else:   # exploration at the coarse level only
                score = np.where(np.isfinite(means), means * (1 - alpha / np.sqrt(np.maximum(cc[c0], 1))), np.inf)
                a = int(np.argmin(score))
            full[q] = True
            obs = []
            if wa >= 0 and a != wa and T[q, a] > bound * means[wa]:
                # bounded exploration: the probe stops at its budget and the query is re-run by the current winner
                extra[q] = bound * means[wa] + T[q, wa] - T[q, a]
                if (~dead[c0]).sum() > 2:
                    dead[c0, a] = True
                obs.append((wa, C[q, wa]))
            else:
                obs.append((a, C[q, a]))
            for a1, v in obs:
                cc[c0, a1] += 1; sc[c0, a1] += v; cf[f0, a1] += 1; sf[f0, a1] += v
                if window > 0:
                    win[t].append((c0, f0, a1, v))
                    if len(win[t]) > window:
                        c1, f1, a2, v2 = win[t].pop(0)
                        cc[c1, a2] -= 1; sc[c1, a2] -= v2; cf[f1, a2] -= 1; sf[f1, a2] -= v2
            ms = np.where(cc[c0] > 0, sc[c0] / np.maximum(cc[c0], 1), np.inf)
            if cc[c0, a] >= 3 and ms[a] > ms.min() * elim and (~dead[c0]).sum() > 2:
                dead[c0, a] = True                             # eliminated in this coarse bucket only
        else:
            hit = cache.get(f0)
            if hit is not None and n - hit[0] < expire and live[hit[1]]:
                a = hit[1]
            else:   # conservative greedy: the coarse winner, unless the fine bucket shows a clearly faster strategy
                mc = np.where(live & (cc[c0] > 0), sc[c0] / np.maximum(cc[c0], 1), np.inf)
                a = int(np.argmin(mc)) if np.isfinite(mc).any() else int(np.flatnonzero(live)[0])
                if cf[f0, a] >= nmin:
                    mf = np.where(live & (cf[f0] >= nmin), sf[f0] / np.maximum(cf[f0], 1), np.inf)
                    b2 = int(np.argmin(mf))
                    if mf[b2] < mf[a] * (1 - margin):
                        a = b2
                cache[f0] = (n, a); full[q] = True
        pick[q] = a
    if stats is not None:
        stats["full_frac"] = float(full.mean())
        stats["extra"] = extra
    return pick


def oasis_run(X, T, types, **kw):
    """Bucketed OASIS; returns (picks, extra time of aborted probes and their re-runs)."""
    st = {}
    p = online_bucket(X, T, types, stats=st, **kw)
    return p, st["extra"]


def bucket_cost_us(df, T, alpha, r, expire, elim, window, tag, est=None):
    X, ty = f_oasis(df), df.qtype.values.astype(float)[:, None]
    Z = np.c_[df[["z0", "z1", "z2"]].values, df.param.values, df.estans.values if est is None else est]
    fx, ft, fy, fz = TMP / f"x_{tag}.bin", TMP / f"t_{tag}.bin", TMP / f"y_{tag}.bin", TMP / f"z_{tag}.bin"
    wmat(fx, X); wmat(ft, T * 1000.0); wmat(fy, ty); wmat(fz, Z)
    return sb("bucket", fx, ft, fy, fz, alpha, r, expire, elim, window)


def lazy_cost_us(df, T, alpha, lam, window, r, expire, tag):
    X, ty = f_oasis(df), df.qtype.values.astype(float)[:, None]
    Z = np.c_[df[["z0", "z1", "z2"]].values, df.param.values]
    fx, ft, fy, fz = TMP / f"x_{tag}.bin", TMP / f"t_{tag}.bin", TMP / f"y_{tag}.bin", TMP / f"z_{tag}.bin"
    wmat(fx, X); wmat(ft, T * 1000.0); wmat(fy, ty); wmat(fz, Z)
    return sb("lazy", fx, ft, fy, fz, alpha, lam, window, r, expire)


# ---------------------------------------------------------------- offline selectors
def label(T):
    return T.argmin(1)


class Offline:
    def __init__(self, kind, feat, seed=0):
        self.kind, self.feat, self.seed = kind, feat, seed

    def fit(self, df, T):
        X, y = self.feat(df), label(T)
        t0 = time.perf_counter()
        if self.kind in ("rf", "dt", "knn", "mlp"):
            self.m = {"rf": RandomForestClassifier(n_estimators=50, max_depth=12, random_state=self.seed, n_jobs=1),
                      "dt": DecisionTreeClassifier(max_depth=10, random_state=self.seed),
                      "knn": KNeighborsClassifier(n_neighbors=5),
                      "mlp": MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=self.seed)}[self.kind]
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
            self.m.fit((X - self.mu) / self.sd if self.kind in ("knn", "mlp") else X, y)
        else:  # per-strategy regressors of log search time
            Y = np.log(T)
            if self.kind == "gbr":
                self.m = [GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=self.seed).fit(X, Y[:, a]) for a in range(M)]
            else:
                self.m = [Ridge(alpha=1.0).fit(X, Y[:, a]) for a in range(M)]
        self.fit_s = time.perf_counter() - t0
        return self

    def predict(self, df):
        X = self.feat(df)
        if self.kind in ("knn", "mlp"):
            X = (X - self.mu) / self.sd
        if self.kind in ("rf", "dt", "knn", "mlp"):
            proba = np.zeros((len(X), M)); proba[:, self.m.classes_] = self.m.predict_proba(X)
            return proba.argmax(1)
        return np.column_stack([m.predict(X) for m in self.m]).argmin(1)

    def cost_us(self, df, tag):
        """Selection cost measured in C++ on this stream's features."""
        X = self.feat(df)
        fx = TMP / f"feat_{tag}.bin"
        if self.kind in ("knn", "mlp"):
            X = (X - self.mu) / self.sd
        wmat(fx, X)
        mf = TMP / f"model_{tag}.txt"
        if self.kind in ("rf", "dt"):
            ests = self.m.estimators_ if self.kind == "rf" else [self.m]
            with open(mf, "w") as f:
                f.write(f"class 1 {len(ests)} {M} 1 0\n")
                for e in ests:
                    t = e.tree_; v = np.zeros((t.node_count, M)); v[:, self.m.classes_] = t.value[:, 0, :] / t.value[:, 0, :].sum(1, keepdims=True)
                    f.write(f"{t.node_count}\n")
                    for i in range(t.node_count):
                        f.write(f"{t.children_left[i]} {t.children_right[i]} {max(t.feature[i], 0)} {t.threshold[i]:.17g} " + " ".join(f"{z:.6g}" for z in v[i]) + "\n")
            return sb("trees", mf, fx)
        if self.kind == "gbr":
            with open(mf, "w") as f:
                f.write(f"reg {M} 50 1 0.1 0\n")
                for g in self.m:
                    # each arm has its own initial value: fold init/lr into every node of its first tree
                    init = float(np.ravel(g.init_.predict(np.zeros((1, X.shape[1]))))[0])
                    for j, est in enumerate(g.estimators_[:, 0]):
                        t = est.tree_; add = init / g.learning_rate if j == 0 else 0.0
                        f.write(f"{t.node_count}\n")
                        for i in range(t.node_count):
                            f.write(f"{t.children_left[i]} {t.children_right[i]} {max(t.feature[i], 0)} {t.threshold[i]:.17g} {t.value[i, 0, 0] + add:.10g}\n")
            return sb("trees", mf, fx)
        if self.kind == "knn":
            tr = TMP / f"knn_{tag}.bin"
            wmat(tr, np.c_[self.m._fit_X, self.m._y]); return sb("knn", tr, fx, 5)
        if self.kind == "mlp":
            W1, W2 = self.m.coefs_; b1, b2 = self.m.intercepts_
            with open(mf, "w") as f:
                f.write(f"{W1.shape[0]} {W1.shape[1]} {W2.shape[1]}\n")
                for arr in (W1.ravel(), b1, W2.ravel(), b2):
                    f.write(" ".join(f"{z:.10g}" for z in arr) + "\n")
            return sb("mlp", mf, fx)
        mf = TMP / f"lin_{tag}.bin"
        wmat(mf, np.c_[np.array([m.coef_ for m in self.m]), [m.intercept_ for m in self.m]])
        return sb("linear", mf, fx)


def online_cost_us(df, T, rule, alpha, lam, window, tag, target="raw", feats=None):
    X, ty = (f_oasis(df) if feats is None else feats), df.qtype.values.astype(float)[:, None]
    fx, ft, fy = TMP / f"x_{tag}.bin", TMP / f"t_{tag}.bin", TMP / f"y_{tag}.bin"
    wmat(fx, X); wmat(ft, T * 1000.0); wmat(fy, ty)
    return sb("online", rule, fx, ft, fy, alpha, lam, window, target)


# ---------------------------------------------------------------- simple baselines
def random_pick(n, seed=0):
    return np.random.default_rng(seed).integers(M, size=n)


def frozen_pick(T, types, per=10):
    pick = np.zeros(len(T), int); seen = np.zeros(3, int); best = [None] * 3; sums = np.zeros((3, M)); cnts = np.zeros((3, M))
    for q in range(len(T)):
        t = types[q]
        if best[t] is None:
            a = seen[t] % M; seen[t] += 1; sums[t, a] += T[q, a]; cnts[t, a] += 1
            if seen[t] >= per * M:
                best[t] = int(np.argmin(sums[t] / cnts[t]))
        else:
            a = best[t]
        pick[q] = a
    return pick


def best_fixed(T, types):
    pick = np.zeros(len(T), int)
    for t in range(3):
        m = types == t
        if m.any():
            pick[m] = T[m].sum(0).argmin()
    return pick


UNISR_LOG = []


def unis_r(df, T, base, every=.2, recent=2000):
    """UnIS retrained every 20% of the stream on the most recent queries, labelled by running all strategies."""
    pick = base.predict(df); extra = np.zeros(len(T)); n = len(T)
    for k in range(1, int(1 / every)):
        b = int(k * every * n); lo = max(0, b - recent)
        m = Offline("rf", f_unis, seed=k).fit(df.iloc[lo:b], T[lo:b])
        pick[b:] = m.predict(df.iloc[b:])
        # relabel: every strategy runs on the recent queries, cut off at twice the fastest one (generous to UnIS-R);
        # retrain: measured fitting time. Both in microseconds, charged at the retraining point.
        relabel = np.minimum(T[lo:b], 2.0 * T[lo:b].min(1, keepdims=True)).sum()
        extra[b] += relabel + m.fit_s * 1e6
        UNISR_LOG.append((b, relabel, m.fit_s * 1e6))
    return pick, extra


def gather(T, pick, over, extra=None):
    tot = T[np.arange(len(T)), pick] + over
    return tot if extra is None else tot + extra


def acc(T, pick):
    return (T[np.arange(len(T)), pick] <= ACC_TOL * T.min(1)).astype(float)


# ---------------------------------------------------------------- OASIS (final rule)
def load_est(df, stream):
    f = W / f"est_{stream}.npy"
    return np.load(f) if f.exists() else est_answers(df, stream)


def ctx_size(df, est, use_est=True):
    """Context of a query = (type, size class). kNN: decade of k. Radius/range: decade of the estimated number of
    answers (summed-area table), or, without the estimate, four classes of the query extent."""
    ty = df.qtype.values
    ksz = np.round(np.log10(df.param.values.clip(1))).astype(int)
    if use_est:
        osz = np.minimum(5, np.floor(np.log10(1 + est))).astype(int)
    else:
        osz = np.minimum(3, (df.ell.values * 4).astype(int))
    return ty, np.where(ty == 0, ksz, osz)


def oasis_ctx(T, ty, size, alpha=0.25, bound=2.0, pace=4, window=500, revive=1000, warm=True):
    """Per-context bandit over the strategies. Every query is timed and updates the mean search time of the strategy
    that answered it. The incumbent of a context is its fastest live strategy; a different strategy runs only as a
    probe: an untried one (at most one probe per `pace` visits of the context) or one whose lower confidence bound is
    below the incumbent. A probe is aborted when it exceeds `bound` times the incumbent mean, the query is re-run by
    the incumbent, and the probed strategy is suspended in that context for `revive` visits. A new context starts
    from the ranking of the nearest visited context of the same type. Returns picks and search time incl. aborts."""
    n = len(T); NK = 3 * 8
    kid = ty * 8 + size
    cnt = np.zeros((NK, M)); sm = np.zeros((NK, M)); dead = np.full((NK, M), -10 ** 9)
    visits = np.zeros(NK, int); probes = np.zeros(NK, int); order = {}
    hist = [[] for _ in range(NK)]
    cost = np.zeros(n); pick = np.zeros(n, int)
    for q in range(n):
        k, t = kid[q], ty[q]; visits[k] += 1; v = visits[k]
        if k not in order:
            near = [j for j in order if j // 8 == t] if warm else []
            if near:
                j = min(near, key=lambda j: (abs(j - k), j > k))
                mu_j = np.where(cnt[j] > 0, sm[j] / np.maximum(cnt[j], 1), np.inf)
                order[k] = [int(a) for a in np.argsort(mu_j, kind="stable")]
            else:
                order[k] = list(range(M))
        live = v - dead[k] > revive
        tried = cnt[k] > 0
        mu = np.where(tried, sm[k] / np.maximum(cnt[k], 1), np.inf)
        inc = int(np.argmin(np.where(live, mu, np.inf))) if (tried & live).any() else -1
        untried = [a for a in order[k] if live[a] and not tried[a]]
        may_probe = v >= pace * (probes[k] + 1)
        if inc < 0:
            a = untried[0] if untried else order[k][0]
        elif untried and may_probe:
            a = untried[0]
        else:
            lcb = np.where(live & tried, mu * (1 - alpha / np.sqrt(np.maximum(cnt[k], 1))), np.inf)
            a = int(np.argmin(lcb))
            if a != inc and not may_probe:
                a = inc
        obs = a; c = T[q, a]
        if inc >= 0 and a != inc:
            probes[k] += 1
            if T[q, a] > bound * mu[inc]:
                c = bound * mu[inc] + T[q, inc]; obs = inc; dead[k, a] = v
        cost[q] = c; pick[q] = a
        cnt[k, obs] += 1; sm[k, obs] += T[q, obs]
        if window:
            hist[k].append((obs, T[q, obs]))
            if len(hist[k]) > window:
                a2, v2 = hist[k].pop(0); cnt[k, a2] -= 1; sm[k, a2] -= v2
    return pick, cost


def ctx_cost_us(df, T, est, alpha, bound, pace, window, revive, tag):
    """Per-query cost of the OASIS rule measured in C++: two clock reads, the answer estimate (summed-area table),
    the context, the choice and the statistics update."""
    ty = df.qtype.values.astype(float)[:, None]
    Z = np.c_[df[["z0", "z1", "z2"]].values, df.param.values, df.ell.values]
    ft, fy, fz = TMP / f"t_{tag}.bin", TMP / f"y_{tag}.bin", TMP / f"z_{tag}.bin"
    wmat(ft, T * 1000.0); wmat(fy, ty); wmat(fz, Z)
    dim = 3 if df.z2.abs().max() > 0 else 2
    return sb("ctx", ft, fy, fz, alpha, bound, pace, window, revive, dim)


def f_lin(df):
    """Common features of the other selectors: OASIS's information (type via one model per type, log answers) plus
    the extent, the distance to the principal axis, the local density and the outside distance."""
    L = np.log10(1 + df.est.values) / 4
    return np.c_[np.ones(len(df)), L, L * L, df.ell.values, df.x3.values, df.x5.values, L * df.x5.values, df.x7.values]
