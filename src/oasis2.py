"""OASIS as described in Section 4 of the 2026-11-13 revision (two-level feature extraction, limited trials with a
temperature per candidate, bounded trials, and recency weighting with decay inside a window).

The replay uses the search times of every plan on every query that the bench measured (see replay_core.py).
`oasis2` returns, per query, the executed plan, the extra time of a stopped trial (limit + re-run by the best known
plan - search time of the executed plan), and the best known plan a* (the prediction of OASIS).

Switches for the incremental ablation:
  features: two-level feature extraction (classes by our distance + cell experience); off = one class per query type
  limited:  limited trials (candidates, a temperature per candidate, time limit); off = one temperature for all plans,
            no candidates, no time limit
  recency:  window + decay; off = keep every observation with the same weight
"""
import math
import struct as _st
from collections import deque

import numpy as np

EV = np.dtype([("kind", "<i4"), ("qtype", "<i4"), ("pid", "<i8"), ("z", "<f8", (3,)), ("param", "<f8"), ("half", "<f8", (3,))])

DEFAULT = dict(tau=2.0, K=32, r=3, N0=8.0, T0=0.1, gamma=1.0, b=2.0, lam=0.999, w=2000, cf=4, seed=0, cand="r", skip=False,
               reset_rho=0.0, beta=0.3, nmin=8, keep=4, center_eta=0.0, retire=0, temp="own",
               features=True, limited=True, recency=False)   # recency weighting dropped (2026-11-13): no effect


# ---------------------------------------------------------------- features of each query (Section 4.2)
def query_features(W, stream, df):
    """Basic quantities phi = (size, shape, density) and the cell of the query point of every query of `stream`.
    The table of point counts is replayed with the insertions of the stream (same grid as the answer estimate)."""
    with open(W / "points.bin", "rb") as f:
        n, d = _st.unpack("<qi", f.read(12)); pts = np.frombuffer(f.read(), dtype=np.float64).reshape(n, d)
    with open(W / f"{stream}.ev", "rb") as f:
        ne, n0, nuse, dd = _st.unpack("<qqqi", f.read(28)); ev = np.frombuffer(f.read(), dtype=EV)
    g = 64 if d == 2 else 24
    lo, hi = pts.min(0), pts.max(0) + 1e-12
    w = (hi - lo) / g
    cell_pts = np.minimum(g - 1, ((pts - lo) / w).astype(int))
    flat_pts = np.ravel_multi_index(tuple(cell_pts.T), (g,) * d)
    counts = np.bincount(flat_pts[:n0], minlength=g ** d).astype(np.int64)
    isq = ev["kind"] == 1
    qev = ev[isq]
    ins = np.cumsum(ev["kind"] == 0)[isq]
    assert len(qev) == len(df), (len(qev), len(df))
    z = qev["z"][:, :d]
    cz = np.clip(((z - lo) / w).astype(int), 0, g - 1)
    flat_q = np.ravel_multi_index(tuple(cz.T), (g,) * d)
    ccount = np.zeros(len(qev)); nins = np.zeros(len(qev)); done = 0
    for i in range(len(qev)):                       # counts change only with insertions
        if ins[i] != done:
            np.add.at(counts, flat_pts[n0 + done:n0 + ins[i]], 1); done = ins[i]
        ccount[i] = counts[flat_q[i]]; nins[i] = n0 + done
    ty = qev["qtype"]
    est = np.where(ty == 0, qev["param"], df["est"].values if "est" in df else qev["param"])
    size = np.log10(1.0 + np.maximum(est, 0))
    half = qev["half"][:, :d]
    aspect = np.where(ty == 2, half.max(1) / np.maximum(half.min(1), 1e-300), 1.0)
    shape = np.log10(np.maximum(aspect, 1.0))
    dens = np.log10((1.0 + ccount) / (1.0 + nins / g ** d))
    phi = np.c_[size, shape, dens]
    z3 = np.zeros((len(qev), 3)); z3[:, :d] = z
    h3 = np.zeros((len(qev), 3)); h3[:, :d] = half
    cz3 = np.zeros((len(qev), 3), np.int64); cz3[:, :d] = cz
    return dict(ty=ty.astype(int), phi=phi, cell=flat_q.astype(np.int64), cz=cz3, z=z3, half=h3,
                param=qev["param"].astype(float), d=np.array(d), g=np.array(g))


def coarse_cells(F, cf):
    """Cells of a grid that is cf times coarser than the table of point counts in every dimension."""
    if cf <= 1:
        return F["cell"]
    d, g = int(F["d"]), int(F["g"]); gc = -(-g // cf)
    return np.ravel_multi_index(tuple((F["cz"][:, :d] // cf).T), (gc,) * d).astype(np.int64)


# ---------------------------------------------------------------- experience with decay inside a window (Section 4.5)
class Exp:
    """Weighted sums of one class: per plan (class experience) and per cell and plan (cell experience).
    Weights are lambda^age; sums are kept in scaled units lambda^-(tau-base), so an update is O(1)."""
    __slots__ = ("lam", "w", "M", "S", "Wt", "cnt", "cells", "win", "t", "base", "lim", "order", "fast")

    def __init__(self, M, lam, w):
        self.lam, self.w, self.M = lam, w, M
        self.lim = int(300 / -math.log(lam)) if lam < 1.0 else 0     # renormalise before lambda^-age overflows
        self.S = np.zeros(M); self.Wt = np.zeros(M); self.cnt = np.zeros(M, int)
        self.cells = {}               # cell -> [S(M), Wt(M), cnt(M)]
        self.win = deque(); self.t = 0; self.base = 0
        self.order = list(range(M))   # order in which untried plans are tried (warm start)
        self.fast = np.full(M, np.nan)  # short-term average of each plan (change detection)

    def watch(self, a, s, rho, beta, nmin, keep):
        """Change-triggered forgetting: if the short-term average of plan a departs from its mean by more than a factor
        rho (slower or faster), the old observations of a are dropped and only its last `keep` remain."""
        f = self.fast[a]
        self.fast[a] = s if np.isnan(f) else (1 - beta) * f + beta * s
        if self.cnt[a] < nmin:
            return False
        mu = self.S[a] / max(self.Wt[a], 1e-300)
        if self.fast[a] > rho * mu or self.fast[a] < mu / rho:
            self.reset_plan(a, keep)
            return True
        return False

    def reset_plan(self, a, keep):
        mine = [e for e in self.win if e[0] == a]
        drop = mine[:-keep] if keep > 0 else mine
        if not drop:
            return
        ids = {id(e) for e in drop}
        self.win = deque(e for e in self.win if id(e) not in ids)
        for (a2, c2, s2, tau) in drop:
            u2 = self._u(tau)
            self.S[a2] -= u2 * s2; self.Wt[a2] -= u2; self.cnt[a2] -= 1
            if c2 is not None:
                v = self.cells[c2]; v[0][a2] -= u2 * s2; v[1][a2] -= u2; v[2][a2] -= 1
                if v[2][a2] == 0:
                    v[0][a2] = 0.0; v[1][a2] = 0.0
        if self.cnt[a] == 0:
            self.S[a] = 0.0; self.Wt[a] = 0.0

    def _u(self, tau):
        return self.lam ** (self.base - tau) if self.lam < 1.0 else 1.0

    def add(self, a, cell, s):
        if self.lam < 1.0 and (self.t - self.base) > self.lim:  # renormalise the scaled sums
            f = self.lam ** (self.t - self.base)
            self.S *= f; self.Wt *= f
            for v in self.cells.values():
                v[0] *= f; v[1] *= f
            self.base = self.t
        u = self.lam ** (self.base - self.t) if self.lam < 1.0 else 1.0
        self.S[a] += u * s; self.Wt[a] += u; self.cnt[a] += 1
        if cell is not None:
            v = self.cells.get(cell)
            if v is None:
                v = self.cells[cell] = [np.zeros(self.M), np.zeros(self.M), np.zeros(self.M, int)]
            v[0][a] += u * s; v[1][a] += u; v[2][a] += 1
        self.win.append((a, cell, s, self.t)); self.t += 1
        if self.w and len(self.win) > self.w:
            a2, c2, s2, tau = self.win.popleft(); u2 = self._u(tau)
            self.S[a2] -= u2 * s2; self.Wt[a2] -= u2; self.cnt[a2] -= 1
            if self.cnt[a2] == 0:
                self.S[a2] = 0.0; self.Wt[a2] = 0.0
            if c2 is not None:
                v = self.cells[c2]; v[0][a2] -= u2 * s2; v[1][a2] -= u2; v[2][a2] -= 1
                if v[2][a2] == 0:
                    v[0][a2] = 0.0; v[1][a2] = 0.0

    def scale(self):                  # current weight of scaled unit 1
        return self.lam ** (self.t - 1 - self.base) if self.lam < 1.0 else 1.0

    def means(self):
        return np.where(self.cnt > 0, self.S / np.maximum(self.Wt, 1e-300), np.inf)

    def counts(self):
        return np.where(self.cnt > 0, self.Wt * self.scale(), 0.0)

    def cell(self, cell):
        v = self.cells.get(cell)
        if v is None:
            return np.full(self.M, np.inf), np.zeros(self.M)
        mu = np.where(v[2] > 0, v[0] / np.maximum(v[1], 1e-300), np.inf)
        return mu, np.where(v[2] > 0, v[1] * self.scale(), 0.0)


# ---------------------------------------------------------------- the selector (Sections 4.2-4.5)
def oasis2(T, ty, phi, cell, **kw):
    p = dict(DEFAULT); p.update(kw)
    n, M = T.shape
    rng = np.random.default_rng(p["seed"])
    lam = p["lam"] if p["recency"] else 1.0
    w = p["w"] if p["recency"] else 0
    centers = {0: [], 1: [], 2: []}   # per type: list of [center(3), count, class id]
    exps = []
    pick = np.zeros(n, int); extra = np.zeros(n); astar = np.full(n, -1)
    trials = np.zeros(n, bool); stopped = np.zeros(n, bool); kind = np.zeros(n, np.int8)
    tq = [0, 0, 0]                    # queries seen per type (clock of class recency)
    class_rec = p["recency"] and (p["center_eta"] > 0 or p["retire"] > 0)
    for q in range(n):
        t = int(ty[q]); x = phi[q]; tq[t] += 1
        # ---- common feature: the closest class of the same type (type penalty = +inf across types)
        cs = centers[t]
        if p["features"] and class_rec and p["retire"] > 0 and cs:   # retire classes that no recent query visited
            keep_cs = [c for c in cs if tq[t] - c[3] <= p["retire"]]
            if len(keep_cs) != len(cs):
                cs[:] = keep_cs
        if p["features"]:
            best, bd = -1, np.inf
            for j, c in enumerate(cs):
                dd = float(((x - c[0]) ** 2).sum())
                if dd < bd:
                    bd, best = dd, j
            if best < 0 or (bd > p["tau"] and len(cs) < p["K"]):
                e = Exp(M, lam, w)
                if best >= 0:                           # warm start: try the plans in the order of the closest class
                    mu0 = exps[cs[best][2]].means()
                    e.order = [int(a) for a in np.argsort(mu0, kind="stable")]
                    if p["skip"] and np.isfinite(mu0).any():   # borrow the mean of plans hopeless in the closest class
                        m0 = mu0[np.isfinite(mu0)].min()
                        for a in range(M):
                            if np.isfinite(mu0[a]) and mu0[a] > p["b"] * m0:
                                e.add(a, None, float(mu0[a]))
                exps.append(e); cs.append([x.copy(), 1, len(exps) - 1, tq[t]]); cid = len(exps) - 1
            else:
                c = cs[best]; c[1] += 1; c[3] = tq[t]
                step = max(1.0 / c[1], p["center_eta"]) if class_rec else 1.0 / c[1]   # centers follow recent queries
                c[0] += (x - c[0]) * step; cid = c[2]
            cl = int(cell[q])
        else:
            if not cs:
                exps.append(Exp(M, lam, w)); cs.append([x.copy(), 1, len(exps) - 1, tq[t]])
            cid = cs[0][2]; cl = None
        E = exps[cid]
        mu_c = E.means(); N_c = E.counts()
        if cl is not None:
            mu_i, N_i = E.cell(cl)
            mix = np.where(np.isfinite(mu_i), (N_i * np.where(np.isfinite(mu_i), mu_i, 0) + p["N0"] * mu_c) / (N_i + p["N0"]), mu_c)
        else:
            N_i = N_c; mix = mu_c
        tried = E.cnt > 0
        # ---- candidates and the best known plan
        if tried.any():
            order = np.argsort(np.where(tried, mu_c, np.inf), kind="stable")
            if p["limited"] and p["cand"] == "ratio":   # plans whose mean is within the time limit of the best mean
                m_best = mu_c[order[0]]
                cand = [int(a) for a in order if tried[a] and mu_c[a] <= p["b"] * m_best]
            else:
                cand = [int(a) for a in order[:(p["r"] if p["limited"] else M)] if tried[a]]
            a_star = min(cand, key=lambda a: mix[a])
        else:
            cand, a_star = [], -1
        untried = [a for a in E.order if not tried[a]]
        # ---- which plan to run
        if untried:
            a = untried[0]                               # a plan without experience is tried once
        elif len(cand) == 1:
            a = cand[0]
        else:
            ratio = np.array([mix[a_star] / mix[a] for a in cand])
            if p["limited"] and p["temp"] == "own":   # observations behind the combined mean: cell's own plus class's, up to N0
                Nk = np.array([N_i[a] + (min(p["N0"], N_c[a]) if cl is not None else 0.0) for a in cand])
                Ta = p["T0"] * (1 + p["gamma"] * np.exp(-Nk)) / (1 + np.log1p(Nk))
            else:
                Ta = np.full(len(cand), p["T0"] / (1 + math.log1p(N_c.sum())))
            wts = ratio ** (1.0 / np.maximum(Ta, 1e-12))
            a = cand[int(np.searchsorted(np.cumsum(wts), rng.random() * wts.sum()))] if wts.sum() > 0 else a_star
        astar[q] = a_star
        kind[q] = 2 if untried else (0 if a == a_star else 1)
        pick[q] = a; s = T[q, a]
        is_trial = a_star >= 0 and a != a_star
        trials[q] = is_trial
        if is_trial and p["limited"] and s > p["b"] * mix[a_star]:
            L = p["b"] * mix[a_star]
            extra[q] = L + T[q, a_star] - s; stopped[q] = True
            E.add(a, cl, L); E.add(a_star, cl, T[q, a_star])
            obs = [(a_star, T[q, a_star])]
        else:
            E.add(a, cl, s)
            obs = [(a, s)]
        if p["recency"] and p["reset_rho"] > 0:          # change-triggered forgetting
            for a2, s2 in obs:
                E.watch(a2, s2, p["reset_rho"], p["beta"], p["nmin"], p["keep"])
    info = dict(classes=len(exps), cells=sum(len(e.cells) for e in exps), trials=float(trials.mean()),
                stopped=float(stopped.mean()), kind=kind)
    return pick, extra, astar, info
