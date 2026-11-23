"""Generate event streams (insertions + kNN/radius/range queries) for one dataset.

Run: python gen.py <project_dir> <dataset> <n_use> [--scal]
Radius and range queries are sized so that the number of points they hold is about a target of 10, 100, 1000 or
10000 (the same four decades and weights as k of kNN queries), using a summed-area table of the current points.
Writes work/<dataset>_<n>/<workload>.ev  with header (int64 n_events, int64 n0, int64 n_use, int32 dim)
followed by 72-byte events (kind, qtype, pid, z[3], param, half[3]).
kind 0 = insert point pid; kind 1 = query. qtype 0 = kNN (param = k), 1 = radius (param = r), 2 = range (half sides).
"""
import os
import struct
import sys
from pathlib import Path

import numpy as np

EV = np.dtype([("kind", "<i4"), ("qtype", "<i4"), ("pid", "<i8"), ("z", "<f8", (3,)), ("param", "<f8"), ("half", "<f8", (3,))])
assert EV.itemsize == 72
KS, KW = [1, 10, 100, 1000], [.3, .3, .25, .15]
TS, TW = [10, 100, 1000, 10000], [.3, .3, .25, .15]   # answer targets of radius and range queries


def load(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<qi", f.read(12))
        return np.frombuffer(f.read(), dtype=np.float64).reshape(n, d)


class Density:
    """Count grid of the points currently in the index with a summed-area table. Counts inside a box are read by
    inclusion-exclusion over the box corners, with multilinear interpolation inside a cell (uniform density per cell)."""

    def __init__(self, pts):
        self.d = d = pts.shape[1]
        self.g = 256 if d == 2 else 64
        self.lo, self.hi = pts.min(0), pts.max(0) + 1e-12
        h, _ = np.histogramdd(pts, bins=self.g, range=list(zip(self.lo, self.hi)))
        S = h
        for ax in range(d):
            S = S.cumsum(ax)
        self.S = np.pad(S, [(1, 0)] * d)
        self.w = (self.hi - self.lo) / self.g
        bits = np.array([[(c >> ax) & 1 for ax in range(d)] for c in range(2 ** d)])
        self.bits = bits
        self.sign = (-1.0) ** (d - bits.sum(1))

    def cum(self, P):
        """Number of points below each row of P (cumulative count, interpolated)."""
        u = np.clip((P - self.lo) / self.w, 0, self.g)
        i0 = np.minimum(np.floor(u).astype(int), self.g - 1)
        f = u - i0
        idx = i0[:, None, :] + self.bits[None, :, :]                       # rows x sub-corners x d
        wgt = np.prod(np.where(self.bits[None, :, :] == 1, f[:, None, :], 1 - f[:, None, :]), axis=2)
        return (self.S[tuple(idx[..., ax] for ax in range(self.d))] * wgt).sum(1)

    def count_box(self, z, half):
        P = np.where(self.bits == 1, z + half, z - half)
        return float(self.sign @ self.cum(P))

    def extent_for(self, z, target, shape, ball):
        """Smallest scale s such that the box z +- s*shape (or the ball of radius s) holds about `target` points."""
        f = (np.pi / 4 if self.d == 2 else np.pi / 6) if ball else 1.0
        lo, hi = np.log(1e-5), np.log(0.3)
        if f * self.count_box(z, np.exp(hi) * shape) < target:
            return np.exp(hi)
        for _ in range(30):
            mid = (lo + hi) / 2
            if f * self.count_box(z, np.exp(mid) * shape) < target:
                lo = mid
            else:
                hi = mid
        return np.exp(hi)


class Gen:
    def __init__(self, pts, rng):
        self.rng, self.d = rng, pts.shape[1]
        self.set_points(pts)
        self.hot = pts[rng.integers(len(pts), size=5)]
        self.zipf_cells = None

    def set_points(self, pts):
        self.pts = pts
        self.lo, self.hi = pts.min(0), pts.max(0)
        self.den = Density(pts)

    # ---- locations
    def loc(self, kind, t=0):
        r, d = self.rng, self.d
        if kind == "data":
            return self.pts[r.integers(len(self.pts))] + r.normal(0, .002, d)
        if kind == "uniform":
            return r.uniform(self.lo, self.hi)
        if kind == "outside":
            span = self.hi - self.lo
            while True:
                z = r.uniform(self.lo - .25 * span, self.hi + .25 * span)
                if ((z < self.lo) | (z > self.hi)).any():
                    return z
        if kind == "hotspot":
            return self.hot[r.integers(len(self.hot))] + r.normal(0, .01, d)
        if kind == "zipf":
            g = 32 if d == 2 else 12
            if self.zipf_cells is None:
                ncell = g ** d
                p = 1.0 / np.arange(1, ncell + 1) ** 1.2
                self.zipf_p = p / p.sum()
                self.zipf_cells = r.permutation(ncell)
            c = self.zipf_cells[r.choice(len(self.zipf_cells), p=self.zipf_p)]
            idx = np.array(np.unravel_index(c, (g,) * d))
            return self.lo + (idx + r.random(d)) / g * (self.hi - self.lo)
        if kind == "trajectory":
            stride = max(1, len(self.pts) // 20000)
            return self.pts[(t * stride) % len(self.pts)] + r.normal(0, .0005, d)
        raise ValueError(kind)

    # ---- one query event
    def query(self, z, qtype=None, big=None, aspect=None):
        r, d = self.rng, self.d
        e = np.zeros((), dtype=EV)
        e["kind"] = 1
        qt = r.integers(3) if qtype is None else qtype
        e["qtype"] = qt
        e["z"][:d] = z
        if qt == 0:
            if big is None:
                k = r.choice(KS, p=KW)
            else:
                k = r.choice(KS[2:] if big else KS[:2])
            e["param"] = k
        else:
            T = r.choice(TS, p=TW) if big is None else r.choice(TS[2:] if big else TS[:2])
            if qt == 1:
                e["param"] = self.den.extent_for(z[:d], T, np.ones(d), ball=True)
            else:
                a = (r.choice([1, 8]) if aspect is None else aspect)
                shape = np.ones(d)
                ax = r.integers(d)
                shape[ax] *= a ** ((d - 1) / d)
                shape[[i for i in range(d) if i != ax]] /= a ** (1 / d)
                e["half"][:d] = self.den.extent_for(z[:d], T, shape, ball=False) * shape
                e["param"] = 0
        return e


def write(path, events, n0, n_use, d):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(events, dtype=EV)
    with open(path, "wb") as f:
        f.write(struct.pack("<qqqi", len(arr), n0, n_use, d))
        f.write(arr.tobytes())
    print(path.name, len(arr), "events", flush=True)


def mixed_loc(g, rng):
    u = rng.random()
    return g.loc("data") if u < .5 else g.loc("uniform") if u < .7 else g.loc("hotspot") if u < .9 else g.loc("outside")


def static_stream(g, rng, Q, locfn):
    return [g.query(locfn(t)) for t in range(Q)]


def main():
    P, name, n_use = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    scal = "--scal" in sys.argv
    full = load(P / "data" / f"{name}.bin")
    rng = np.random.default_rng(2026)
    n_use = min(n_use, len(full))
    idx = np.sort(rng.choice(len(full), n_use, replace=False)) if n_use < len(full) else np.arange(len(full))
    pts = np.ascontiguousarray(full[idx])
    d = pts.shape[1]
    W = P / os.environ.get("OASIS_WORK", "work8") / f"{name}_{n_use}"
    W.mkdir(parents=True, exist_ok=True)
    with open(W / "points.bin", "wb") as f:
        f.write(struct.pack("<qi", n_use, d))
        f.write(pts.tobytes())

    g = Gen(pts, np.random.default_rng(11))
    write(W / "mixed.ev", static_stream(g, g.rng, 5000 if scal else 20000, lambda t: mixed_loc(g, g.rng)), n_use, n_use, d)
    if scal:
        return
    g = Gen(pts, np.random.default_rng(12))
    write(W / "valid.ev", static_stream(g, g.rng, 10000, lambda t: mixed_loc(g, g.rng)), n_use, n_use, d)
    g = Gen(pts, np.random.default_rng(13))
    write(W / "train_full.ev", static_stream(g, g.rng, 20000, lambda t: g.loc("data")), n_use, n_use, d)
    for i, kind in enumerate(["uniform", "data", "hotspot", "zipf", "trajectory"]):
        g = Gen(pts, np.random.default_rng(20 + i))
        write(W / f"dist_{kind}.ev", static_stream(g, g.rng, 10000, lambda t, k=kind: g.loc(k, t)), n_use, n_use, d)

    # drift: four phases whose queries favour different strategies
    g = Gen(pts, np.random.default_rng(30))
    ev, Q = [], 20000
    for t in range(Q):
        ph = 4 * t // Q
        if ph == 0:
            ev.append(g.query(g.loc("data"), qtype=0 if g.rng.random() < .7 else None, big=False))
        elif ph == 1:
            ev.append(g.query(g.loc("hotspot"), qtype=1 if g.rng.random() < .7 else None, big=True))
        elif ph == 2:
            ev.append(g.query(g.loc("uniform") if g.rng.random() < .6 else g.loc("outside"),
                              qtype=2 if g.rng.random() < .7 else None, aspect=8))
        else:
            ev.append(g.query(g.loc("zipf"), qtype=0 if g.rng.random() < .7 else None, big=True))
    write(W / "drift.ev", ev, n_use, n_use, d)

    # insertion: start with half of the points, insert the rest in order between queries
    n0 = n_use // 2
    g = Gen(pts[:n0], np.random.default_rng(40))
    write(W / "train_half.ev", static_stream(g, g.rng, 20000, lambda t: g.loc("data")), n0, n_use, d)
    ev, Q = [], 20000
    per = (n_use - n0) / Q
    nxt = n0
    for t in range(Q):
        tgt = int(n0 + per * (t + 1))
        while nxt < tgt:
            e = np.zeros((), dtype=EV); e["kind"] = 0; e["pid"] = nxt
            ev.append(e); nxt += 1
        if t % 1000 == 0 and t > 0:
            g.set_points(pts[:nxt])
        recent = pts[max(n0, nxt - 20000):nxt] if nxt > n0 else pts[:n0]
        z = recent[g.rng.integers(len(recent))] + g.rng.normal(0, .002, d) if g.rng.random() < .7 else g.loc("data")
        ev.append(g.query(z))
    write(W / "insert.ev", ev, n0, n_use, d)


if __name__ == "__main__":
    main()
