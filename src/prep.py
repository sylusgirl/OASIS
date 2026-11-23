"""Prepare datasets: raw files -> data/<name>.bin (float64, insertion order, scaled into the unit box).

File format: int64 n, int32 dim, then n*dim float64.
Insertion order: Uber and Porto by time; Dragon in scan order; GeoNames shuffled with a fixed seed.
Run: python prep.py <project_dir>
"""
import io
import json
import struct
import sys
import zipfile
import gzip
from pathlib import Path

import numpy as np

P = Path(sys.argv[1])
RAW, DATA = P / "raw", P / "data"
DATA.mkdir(exist_ok=True)


def save(name, xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    lo, hi = xyz.min(0), xyz.max(0)
    xyz = (xyz - lo) / (hi - lo).max()          # one scale for all axes keeps the shape
    with open(DATA / f"{name}.bin", "wb") as f:
        f.write(struct.pack("<qi", xyz.shape[0], xyz.shape[1]))
        f.write(xyz.tobytes())
    print(name, xyz.shape, flush=True)


def uber():
    parts = []
    for m in ["apr", "may", "jun", "jul", "aug", "sep"]:
        txt = (RAW / f"uber-{m}.csv").read_text().splitlines()[1:]
        t, la, lo = [], [], []
        for line in txt:
            a = line.split(",")
            if len(a) < 3:
                continue
            d, h = a[0].strip('"').split(" ")
            mo, dd, yy = d.split("/")
            hh, mi, ss = h.split(":")
            t.append(((int(mo) * 31 + int(dd)) * 24 + int(hh)) * 3600 + int(mi) * 60 + int(ss))
            la.append(float(a[1])); lo.append(float(a[2]))
        parts.append((np.array(t), np.array(lo), np.array(la)))
    t = np.concatenate([p[0] for p in parts]); x = np.concatenate([p[1] for p in parts]); y = np.concatenate([p[2] for p in parts])
    o = np.argsort(t, kind="stable")
    keep = (x > -74.4) & (x < -73.6) & (y > 40.5) & (y < 41.0)   # New York City area
    o = o[keep[o]]
    save("Uber", np.c_[x[o], y[o]])


def geonames():
    with zipfile.ZipFile(RAW / "US.zip") as z:
        raw = z.read("US.txt").decode("utf-8")
    la, lo = [], []
    for line in raw.splitlines():
        a = line.split("\t")
        if len(a) > 5:
            la.append(float(a[4])); lo.append(float(a[5]))
    x, y = np.array(lo), np.array(la)
    keep = (x > -125) & (x < -66) & (y > 24) & (y < 50)           # contiguous United States
    xy = np.c_[x[keep], y[keep]]
    rng = np.random.default_rng(7)
    save("GeoNames", xy[rng.permutation(len(xy))])


def porto(max_points=4_000_000):
    with zipfile.ZipFile(RAW / "porto.zip") as z:
        names = z.namelist()
        inner = [n for n in names if n.lower().endswith("train.csv.zip") or n.lower().endswith("train.csv")][0]
        data = z.read(inner)
    if inner.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z2:
            data = z2.read([n for n in z2.namelist() if n.lower().endswith(".csv")][0])
    import csv
    reader = csv.reader(io.StringIO(data.decode("utf-8")))
    head = next(reader)
    it, ip = head.index("TIMESTAMP"), head.index("POLYLINE")
    t, x, y = [], [], []
    for cells in reader:
        try:
            ts = int(cells[it]); poly = json.loads(cells[ip])
        except Exception:
            continue
        for j, (lo, la) in enumerate(poly):
            t.append(ts + 15 * j); x.append(lo); y.append(la)
        if len(t) >= max_points * 1.2:
            break
    t, x, y = np.array(t), np.array(x), np.array(y)
    keep = (x > -8.75) & (x < -8.45) & (y > 41.05) & (y < 41.30)  # Porto metropolitan area
    o = np.argsort(t[keep], kind="stable")
    save("Porto", np.c_[x[keep][o], y[keep][o]][:max_points])


def dragon():
    with gzip.open(RAW / ("dragon_up.ply.gz" if (RAW / "dragon_up.ply.gz").exists() else "dragon.ply.gz"), "rb") as f:
        buf = f.read()
    end = buf.index(b"end_header") + len(b"end_header")
    while buf[end:end + 1] in (b"\r", b"\n"):
        end += 1
    header = buf[:end].decode("ascii", "ignore").splitlines()
    n = next(int(h.split()[-1]) for h in header if h.startswith("element vertex"))
    props, in_vertex = [], False
    for h in header:
        if h.startswith("element"):
            in_vertex = h.startswith("element vertex")
        elif h.startswith("property") and in_vertex:
            props.append(h.split()[1:])
    fmt = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1", "int": "i4", "int32": "i4"}
    little = any("binary_little_endian" in h for h in header)
    dt = np.dtype([(p[-1], ("<" if little else ">") + fmt[p[0]]) for p in props])
    v = np.frombuffer(buf, dtype=dt, count=n, offset=end)
    save("Dragon", np.c_[v["x"], v["y"], v["z"]].astype(np.float64))


if __name__ == "__main__":
    for name, fn in [("Uber", uber), ("GeoNames", geonames), ("Dragon", dragon), ("Porto", porto)]:
        try:
            fn()
        except Exception as e:  # report and continue with the other datasets
            print("FAILED", name, repr(e), flush=True)
