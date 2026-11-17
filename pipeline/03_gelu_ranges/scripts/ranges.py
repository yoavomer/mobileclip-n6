"""Per-tensor int8 range candidates from the FP32 calib-A histograms (results/fp32_stats_calibA_n500.npz).
All ranges are scalar asymmetric int8 (s, zp) in the ORT convention (lo <= 0 <= hi, zp = round(-128 - lo/s)).
  pct(p, sym=True)  ORT-style: threshold on |x| at cumulative share p/100, range (-T, T) clamped to [min, max]
  pct_asym(p)       signed histogram, (100-p)/2 % cut on each side
  minmax            [min, max]
  mse               argmin over [lo, hi] of E[(Q(x) - x)^2]  (inside: s^2/12, outside: squared distance to the edge)
                    grid: hi over 200 log-spaced values down to max/1000, lo likewise (or lo = min if one-sided)
Bin-centre approximation with 8192 bins on [min, max]."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import numpy as np, pathlib

R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r""); W3 = R / "night" / "W3_beyond"


class Stats:
    def __init__(self, path=W3 / "results" / "fp32_stats_calibA_n500.npz"):
        z = np.load(path)
        self.names = [str(x) for x in z["names"]]; self.idx = {n: i for i, n in enumerate(self.names)}
        self.mins, self.maxs, self.H = z["mins"], z["maxs"], z["hist"].astype(np.float64)
        self.NB = int(z["nbins"]); self.count = z["count"]; self.sum = z["sum"]; self.sumsq = z["sumsq"]

    def hist(self, t):
        i = self.idx[t]; mn, mx = self.mins[i], self.maxs[i]
        w = (mx - mn) / self.NB
        c = mn + (np.arange(self.NB) + 0.5) * w
        return c, self.H[i], w, mn, mx


def q_of(lo, hi):
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    s = (hi - lo) / 255.0 or 1.0
    zp = int(np.clip(np.round(-128 - lo / s), -128, 127))
    return float(np.float32(s)), zp


def rng_of(s, zp): return ((-128 - zp) * s, (127 - zp) * s)


def mse_of(c, h, w, s, zp):
    lo, hi = rng_of(s, zp)
    inside = (c >= lo) & (c <= hi)
    e = np.where(inside, s * s / 12.0, np.where(c < lo, (lo - c) ** 2, (c - hi) ** 2) + w * w / 12.0)
    return float((h * e).sum() / h.sum())


def pct(S, t, p):
    c, h, w, mn, mx = S.hist(t)
    a = np.abs(c); o = np.argsort(a); cdf = np.cumsum(h[o]) / h.sum()
    T = a[o][min(np.searchsorted(cdf, p / 100.0), len(o) - 1)] + w / 2
    return max(-T, mn), min(T, mx)


def pct_asym(S, t, p):
    c, h, w, mn, mx = S.hist(t)
    cdf = np.cumsum(h) / h.sum(); q = (100.0 - p) / 200.0
    lo = c[min(np.searchsorted(cdf, q), len(c) - 1)] - w / 2
    hi = c[min(np.searchsorted(cdf, 1 - q), len(c) - 1)] + w / 2
    return lo, hi


def minmax(S, t):
    i = S.idx[t]; return float(S.mins[i]), float(S.maxs[i])


def mse_opt(S, t, n=120):
    c, h, w, mn, mx = S.hist(t)
    his = [0.0] if mx <= 0 else list(np.geomspace(mx, max(mx / 1000, 1e-9), n))
    los = [0.0] if mn >= 0 else list(-np.geomspace(-mn, max(-mn / 1000, 1e-9), n))
    best = (np.inf, None)
    # coarse 2-D search on a sub-grid, then 1-D refinements
    for hi in his[::4]:
        for lo in los[::4]:
            s, z = q_of(lo, hi); e = mse_of(c, h, w, s, z)
            if e < best[0]: best = (e, (lo, hi))
    for _ in range(2):
        lo, hi = best[1]
        for hi2 in his:
            s, z = q_of(lo, hi2); e = mse_of(c, h, w, s, z)
            if e < best[0]: best = (e, (lo, hi2))
        lo, hi = best[1]
        for lo2 in los:
            s, z = q_of(lo2, hi); e = mse_of(c, h, w, s, z)
            if e < best[0]: best = (e, (lo2, hi))
    return best[1]


def err(S, t, lo, hi):
    c, h, w, mn, mx = S.hist(t); s, z = q_of(lo, hi); return mse_of(c, h, w, s, z)
