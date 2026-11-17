"""Local (single GELU unit) error scan of the chain-input int8 range [lo, hi], from the FP32 calib-pool histograms.
error(lo, hi) = mean over FP32 values x of ( Qout(GELU(Qin_[lo,hi](x))) - GELU(x) )^2, i.e. the board GELU unit
(int8 input at the conv-output Q, exact GELU, one rounding at the chain-output Q, which is kept as in P99.95).
Also: 'free' bounds = the widest clip that changes no output code for any clipped value.
-> results/scan_local_n<N>.json ; usage: python -I scan_local.py [N=200]"""
import sys, json, math, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
z = np.load(w2lib.W2 / "results" / f"fp32_gelu_hist_n{N}.npz")
chains = {c["x"]: c for c in w2lib.gelu_chains(w2lib.load())}
G = w2lib.gelu


def qo(v, s, zp):
    return (np.clip(np.rint(v / s) + zp, -128, 127) - zp) * s


def err(xc, w, lo, hi, so, zo):
    s, zp = w2lib.range_to_qp(lo, hi)
    xq = (np.clip(np.rint(xc / s) + zp, -128, 127) - zp) * s
    d = qo(G(xq), so, zo) - G(xc)
    return float((w * d * d).sum() / w.sum())


res = []
for k, t in enumerate(z["names"]):
    t = str(t); c = chains[t]
    R = float(z["R"][k]); H = z["H"][k]; NB = H.size
    centers = -R + (np.arange(NB) + 0.5) * (2 * R / NB)
    nz = H > 0; xc = centers[nz]; w = H[nz].astype(np.float64)
    so, zo = c["s_out"], c["z_out"]
    lo0, hi0 = w2lib.qp_to_range(c["s_in"], c["z_in"])
    out_lo, out_hi = w2lib.qp_to_range(so, zo)
    floor = float((w * (qo(G(xc), so, zo) - G(xc)) ** 2).sum() / w.sum())
    e0 = err(xc, w, lo0, hi0, so, zo)
    # free bounds: lo_free = largest lo (most negative side clipped) with round(GELU(lo)/so) == round(0/so) and lo < -0.75
    xs = np.linspace(-0.76, -12, 50000)
    okl = np.rint(G(xs) / so) == 0
    lo_free = float(xs[np.argmax(okl)]) if okl.any() else lo0   # first (largest) x where the code equals code(0)
    # hi_free = smallest hi with GELU(hi) >= out_hi + so/2 (rounds to / saturates at the top code)
    xh = np.linspace(0, 3 * max(out_hi, 1), 200000)
    hi_free = float(xh[np.argmax(G(xh) >= out_hi + so / 2)])
    e_free = err(xc, w, lo_free, hi_free, so, zo)
    # grid search
    los = np.unique(np.concatenate([np.linspace(lo0, -0.5, 40), [lo_free]]))
    his = np.unique(np.concatenate([np.linspace(0.3 * hi_free, max(hi0, hi_free), 40), [hi_free]]))
    best = (e0, lo0, hi0)
    for lo in los:
        for hi in his:
            e = err(xc, w, lo, hi, so, zo)
            if e < best[0]: best = (e, float(lo), float(hi))
    # refine around best
    e_b, lo_b, hi_b = best
    dl = (los[1] - los[0]) if len(los) > 1 else 0.1; dh = his[1] - his[0]
    for lo in np.linspace(lo_b - 2 * abs(dl), min(lo_b + 2 * abs(dl), -0.3), 21):
        for hi in np.linspace(max(hi_b - 2 * dh, 0.1), hi_b + 2 * dh, 21):
            e = err(xc, w, lo, hi, so, zo)
            if e < best[0]: best = (e, float(lo), float(hi))
    frac_below_free = float(w[xc < lo_free].sum() / w.sum()); frac_above_free = float(w[xc > hi_free].sum() / w.sum())
    r = dict(tensor=t, fp32_min=float(z["min"][k]), fp32_max=float(z["max"][k]), cur=[lo0, hi0], cur_scale=c["s_in"],
             out=[out_lo, out_hi], out_scale=so, floor=floor, err_cur=e0, free=[lo_free, hi_free],
             free_scale=w2lib.range_to_qp(lo_free, hi_free)[0], err_free=e_free, best=[best[1], best[2]],
             best_scale=w2lib.range_to_qp(best[1], best[2])[0], err_best=best[0],
             frac_below_free=frac_below_free, frac_above_free=frac_above_free)
    res.append(r)
    print(f"{k:2d} {t[7:60]:53s} cur[{lo0:7.2f},{hi0:6.2f}] e={e0:.3e} | free[{lo_free:6.2f},{hi_free:6.2f}] e={e_free:.3e} "
          f"| best[{best[1]:6.2f},{best[2]:6.2f}] e={best[0]:.3e} | floor {floor:.3e}  ratio cur/best {e0 / best[0]:.2f}", flush=True)
json.dump(res, open(w2lib.W2 / "results" / f"scan_local_n{N}.json", "w"), indent=1)
