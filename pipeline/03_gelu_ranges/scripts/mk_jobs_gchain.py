"""GELU-chain top jobs (non-stem chains). On the board a GELU chain is: int8 conv/SE-Mul output (chain input Q) -> ACTIV
unit -> int8 GELU output (chain output Q). W2's M1 (in S3) cut each chain-input range to [lo_W2, ~top of the GELU-output
range], so the GELU-OUTPUT top (P99.95 on 24 COCO images) now clips the whole chain. Here the output top is raised to a
new value h (method below) and the input top to GELU^-1(h) (the smallest x whose GELU reaches h); the input lower bound
(W2's GELU-aware bound) and the output lower bound (GELU minimum, -0.17) are kept.
methods: p9999 / p99999 / mse (of the FP32 GELU-output histogram, calib-A) / xF (current top times F)
usage: mk_jobs_gchain.py <base.onnx> <out.json> <methods comma> [--each] [--only substr1,substr2]
  default: one job per method changing all 27 non-stem chains; --each: one job per chain and method."""
import sys, json, math, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, w3lib as L, ranges as RG


def gelu(x): return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


def ginv(h):
    lo, hi = 0.0, max(10.0, h + 1)
    for _ in range(80):
        mid = (lo + hi) / 2
        if gelu(mid) < h: lo = mid
        else: hi = mid
    return hi


_G = np.vectorize(lambda x: 0.5 * x * (1 + math.erf(x / math.sqrt(2))))


_GX = {}


def chain_err(S, c, lo_i, hi_i, lo_o, hi_o):
    """W2-style local error of one GELU unit (int8 in -> exact GELU -> int8 out) vs GELU(x), FP32 input histogram."""
    xc, h, w, mn, mx = S.hist(c["input"])
    if c["input"] not in _GX: _GX[c["input"]] = _G(xc)
    gx = _GX[c["input"]]
    si, zi = RG.q_of(lo_i, hi_i); so, zo = RG.q_of(lo_o, hi_o)
    code = np.clip(np.round(xc / si) + zi, -128, 127).astype(np.int64)
    lut = (np.clip(np.round(_G((np.arange(-128, 128) - zi) * si) / so) + zo, -128, 127) - zo) * so
    yd = lut[code + 128]
    return float((h * (yd - gx) ** 2).sum() / h.sum())


def chain_edit(m, S, c, meth, I):
    so, zo = I.qparams(c["output"]); lo_o, hi_o = L.q_to_range(so, zo)
    si, zi = I.qparams(c["input"]); lo_i, hi_i = L.q_to_range(si, zi)
    lo_i_new = lo_i
    if meth.startswith("x"): h = hi_o * float(meth[1:])
    elif meth.startswith("mse"):
        h = RG.mse_opt(S, c["output"])[1] * (float(meth[3:]) if len(meth) > 3 else 1.0)
    elif meth in ("cl1", "cl2"):
        mxo = S.maxs[S.idx[c["output"]]]
        hs = np.geomspace(hi_o, max(hi_o * 1.01, mxo), 60)
        los = [lo_i] if meth == "cl1" else list(np.linspace(-4.5, -1.0, 15)) + [lo_i]
        best = (chain_err(S, c, lo_i, hi_i, lo_o, hi_o), hi_o, lo_i)
        for hh in hs:
            for ll in los:
                e = chain_err(S, c, ll, max(hi_i, ginv(hh)), lo_o, hh)
                if e < best[0]: best = (e, hh, ll)
        h, lo_i_new = best[1], best[2]
    else: h = RG.pct(S, c["output"], float(meth[1:3] + "." + meth[3:]))[1]
    h = max(h, hi_o)                                # only raise the top
    s_o, z_o = RG.q_of(lo_o, h)
    s_i, z_i = RG.q_of(lo_i_new, max(hi_i, ginv(h)))
    return [f"{c['output']}@{s_o!r}@{z_o}", f"{c['input']}@{s_i!r}@{z_i}"], (lo_o, hi_o, h, lo_i, hi_i, ginv(h))


if __name__ == "__main__":
    base, outp, methods = sys.argv[1:4]
    each = "--each" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    m = L.load(base); S = RG.Stats(); I = L.Info(m)
    ch = [c for c in L.gelu_chains(m) if "patch_embed" not in c["input"]]
    if only: ch = [c for c in ch if any(o in c["output"] for o in only)]
    jobs = []
    for meth in methods.split(","):
        allp = []
        for c in ch:
            p, info = chain_edit(m, S, c, meth, I)
            allp += p
            if each: jobs.append([f"gch:{meth}:{c['base']}", "q=" + "|".join(p)])
            print(meth, c["base"].split("/model/")[-1][:50], "out [%.3f,%.3f]->top %.3f | in [%.3f,%.3f]->top %.3f" % info)
        if not each: jobs.append([f"gch_all{len(ch)}:{meth}", "q=" + "|".join(allp)])
    json.dump(jobs, open(outp, "w"), indent=0); print(len(jobs), "jobs")
