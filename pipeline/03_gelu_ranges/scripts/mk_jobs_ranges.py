"""Group-wise range-method jobs: for each tensor class and method, every LEADER tensor of that class (followers = outputs of
data-movement ops with the same Q params get the same new params) gets the method's int8 range from the FP32 calib-A
histograms. Excluded: A2 operands, GELU chain tensors (W2's M1 / ACTIV tables), SE Sigmoid in/out (ACTIV tables),
softmax inputs (Q·Kᵀ MatMul outputs: compiled exp table), and anything not listed.
usage: mk_jobs_ranges.py <base.onnx> <out.json> <classes comma> <methods comma> [prefix]
methods: p9995 / p9999 / p999 / p99999 (ORT-style |x| percentile, clamped to [min,max]), minmax, mse"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, w3lib as L, ranges as RG

EXCL_SUFFIX = ("token_mixer/MatMul_output_0",)


def leaders(m, cls):
    F = L.followers(m); fol = {x for v in F.values() for x in v}
    out = []
    for t in L.Info(m).actQ:
        if t in fol or t in L.ATT or t.endswith(EXCL_SUFFIX): continue
        if L.tclass(t, m) in cls: out.append(t)
    return out, F


def method_range(S, t, meth):
    if meth.startswith("p") and meth[1:].isdigit():
        p = float(meth[1:3] + "." + meth[3:]); return RG.pct(S, t, p)
    if meth == "minmax": return RG.minmax(S, t)
    if meth == "mse": return RG.mse_opt(S, t)
    raise ValueError(meth)


def spec_for(m, S, ts, F, meth):
    parts = []
    for t in ts:
        lo, hi = method_range(S, t, meth)
        s, z = RG.q_of(lo, hi)
        for x in [t] + F.get(t, []): parts.append(f"{x}@{s!r}@{z}")
    return "q=" + "|".join(parts)


if __name__ == "__main__":
    base, outp, classes, methods = sys.argv[1:5]; pre = sys.argv[5] if len(sys.argv) > 5 else ""
    m = L.load(base); S = RG.Stats()
    jobs = []
    for c in classes.split(","):
        ts, F = leaders(m, {c})
        for meth in methods.split(","):
            jobs.append([f"{pre}{c}:{meth}", spec_for(m, S, ts, F, meth)])
        print(c, len(ts))
    json.dump(jobs, open(outp, "w"), indent=0); print(len(jobs), "jobs")
