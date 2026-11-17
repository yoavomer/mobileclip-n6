"""Scale-multiplier jobs: for each given leader tensor (followers get the same params), scale s -> f*s with the zero point
kept (range [lo, hi] -> [f lo, f hi]; for the 3 SE Relu tensors zp -128 stays, so the Relu clamp is kept).
usage: mk_jobs_mult.py <base.onnx> <out.json> <f1,f2,..> <tensor1> [tensor2 ...]   (prefix 'tensor@' means leader name)"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent)); import w3lib as L
base, outp, fs = sys.argv[1:4]; ts = sys.argv[4:]
m = L.load(base); I = L.Info(m); F = L.followers(m)
jobs = []
for t in ts:
    assert t not in L.ATT, t
    s, z = I.qparams(t)
    for f in [float(x) for x in fs.split(",")]:
        parts = [f"{x}@{float(s * f)!r}@{z}" for x in [t] + F.get(t, [])]
        jobs.append([f"mul{f}:{t}", "q=" + "|".join(parts)])
json.dump(jobs, open(outp, "w"), indent=0); print(len(jobs), "jobs")
