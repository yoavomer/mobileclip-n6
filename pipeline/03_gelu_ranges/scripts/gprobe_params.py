"""Table-aware step for the W3 GELU-chain ranges (method cl2 = chain-local optimum of (input lo, top) per chain).
For each of the 27 non-stem chains: the cl2 point and 5 neighbours (top x0.97 / x1.03 / x1.06, input lo -0.2 / +0.2)
-> (s_in, z_in, s_out, z_out) options, each with its chain-local error. Writes results/gprobe_options.json and the probe
parameter lists results/gprobe_a_params.json / gprobe_b_params.json (81 branches each, W2's probe size).
usage: gprobe_params.py <base.onnx>"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, w3lib as L, ranges as RG
from mk_jobs_gchain import chain_err, ginv, chain_edit

base = sys.argv[1]
m = L.load(base); S = RG.Stats(); I = L.Info(m)
ch = [c for c in L.gelu_chains(m) if "patch_embed" not in c["input"]]
opts = []
for c in ch:
    p, info = chain_edit(m, S, c, "cl2", I)
    (to, so, zo), (ti, si, zi) = [x.split("@") for x in p]
    so, zo, si, zi = float(so), int(zo), float(si), int(zi)
    lo_o, h = L.q_to_range(so, zo); lo_i, hi_i = L.q_to_range(si, zi)
    s0o, z0o = I.qparams(c["output"]); lo_o0 = L.q_to_range(s0o, z0o)[0]
    s0i, z0i = I.qparams(c["input"]); hi_i0 = L.q_to_range(s0i, z0i)[1]
    cand = []
    for fh, dlo in ((1.0, 0.0), (0.97, 0.0), (1.03, 0.0), (1.06, 0.0), (1.0, -0.2), (1.0, 0.2)):
        hh = h * fh; ll = lo_i + dlo
        s_o, z_o = RG.q_of(lo_o0, hh); s_i, z_i = RG.q_of(ll, max(hi_i0, ginv(hh)))
        e = chain_err(S, c, ll, max(hi_i0, ginv(hh)), lo_o0, hh)
        cand.append({"fh": fh, "dlo": dlo, "params": [float(np.float32(s_i)), int(z_i), float(np.float32(s_o)), int(z_o)], "local_err": e})
    opts.append({"input": c["input"], "output": c["output"], "base": c["base"], "options": cand,
                 "cur": [s0i, z0i, s0o, z0o], "cur_err": chain_err(S, c, *L.q_to_range(s0i, z0i), *L.q_to_range(s0o, z0o))})
json.dump(opts, open(L.W3 / "results" / "gprobe_options.json", "w"), indent=1)
allp = [o2["params"] for o in opts for o2 in o["options"]]
# 2 validation branches first in each probe: P99.95's own network.4.3 fc1 and network.7.0 fc1 chains, whose compiled
# tables are known from quant/layers/gws_p9995 (1 and 10 codes differ from ideal) -> checks the branch pairing.
P = L.load(L.BASE); IP = L.Info(P); val = []
for t_in, t_out in (("/model/network.4/network.4.3/convffn/fc1/Conv_output_0", "/model/network.4/network.4.3/convffn/act/Mul_1_output_0"),
                    ("/model/network.7/network.7.0/convffn/fc1/Conv_output_0", "/model/network.7/network.7.0/convffn/act/Mul_1_output_0")):
    (a, b), (c_, d) = IP.qparams(t_in), IP.qparams(t_out); val.append([a, b, c_, d])
json.dump(val + allp[:81], open(L.W3 / "results" / "gprobe_a_params.json", "w"))
json.dump(val + allp[81:], open(L.W3 / "results" / "gprobe_b_params.json", "w"))
print(len(opts), "chains,", len(allp), "probe branches")
