"""Local GELU-unit error with the COMPILED table (from a probe compile) vs with the ideal table, per probed range.
error = mean over FP32 calib values x of ((T[Qin(x)] - z_out) * s_out - GELU(x))^2, T = compiled or ideal 256-entry table.
usage: python -I table_aware.py <probe_read.json> <labels.json> [hist N=200]  -> prints, writes <probe_read>_eval.json"""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib

rd = json.load(open(sys.argv[1])); labels = json.load(open(sys.argv[2]))
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
z = np.load(w2lib.W2 / "results" / f"fp32_gelu_hist_n{N}.npz")
names = [str(x) for x in z["names"]]
chains = w2lib.gelu_chains(w2lib.load())
out = []
for r, lab in zip(rd, labels):
    s_in, z_in, s_out, z_out = r["params"]
    # which tensor: label "peK_..." -> stem K; "val_*_peK" -> stem K; "t<idx>_..." -> chain idx
    if lab.startswith("t"):
        k = int(lab[1:].split("_")[0]); t = chains[k]["x"]
    else:
        t = w2lib.STEM[int(lab.split("pe")[1][0])]
    i = names.index(t); R = float(z["R"][i]); H = z["H"][i]; NB = H.size
    xc = (-R + (np.arange(NB) + 0.5) * (2 * R / NB))[H > 0]; w = H[H > 0].astype(np.float64)
    q = np.clip(np.rint(xc / s_in) + z_in, -128, 127).astype(np.int64)
    g = w2lib.gelu(xc)
    e = {}
    for kind in ("lut", "ideal"):
        T = np.asarray(r[kind], np.int64)
        y = (T[q + 128] - z_out) * s_out
        e[kind] = float((w * (y - g) ** 2).sum() / w.sum())
    # share of the data that falls on codes where compiled != ideal
    bad = np.asarray(r["lut"]) != np.asarray(r["ideal"])
    share_bad = float(w[bad[q + 128]].sum() / w.sum())
    rec = {"label": lab, "tensor": t, "params": r["params"], "range": list(w2lib.qp_to_range(s_in, z_in)), "rows": r["rows"],
           "n_diff": r["n_diff"], "data_on_diff_codes": share_bad, "err_compiled": e["lut"], "err_ideal": e["ideal"]}
    out.append(rec)
    print(f"{lab:26s} range [{rec['range'][0]:7.2f},{rec['range'][1]:6.2f}] rows {r['rows']:2d} diff {r['n_diff']:2d} "
          f"data-on-diff {100 * share_bad:5.1f}%  err compiled {e['lut']:.3e} ideal {e['ideal']:.3e}")
json.dump(out, open(sys.argv[1].replace(".json", "_eval.json"), "w"), indent=1)
