"""Build a W3 candidate: base ONNX + spec (w3lib.apply_spec; q-edits also re-express conv biases whose input scale changed)
-> pruned, checker-validated ONNX + <out>.json (changed Q params old -> new, biases changed, A2 zero points, SHA256).
usage: build_cand.py <base.onnx> <out.onnx> <spec | @spec_file.txt>"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, onnx, w3lib as L
from onnx import numpy_helper as nh

base, out, spec = sys.argv[1:4]
if spec.startswith("@"): spec = pathlib.Path(spec[1:]).read_text().strip()
m0 = L.load(base); m = L.apply_spec(m0, spec); L.prune(m)
onnx.checker.check_model(m)
ok, zps = L.a2_check(m); assert ok, zps
I0, I1 = L.Info(m0), L.Info(m)
ch = {t: {"old": I0.qparams(t), "new": I1.qparams(t), "old_range": L.q_to_range(*I0.qparams(t)), "new_range": L.q_to_range(*I1.qparams(t))}
      for t in I0.actQ if I0.qparams(t) != I1.qparams(t)}
def biases(mm):
    I = L.Info(mm); r = {}
    for c in mm.graph.node:
        if c.op_type in ("Conv", "Gemm") and len(c.input) > 2:
            p = I.prod.get(c.input[2])
            if p is not None and p.op_type == "DequantizeLinear" and p.input[0] in I.ini:
                r[c.name] = (nh.to_array(I.ini[p.input[0]]), nh.to_array(I.ini[p.input[1]]))
    return r
b0, b1 = biases(m0), biases(m)
bch = [k for k in b0 if not (np.array_equal(b0[k][0], b1[k][0]) and np.array_equal(b0[k][1], b1[k][1]))]
onnx.save(m, out)
info = {"base": str(base), "base_sha256": L.sha256_file(base), "out": str(out), "sha256": L.sha256_file(out),
        "a2_ok": ok, "a2": {k: list(v) for k, v in zps.items()}, "n_q_changed": len(ch), "q_changed": ch,
        "biases_rescaled": bch, "spec_len": len(spec)}
json.dump(info, open(str(out) + ".json", "w"), indent=1, default=str)
print(json.dumps({k: info[k] for k in ("out", "sha256", "a2_ok", "n_q_changed")}), "biases rescaled", len(bch))
