"""Orchestrator check of one candidate (PC only).
usage: python -I verify_candidate.py <cand.onnx> <compile_out_base or -> [ref.onnx]
Prints:
  - SHA256 of the ONNX
  - A2 rule: zero point of the 4 QuantizeLinear nodes on the attention weight operands (must all be 0)
  - which initializers differ from the reference ONNX (default: P99.95 A2), grouped by kind
    (int8 weights / int32 biases / scales / zero points / other)
  - for a compile dir: epochs HW/SW/hybrid/no-op, SW MACs, hyperRAM, rel bin SHA256 (via W2's compile_report.summary,
    same leaf rule and 'epoch OPS'/2 rule as REPORT.md), and the type-level mapping diff vs P99.95 (maptype.py)."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, hashlib, pathlib, subprocess, json, collections
import numpy as np, onnx
from onnx import numpy_helper
R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
PY = str(R / ".venv" / "Scripts" / "python.exe")
A2 = [f"/model/network.7/network.7.{b}/token_mixer/{t}_output_0" for b in (0, 1) for t in ("Softmax", "Mul")]
cand = pathlib.Path(sys.argv[1]); comp = sys.argv[2] if len(sys.argv) > 2 else "-"
ref = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else R / "quant/layers/mci0_256_p9995.onnx"
print("onnx", cand, "sha256", hashlib.sha256(cand.read_bytes()).hexdigest())
m = onnx.load(str(cand)); init = {i.name: i for i in m.graph.initializer}
ok = True
for t in A2:
    qs = [n for n in m.graph.node if n.op_type == "QuantizeLinear" and n.input[0] == t]
    if len(qs) != 1: print("A2", t, "FOUND", len(qs), "Q nodes -> FAIL"); ok = False; continue
    zp = numpy_helper.to_array(init[qs[0].input[2]]); sc = numpy_helper.to_array(init[qs[0].input[1]])
    good = np.all(zp == 0); ok &= bool(good)
    print(f"A2 {t.split('/')[-3]}/{t.split('/')[-1]:18s} zp={zp.tolist()} scale={float(sc):.7f} {'OK' if good else 'FAIL'}")
print("A2 RULE:", "PASS" if ok else "FAIL")
r = onnx.load(str(ref)); rinit = {i.name: i for i in r.graph.initializer}
kinds = collections.Counter(); same = 0; only = []
for n, i in init.items():
    if n not in rinit: only.append(n); continue
    a, b = numpy_helper.to_array(i), numpy_helper.to_array(rinit[n])
    if a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b): same += 1; continue
    k = ("int8 tensor (weights)" if a.dtype == np.int8 and a.size > 1 else "int32 tensor (biases)" if a.dtype == np.int32
         else "zero point" if a.dtype in (np.int8, np.uint8, np.int16) else "float scale" if a.dtype == np.float32 and a.size <= 1024
         else f"other {a.dtype}")
    kinds[k] += 1
print(f"initializers vs {ref.name}: same {same}, changed {dict(kinds)}, only in candidate {len(only)}, only in ref {len(set(rinit) - set(init))}")
print("nodes cand", len(m.graph.node), "ref", len(r.graph.node), "op types equal:",
      collections.Counter(n.op_type for n in m.graph.node) == collections.Counter(n.op_type for n in r.graph.node))
if comp != "-":
    sys.path.insert(0, str(R / "night/W2_stem/scripts")); import compile_report
    s = compile_report.summary(comp)
    for k in ("exit", "epochs", "sw_macs", "hyperRAM", "report_total_epochs", "rel_bytes", "rel_sha256", "mapping_vs_p9995", "mapping_vs_a2", "error"):
        if k in s: print(k, s[k])
    ci = pathlib.Path(comp) / "gout" / "c_info.json"
    if ci.exists():
        o = subprocess.run([PY, "-I", str(R / "night/final/maptype.py"), str(ci), str(R / "quant/layers/gws_p9995/neural_art__network/c_info.json")],
                           capture_output=True, text=True)
        print("maptype vs p9995:\n" + o.stdout.strip())
