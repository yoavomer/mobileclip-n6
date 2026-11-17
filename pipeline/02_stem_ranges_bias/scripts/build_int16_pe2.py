"""M5: P99.95 A2 with ONLY the patch_embed.2 conv output as INT16 QDQ (com.microsoft QuantizeLinear/DequantizeLinear,
int16 zp 0, symmetric), built the same way as quant/int16/build_int16.py (own copy, W2 may not touch quant/int16):
T = max(|lowest|, |highest|) of that tensor in quant/sweep/P99_95_24_ranges.json, scale = 2T/65535.
-> night/W2_stem/models/I16_pe2.onnx + results/I16_pe2_build.json"""
import sys, json, pathlib
import numpy as np, onnx
from onnx import numpy_helper as nh, helper as h
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib

T_NAME = w2lib.STEM[2]
rng = json.load(open(w2lib.R / "quant" / "sweep" / "P99_95_24_ranges.json"))["data"]
val = lambda x: float(np.asarray(x["data"]).ravel()[0]) if isinstance(x, dict) else float(x)
m = w2lib.load(); g = m.graph
ini = {x.name: x for x in g.initializer}
before = {x.name: x.SerializeToString() for x in g.initializer}
lo, hi = val(rng[T_NAME]["lowest"]), val(rng[T_NAME]["highest"]); T = max(abs(lo), abs(hi))
s = np.float32(2 * T / 65535.0)
q = next(n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] == T_NAME)
dqs = [n for n in g.node if n.op_type == "DequantizeLinear" and n.input[0] == q.output[0]]
s8, z8 = float(nh.to_array(ini[q.input[1]])), int(nh.to_array(ini[q.input[2]]))
sn, zn = T_NAME + "__s16", T_NAME + "__zp16"
g.initializer.extend([nh.from_array(np.array(s, np.float32), sn), nh.from_array(np.array(0, np.int16), zn)])
for n in [q] + dqs:
    n.domain = "com.microsoft"; n.input[1] = sn; n.input[2] = zn
used = {i for n in g.node for i in n.input}
dropped = [x.name for x in g.initializer if x.name not in used]
keep = [x for x in g.initializer if x.name in used]
del g.initializer[:]; g.initializer.extend(keep)
if not any(o.domain == "com.microsoft" for o in m.opset_import):
    m.opset_import.append(h.make_opsetid("com.microsoft", 1))
after = {x.name: x.SerializeToString() for x in g.initializer}
assert not [k for k in after if k in before and after[k] != before[k]]
ok, zps = w2lib.a2_check(m)
assert ok, zps
out = w2lib.W2 / "models" / "I16_pe2.onnx"
sha = w2lib.save(m, out)
info = {"tensor": T_NAME, "calib_lowest": lo, "calib_highest": hi, "T": T, "int16_scale": float(s), "int16_zp": 0,
        "int8_scale": s8, "int8_zp": z8, "n_dq": len(dqs), "dropped_unused_initializers": dropped,
        "a2_zps": zps, "out": str(out), "sha256": sha}
json.dump(info, open(w2lib.W2 / "results" / "I16_pe2_build.json", "w"), indent=1)
print(json.dumps(info, indent=1))
