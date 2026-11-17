"""Copy the int32 conv-bias initializers that a bias-corrected model (made from P99.95 by bias_corr.py) changed into another
P99.95-derived model (e.g. an M1 range candidate). Valid because M1 only changes GELU-input (conv-output) Q params,
never a conv's input scale or weight scale, so bias scales are identical (asserted).
usage: python -I transplant_bias.py <target.onnx> <bias_corrected.onnx> <out.onnx>"""
import sys, json, pathlib
import numpy as np
from onnx import numpy_helper as nh
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib

tgt, bc, out = sys.argv[1:4]
base = {t.name: t for t in w2lib.load(w2lib.P9995).graph.initializer}
B = w2lib.load(bc); Bi = {t.name: t for t in B.graph.initializer}
m = w2lib.load(tgt); Ti = {t.name: t for t in m.graph.initializer}
n = 0
for name, t in Bi.items():
    if t.data_type == 6 and name in base and t.SerializeToString() != base[name].SerializeToString():   # 6 = INT32 (all conv biases; v1 matched only *bias_quantized and missed 20 onnx::Conv_*_quantized)
        sc = name + "_scale"
        assert np.array_equal(nh.to_array(Bi[sc]), nh.to_array(Ti[sc])), sc
        assert Ti[name].SerializeToString() == base[name].SerializeToString(), f"target bias already changed: {name}"
        Ti[name].CopyFrom(t); n += 1
ok, zps = w2lib.a2_check(m); assert ok, zps
sha = w2lib.save(m, out)
print(json.dumps({"out": out, "sha256": sha, "biases_transplanted": n, "a2_zps": zps}))
