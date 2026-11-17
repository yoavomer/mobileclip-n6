import os  # MCLIP_ROOT = the working folder (default: current folder)
import onnx, re, collections, sys, json
from onnx import numpy_helper as nh
import numpy as np
m = onnx.load(os.environ.get("MCLIP_ROOT", ".") + r"\quant\layers\mci0_256_p9995.onnx"); g = m.graph
ini = {t.name: nh.to_array(t) for t in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons = collections.defaultdict(list)
for n in g.node:
    for i in n.input: cons[i].append(n)
qn = [n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] not in ini]
print("activation Q nodes:", len(qn))
def block(t):
    s = re.search(r"/model/(patch_embed|conv_exp|head)", t) or re.search(r"(network\.\d+\.\d+|network\.\d+)", t)
    return s.group(1) if s else t
cnt = collections.Counter(block(n.input[0]) for n in qn)
for k, v in cnt.items(): print(f"{k:20s} {v}")
ops = collections.Counter(n.op_type for n in g.node)
print(ops)
# print block network.3 tensors
for n in qn:
    t = n.input[0]
    if block(t) in ("network.3", "network.7.0", "head", "conv_exp", "network.6", "network.2.0") or not t.startswith("/model"):
        p = prod.get(t)
        print(block(t), p.op_type if p else "-", t, float(ini[n.input[1]]), int(ini[n.input[2]]), len(cons[n.output[0]]))
