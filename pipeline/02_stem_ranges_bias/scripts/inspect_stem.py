import os  # MCLIP_ROOT = the working folder (default: current folder)
import onnx, numpy as np
from onnx import numpy_helper as nh
m = onnx.load(os.environ.get("MCLIP_ROOT", ".") + r"\quant\layers\mci0_256_p9995.onnx")
g = m.graph
init = {t.name: nh.to_array(t) for t in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
print("opset", [(o.domain, o.version) for o in m.opset_import], "nodes", len(g.node))
for k in range(3):
    t = f"/model/patch_embed/patch_embed.{k}/reparam_conv/Conv_output_0"
    conv = prod[t]
    print("==", t, conv.op_type, conv.name, conv.input, [ (a.name, onnx.helper.get_attribute_value(a)) for a in conv.attribute])
    for i in conv.input:
        p = prod.get(i)
        print("   in", i, p.op_type if p else "init", p.input if p else init[i].shape)
        if p is not None and p.op_type == "DequantizeLinear":
            for j in p.input:
                if j in init: print("      ", j, init[j].dtype, init[j].shape, init[j].ravel()[:4])
    def walk(x, d=0):
        for c in cons.get(x, []):
            ex = [ (j, init[j].ravel()[:3], init[j].dtype) for j in c.input if j in init]
            print("   "*(d+1), c.op_type, c.name, "->", c.output, ex)
            if d < 9 and c.op_type not in ("Conv",):
                walk(c.output[0], d+1)
    walk(t)
