import os  # MCLIP_ROOT = the working folder (default: current folder)
import onnx, re, collections
from onnx import numpy_helper as nh
m = onnx.load(os.environ.get("MCLIP_ROOT", ".") + r"\quant\layers\mci0_256_p9995.onnx"); g = m.graph
ini = {t.name for t in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons = collections.defaultdict(list)
for n in g.node:
    for i in n.input: cons[i].append(n)
for t in ["gemm_input_reshape_arg","gemm_output_reshape_arg","gemm_input_reshape_arg_token_1","gemm_output_reshape_arg_token_4","embedding_QuantizeLinear_Input"]:
    p = prod[t]; print(t, "prod", p.op_type, p.name, list(p.input)[:2])
    q = [c for c in cons[t]][0]; dq = cons[q.output[0]][0]
    print("   cons of DQ:", [(c.op_type, c.name) for c in cons[dq.output[0]]])
# network.7.0 tensors
for n in g.node:
    if n.op_type == "QuantizeLinear" and n.input[0] not in ini and "network.7/network.7.0" in n.input[0]:
        print(prod[n.input[0]].op_type, n.input[0])
print([o.name for o in g.output], [i.name for i in g.input])
