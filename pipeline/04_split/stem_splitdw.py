"""W4 lever "stem split, variant B": keep patch_embed.0 (Conv 3->64 + GELU) as ONE node chain (so the compiler inserts the
input zero-offset Sub only once), then Split its quantized output into K channel groups, run patch_embed.1 (depthwise
3x3 s2 + GELU) per group and Concat before patch_embed.2. Same int8 weights / scales / zero points (sliced per channel),
same per-tensor activation scale/zp on every new tensor -> integer-identical (checked in ORT).
Variant A (stem_split.py) splits patch_embed.0 too; the compiler then made K copies of the input Sub (split4: all in
npuRAM, one stem group pushed to hyperRAM; split8: 5 copies in hyperRAM). Variant B tries to avoid both.
usage: stem_splitdw.py <src.onnx> <dst.onnx> [K=4]"""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, copy, hashlib
import numpy as np, onnx
from onnx import helper as h, numpy_helper as nh

src, dst = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
m = onnx.load(src); g = m.graph
ini = {t.name: t for t in g.initializer}
prod = {o: n for n in g.node for o in n.output}
byname = {n.name: n for n in g.node}
P0, P1, P2 = "/model/patch_embed/patch_embed.0/", "/model/patch_embed/patch_embed.1/", "/model/patch_embed/patch_embed.2/"
conv1 = byname[P1 + "reparam_conv/Conv"]; conv2 = byname[P2 + "reparam_conv/Conv"]
q0 = byname[P0 + "activation/Mul_1_output_0_QuantizeLinear"]      # block-0 output Q (scale/zp reused)
dq0 = byname[P0 + "activation/Mul_1_output_0_DequantizeLinear"]
C = 64; G = C // K
names1 = [P1 + x for x in ("reparam_conv/Conv_output_0_QuantizeLinear", "reparam_conv/Conv_output_0_DequantizeLinear",
                           "activation/Div", "activation/Erf", "activation/Erf_output_0_QuantizeLinear", "activation/Erf_output_0_DequantizeLinear",
                           "activation/Add", "activation/Add_output_0_QuantizeLinear", "activation/Add_output_0_DequantizeLinear",
                           "activation/Mul", "activation/Mul_output_0_QuantizeLinear", "activation/Mul_output_0_DequantizeLinear",
                           "activation/Mul_1", "activation/Mul_1_output_0_QuantizeLinear", "activation/Mul_1_output_0_DequantizeLinear")]
chain1 = [byname[x] for x in names1]
new = []
sp = P0 + f"splitdw{K}/split"
splits = nh.from_array(np.array([G] * K, np.int64), sp + "_sizes"); g.initializer.append(splits)
parts = [f"{sp}_out{k}" for k in range(K)]
new.append(h.make_node("Split", [dq0.output[0], splits.name], parts, name=sp, axis=1))
outs = []
for k in range(K):
    sfx = f"_dw{k}of{K}"
    new.append(h.make_node("QuantizeLinear", [parts[k], q0.input[1], q0.input[2]], [parts[k] + "_q"], name=sp + f"_Q{k}"))
    new.append(h.make_node("DequantizeLinear", [parts[k] + "_q", q0.input[1], q0.input[2]], [parts[k] + "_dq"], name=sp + f"_DQ{k}"))
    xs = []
    for which, dqn in (("w", prod[conv1.input[1]]), ("b", prod[conv1.input[2]])):
        qq, ss, zz = (ini[i] for i in dqn.input)
        sl = slice(k * G, (k + 1) * G)
        a = [nh.from_array(np.ascontiguousarray(nh.to_array(t)[sl]), t.name + sfx) for t in (qq, ss, zz)]
        g.initializer.extend(a)
        ax = [at for at in dqn.attribute if at.name == "axis"]
        new.append(h.make_node("DequantizeLinear", [t.name for t in a], [dqn.output[0] + sfx], name=dqn.name + sfx, **({"axis": ax[0].i} if ax else {})))
        xs.append(dqn.output[0] + sfx)
    attrs = {at.name: h.get_attribute_value(at) for at in conv1.attribute}; attrs["group"] = G
    ren = {conv1.output[0]: conv1.output[0] + sfx}
    new.append(h.make_node("Conv", [parts[k] + "_dq", xs[0], xs[1]], [conv1.output[0] + sfx], name=conv1.name + sfx, **attrs))
    for n in chain1:
        ins = [ren.get(i, i) for i in n.input]; o2 = [o + sfx for o in n.output]
        for o, oo in zip(n.output, o2): ren[o] = oo
        nn = h.make_node(n.op_type, ins, o2, name=n.name + sfx); nn.attribute.extend(copy.deepcopy(list(n.attribute))); new.append(nn)
    outs.append(ren[chain1[-1].output[0]])
q1 = chain1[-2]
cat = P1 + f"splitdw{K}/concat"
new.append(h.make_node("Concat", outs, [cat + "_out"], name=cat, axis=1))
new.append(h.make_node("QuantizeLinear", [cat + "_out", q1.input[1], q1.input[2]], [cat + "_q"], name=cat + "_Q"))
new.append(h.make_node("DequantizeLinear", [cat + "_q", q1.input[1], q1.input[2]], [cat + "_dq"], name=cat + "_DQ"))
conv2.input[0] = cat + "_dq"
remove = [conv1] + chain1 + [prod[conv1.input[1]], prod[conv1.input[2]]]
for n in remove: g.node.remove(n)
idx = list(g.node).index(dq0) + 1
for j, n in enumerate(new): g.node.insert(idx + j, n)
used = {i for n in g.node for i in n.input}
for t in [t for t in g.initializer if t.name not in used]: g.initializer.remove(t)
onnx.checker.check_model(m); onnx.save(m, dst)

import onnxruntime as ort
res = {"src": src, "dst": dst, "K": K}
x = np.load(os.environ.get("MCLIP_ROOT", ".") + r"\calib_inputs.npy")[:4]
xs = [x[i:i + 1] for i in range(4)] + [np.random.default_rng(0).random((1, 3, 256, 256), dtype=np.float32)]
for label, opt in (("default", None), ("opt_off", ort.GraphOptimizationLevel.ORT_DISABLE_ALL)):
    so = ort.SessionOptions(); so.intra_op_num_threads = 3
    if opt is not None: so.graph_optimization_level = opt
    sa = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"]); sb = ort.InferenceSession(dst, so, providers=["CPUExecutionProvider"])
    res[f"max_abs_diff_{label}"] = max(float(np.abs(sa.run(None, {"input": xi})[0] - sb.run(None, {"input": xi})[0]).max()) for xi in xs)
I2 = {t.name: nh.to_array(t) for t in onnx.load(dst).graph.initializer}
a2 = {n.input[0]: int(I2[n.input[2]]) for n in onnx.load(dst).graph.node if n.op_type == "QuantizeLinear" and n.input[0].startswith("/model/network.7/")
      and n.input[0].endswith(("token_mixer/Softmax_output_0", "token_mixer/Mul_output_0"))}
res["a2_zero_points"] = a2; res["a2_ok"] = len(a2) == 4 and all(v == 0 for v in a2.values())
res["dst_sha256"] = hashlib.sha256(open(dst, "rb").read()).hexdigest()
print(json.dumps(res, indent=1)); json.dump(res, open(dst.replace(".onnx", "_info.json"), "w"), indent=1)
