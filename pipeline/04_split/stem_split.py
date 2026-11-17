"""W4 lever "stem split": rewrite the stem of a QDQ model so that patch_embed.0 (Conv 3->64, 3x3 s2 + GELU) and
patch_embed.1 (depthwise 3x3 s2 + GELU) run as K independent channel groups (64/K channels each), concatenated
before patch_embed.2. Every group reuses the SAME int8 weights (sliced along the output-channel axis), the same
per-channel weight scales / zero points (sliced), the same int32 biases (sliced) and the SAME per-tensor activation
scale / zero point for every intermediate tensor -> integer-identical to the original graph (checked in ORT below).
Why: in the compiled P99.95 model the 1 MB patch_embed.0 output [64,128,128] lives in hyperRAM and the depthwise
conv reads it one channel at a time, one byte per 64-B pixel (1,048,576 strided single-byte hyperRAM reads); the
board time model (fit_time.py) attributes ~0.16 us to each such read. With K groups of 16 channels a group output is
256 KB and can fit in one npuRAM pool if the compiler schedules group by group. Whether it does is checked by a compile.
usage: stem_split.py <src.onnx> <dst.onnx> [K=4]
Nothing else in the graph changes; the A2 attention operands are untouched (checked)."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, copy, hashlib
import numpy as np, onnx
from onnx import helper as h, numpy_helper as nh

src, dst = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
m = onnx.load(src)
g = m.graph
ini = {t.name: t for t in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input:
        cons.setdefault(i, []).append(n)

P0, P1, P2 = "/model/patch_embed/patch_embed.0/", "/model/patch_embed/patch_embed.1/", "/model/patch_embed/patch_embed.2/"
conv0 = next(n for n in g.node if n.name == P0 + "reparam_conv/Conv")
conv1 = next(n for n in g.node if n.name == P1 + "reparam_conv/Conv")
conv2 = next(n for n in g.node if n.name == P2 + "reparam_conv/Conv")
C = 64
assert C % K == 0
G = C // K


def chain_nodes(conv):
    """the nodes from the conv's output Q up to (and incl.) the DQ of the block's activation/Mul_1 output"""
    pre = conv.name.rsplit("reparam_conv/Conv", 1)[0]
    names = [pre + "reparam_conv/Conv_output_0_QuantizeLinear", pre + "reparam_conv/Conv_output_0_DequantizeLinear",
             pre + "activation/Div", pre + "activation/Erf",
             pre + "activation/Erf_output_0_QuantizeLinear", pre + "activation/Erf_output_0_DequantizeLinear",
             pre + "activation/Add", pre + "activation/Add_output_0_QuantizeLinear", pre + "activation/Add_output_0_DequantizeLinear",
             pre + "activation/Mul", pre + "activation/Mul_output_0_QuantizeLinear", pre + "activation/Mul_output_0_DequantizeLinear",
             pre + "activation/Mul_1", pre + "activation/Mul_1_output_0_QuantizeLinear", pre + "activation/Mul_1_output_0_DequantizeLinear"]
    byname = {n.name: n for n in g.node}
    return [byname[x] for x in names]


def slice_param(dq_node, k, tag):
    """DQ node of a per-channel int8 weight or int32 bias -> new initializers + new DQ node for channel group k"""
    q, s, z = (ini[i] for i in dq_node.input)
    qa, sa, za = nh.to_array(q), nh.to_array(s), nh.to_array(z)
    sl = slice(k * G, (k + 1) * G)
    nq = nh.from_array(np.ascontiguousarray(qa[sl]), f"{q.name}_g{k}of{K}")
    ns = nh.from_array(np.ascontiguousarray(sa[sl]), f"{s.name}_g{k}of{K}")
    nz = nh.from_array(np.ascontiguousarray(za[sl]), f"{z.name}_g{k}of{K}")
    g.initializer.extend([nq, ns, nz])
    out = f"{dq_node.output[0]}_g{k}of{K}"
    axis = [a for a in dq_node.attribute if a.name == "axis"]
    node = h.make_node("DequantizeLinear", [nq.name, ns.name, nz.name], [out], name=f"{dq_node.name}_g{k}of{K}",
                       **({"axis": axis[0].i} if axis else {}))
    return node, out


def clone_block(conv, k, x_in, group_attr):
    """conv k-th channel group + its GELU chain; returns (nodes, name of the block-output DQ tensor)"""
    new = []
    wdq, bdq = prod[conv.input[1]], prod[conv.input[2]]
    wn, wout = slice_param(wdq, k, "w"); bn, bout = slice_param(bdq, k, "b")
    new += [wn, bn]
    ren = {}
    sfx = f"_g{k}of{K}"
    cout = conv.output[0] + sfx
    attrs = {a.name: h.get_attribute_value(a) for a in conv.attribute}
    if group_attr is not None:
        attrs["group"] = group_attr
    new.append(h.make_node("Conv", [x_in, wout, bout], [cout], name=conv.name + sfx, **attrs))
    ren[conv.output[0]] = cout
    for n in chain_nodes(conv):
        ins = [ren.get(i, i) for i in n.input]
        outs = [o + sfx for o in n.output]
        for o, o2 in zip(n.output, outs):
            ren[o] = o2
        nn = h.make_node(n.op_type, ins, outs, name=n.name + sfx)
        nn.attribute.extend(copy.deepcopy(list(n.attribute)))
        new.append(nn)
    last = chain_nodes(conv)[-1].output[0]
    return new, ren[last]


x0 = conv0.input[0]                      # input_DequantizeLinear_Output
new_nodes, outs = [], []
for k in range(K):
    n0, y0 = clone_block(conv0, k, x0, None)
    n1, y1 = clone_block(conv1, k, y0, G)  # depthwise: group = channels of the group
    new_nodes += n0 + n1
    outs.append(y1)
# concat the K block outputs, requantize with the SAME scale/zp as the original patch_embed.1 output
q1 = chain_nodes(conv1)[-2]              # Mul_1_output_0_QuantizeLinear of patch_embed.1
cat = P1 + f"split{K}/concat"
new_nodes.append(h.make_node("Concat", outs, [cat + "_out"], name=cat, axis=1))
new_nodes.append(h.make_node("QuantizeLinear", [cat + "_out", q1.input[1], q1.input[2]], [cat + "_q"], name=cat + "_Q"))
new_nodes.append(h.make_node("DequantizeLinear", [cat + "_q", q1.input[1], q1.input[2]], [cat + "_dq"], name=cat + "_DQ"))
# rewire patch_embed.2 conv
old_in = conv2.input[0]
conv2.input[0] = cat + "_dq"
# remove the original stem nodes (conv0 + chain, conv1 + chain and the 4 original weight/bias DQ nodes)
remove = [conv0] + chain_nodes(conv0) + [conv1] + chain_nodes(conv1)
remove += [prod[conv0.input[1]], prod[conv0.input[2]], prod[conv1.input[1]], prod[conv1.input[2]]]
for n in remove:
    g.node.remove(n)
idx = list(g.node).index(prod[x0]) + 1          # right after the input DequantizeLinear
for j, n in enumerate(new_nodes):
    g.node.insert(idx + j, n)
# drop unused initializers
used = {i for n in g.node for i in n.input}
for t in [t for t in g.initializer if t.name not in used]:
    g.initializer.remove(t)
onnx.checker.check_model(m)
onnx.save(m, dst)

# ---- checks: A2 rule and ORT equality ----
mm = onnx.load(dst); I2 = {t.name: nh.to_array(t) for t in mm.graph.initializer}
a2 = {}
for b in "01":
    for t in ("Softmax_output_0", "Mul_output_0"):
        nm = f"/model/network.7/network.7.{b}/token_mixer/{t}_QuantizeLinear"
        qn = next(n for n in mm.graph.node if n.name == nm)
        a2[nm] = (float(I2[qn.input[1]]), int(I2[qn.input[2]]))
import onnxruntime as ort
res = {"src": src, "dst": dst, "K": K, "a2_zero_points": a2, "a2_ok": all(v[1] == 0 for v in a2.values())}
x = np.load(os.environ.get("MCLIP_ROOT", ".") + r"\calib_inputs.npy")[:4]
rng = np.random.default_rng(0)
xs = [x[i:i + 1] for i in range(4)] + [rng.random((1, 3, 256, 256), dtype=np.float32)]
for label, opt in (("default", None), ("opt_off", ort.GraphOptimizationLevel.ORT_DISABLE_ALL)):
    so = ort.SessionOptions(); so.intra_op_num_threads = 3
    if opt is not None:
        so.graph_optimization_level = opt
    sa = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(dst, so, providers=["CPUExecutionProvider"])
    md = 0.0
    for xi in xs:
        ea = sa.run(None, {"input": xi})[0]; eb = sb.run(None, {"input": xi})[0]
        md = max(md, float(np.abs(ea - eb).max()))
    res[f"max_abs_diff_{label}"] = md
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
res["src_sha256"], res["dst_sha256"] = sha(src), sha(dst)
res["nodes_src"], res["nodes_dst"] = len(onnx.load(src).graph.node), len(mm.graph.node)
print(json.dumps(res, indent=1))
json.dump(res, open(dst.replace(".onnx", "_info.json"), "w"), indent=1)
