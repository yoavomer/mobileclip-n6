"""mci0_256_qkvconv: the full INT8 model mci0_256_int8.onnx with the qkv projection of both attention blocks
(network.7.0 / 7.1) rewritten as Conv 1x1 on [1,512,8,8], exactly as split/make_qkvconv.py did for part B:
same int8 weights (transposed to [1536,512,1,1]), same per-channel scales/zps (axis 0), Conv -> Q -> DQ ->
Reshape[1,1536,64] -> Q/DQ -> Transpose[0,2,1] -> Q with the SAME output tensor name/scale/zp as the original qkv Q.
Nothing else in the graph changes.
Check: ORT vs mci0_256_int8 on 8 calibration images, with ORT default kernels and with optimizations disabled."""
import pathlib, collections, numpy as np, onnx, onnxruntime as ort
from onnx import helper as h, numpy_helper as nh

D = pathlib.Path(__file__).parent


def rewrite_int8(m):           # copied verbatim from split/make_qkvconv.py (that module rebuilds files on import)
    g = m.graph
    init = {t.name: t for t in g.initializer}
    for b in ("7.0", "7.1"):
        P = f"/model/network.7/network.{b}/"
        bnq = P + "norm/BatchNormalization_output_0_QuantizeLinear_Output"
        s_bn, z_bn = P + "norm/BatchNormalization_output_0_scale", P + "norm/BatchNormalization_output_0_zero_point"
        mm = next(n for n in g.node if n.name == P + "token_mixer/qkv/MatMul")
        wdq = next(n for n in g.node if mm.input[1] in n.output)
        qn = next(n for n in g.node if mm.output[0] in n.input and n.op_type == "QuantizeLinear")
        qout, s_out, z_out = qn.output[0], qn.input[1], qn.input[2]
        w_q, w_s, w_z = (init[i] for i in wdq.input)
        wt = nh.from_array(nh.to_array(w_q).T.reshape(1536, 512, 1, 1).copy(), w_q.name + "_conv")
        shp = nh.from_array(np.array([1, 1536, 64], np.int64), P + "qkvconv_shape")
        g.initializer.extend([wt, shp])
        t = lambda s: P + "qkvconv/" + s
        new = [
            h.make_node("DequantizeLinear", [bnq, s_bn, z_bn], [t("x")], name=t("dq_x")),
            h.make_node("DequantizeLinear", [wt.name, w_s.name, w_z.name], [t("w")], name=t("dq_w"), axis=0),
            h.make_node("Conv", [t("x"), t("w")], [t("conv")], name=P + "token_mixer/qkv/Conv", kernel_shape=[1, 1]),
            h.make_node("QuantizeLinear", [t("conv"), s_out, z_out], [t("conv_q")], name=t("q_conv")),
            h.make_node("DequantizeLinear", [t("conv_q"), s_out, z_out], [t("conv_dq")], name=t("dq_conv")),
            h.make_node("Reshape", [t("conv_dq"), shp.name], [t("r")], name=t("reshape")),
            h.make_node("QuantizeLinear", [t("r"), s_out, z_out], [t("r_q")], name=t("q_r")),
            h.make_node("DequantizeLinear", [t("r_q"), s_out, z_out], [t("r_dq")], name=t("dq_r")),
            h.make_node("Transpose", [t("r_dq")], [t("tr")], name=t("transpose"), perm=[0, 2, 1]),
            h.make_node("QuantizeLinear", [t("tr"), s_out, z_out], [qout], name=qn.name),
        ]
        idx = list(g.node).index(qn)
        g.node.remove(qn)
        for k, n in enumerate(new):
            g.node.insert(idx + k, n)
    return m


def clean_and_save(m, path):
    g = m.graph
    while True:
        used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
        dead = [n for n in g.node if not any(o in used for o in n.output)]
        if not dead: break
        for n in dead: g.node.remove(n)
    used = {i for n in g.node for i in n.input}
    for t in [t for t in g.initializer if t.name not in used]: g.initializer.remove(t)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


if __name__ == "__main__":
    SRC, DST = D / "mci0_256_int8.onnx", D / "mci0_256_qkvconv.onnx"
    clean_and_save(rewrite_int8(onnx.load(str(SRC))), DST)
    for p in (SRC, DST):
        c = collections.Counter(n.op_type for n in onnx.load(str(p)).graph.node)
        print(f"{p.name:26s} nodes {sum(c.values()):5d}  Conv {c['Conv']:3d}  MatMul {c['MatMul']:2d}  Gemm {c['Gemm']}  Reshape {c['Reshape']:2d}  Transpose {c['Transpose']:2d}")
    x = np.load(D / "calib_inputs.npy")[:8]
    for label, opt in (("ORT default kernels", None), ("ORT optimizations off", ort.GraphOptimizationLevel.ORT_DISABLE_ALL)):
        so = ort.SessionOptions()
        if opt is not None: so.graph_optimization_level = opt
        a = ort.InferenceSession(str(SRC), so, providers=["CPUExecutionProvider"])
        b = ort.InferenceSession(str(DST), so, providers=["CPUExecutionProvider"])
        md, cs, nflip = 0.0, [], 0
        for i in range(len(x)):
            ea = a.run(None, {"input": x[i:i + 1]})[0].astype(np.float64).ravel()
            eb = b.run(None, {"input": x[i:i + 1]})[0].astype(np.float64).ravel()
            md = max(md, float(np.abs(ea - eb).max())); nflip += int((ea != eb).sum())
            cs.append(float(ea @ eb / np.linalg.norm(ea) / np.linalg.norm(eb)))
        print(f"{label:24s}: max|int8 - qkvconv| = {md:.6g} ; elements that differ {nflip}/{8 * 512} ; cosine min {min(cs):.9f} mean {np.mean(cs):.9f}")
