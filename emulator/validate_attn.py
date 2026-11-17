"""Experiment V2: attention / stage cuts of the pre-A2 qkvconv model (mci0_256_qkvconv.onnx, MinMax, softmax 1/255 zp -128,
non-zero weight-operand zps) — board dumps diag/board_*.npy, inputs fixed and q_0 (all-zero int8).
(a) conditional segments: ORT fed with the board's own upstream int8 tensors (diag/seg_*.onnx + 2 new segments here)
(b) unconditional cuts from the model input (bisect/ops/ops_*_int8.onnx, bisect/cut_*_int8.onnx)
(c) full model, fixed input, all 512 values: factcheck/board_hex.txt
Each: plain ORT (default, opt-off) and make_boardlike (default, opt-off) vs board: cos on q-zp / max LSB / % exact.
-> validate_attn.json"""
import json, numpy as np, onnx
from onnx.utils import extract_model
from common import R, HERE, sess, stats, fmt, FIXED_Q
from boardlike import make_boardlike
D = R / "diag"; SRC = R / "mci0_256_qkvconv.onnx"; M = HERE / "models"
P7 = "/model/network.7/network.7.0/"
QKV = P7 + "qkvconv/conv_q"; QK = P7 + "token_mixer/MatMul_output_0_QuantizeLinear_Output"
SM = P7 + "token_mixer/Softmax_output_0_QuantizeLinear_Output"
for tag, ins, outs in (("qkv_to_qk", [QKV], [QK]), ("qk_to_sm", [QK], [SM])):
    if not (M / f"seg_{tag}.onnx").exists(): extract_model(str(SRC), str(M / f"seg_{tag}.onnx"), ins, outs)
src = onnx.load(str(SRC)); ini = {t.name: onnx.numpy_helper.to_array(t) for t in src.graph.initializer}
qz = {n.output[0]: int(ini[n.input[2]]) for n in src.graph.node if n.op_type == "QuantizeLinear"}
COND = [("qkv -> Q.K^T", M / "seg_qkv_to_qk.onnx", ["qkv"], "qk"),
        ("Q.K^T -> softmax", M / "seg_qk_to_sm.onnx", ["qk"], "sm"),
        ("qkv -> softmax", D / "seg_qkv_to_sm.onnx", ["qkv"], "sm"),
        ("qkv+softmax -> attn.V", D / "seg_qkvsm_to_av.onnx", ["qkv", "sm"], "av"),
        ("attn.V -> layer-scale Mul", D / "seg_av_to_ls.onnx", ["av"], "ls"),
        ("n6 -> qkv", D / "seg_n6_to_qkv.onnx", ["n6"], "qkv"),
        ("n6 -> 7.0 Add", D / "seg_n6_to_add.onnx", ["n6"], "add"),
        ("n6 -> a0 (end of 7.0)", D / "seg_n6_to_a0.onnx", ["n6"], "a0"),
        ("n6 -> s3 (end of stage 3)", D / "seg_ref_n6_to_s3.onnx", ["n6"], "s3")]
UNC = [("n6", R / "bisect/ops/ops_n6_int8.onnx", "n6"), ("qkv", R / "bisect/ops/ops_70_qkv_int8.onnx", "qkv"),
       ("Q.K^T", R / "bisect/ops/ops_70_qk_int8.onnx", "qk"), ("softmax", R / "bisect/ops/ops_70_softmax_int8.onnx", "sm"),
       ("attn.V", R / "bisect/ops/ops_70_attnv_int8.onnx", "av"), ("layer-scale Mul", R / "bisect/ops/ops_70_ls_int8.onnx", "ls"),
       ("7.0 Add", R / "bisect/ops/ops_70_add_int8.onnx", "add"), ("a0", R / "bisect/cut_a0_int8.onnx", "a0"),
       ("s3", R / "bisect/cut_s3_int8.onnx", "s3")]
INQ = {"fixed": FIXED_Q, "q_0": np.zeros_like(FIXED_Q)}
res = {}


def run_all(path, feeds_by_input, zp, board_by_input, label):
    bl = M / (path.stem + "_bl.onnx")
    rep = make_boardlike(path, bl)
    S = {"ORT_default": sess(path, True), "ORT_opt_off": sess(path, False), "BL_default": sess(bl, True), "BL_ext": sess(bl, "ext")}
    rr = {"rules": {"gelu": len(rep.get("gelu", [])), "softmax": sum("skipped" not in r for r in rep.get("softmax", [])),
                    "matmul_wzp_rewritten": sum(r["rewritten"] for r in rep.get("matmul_wzp", [])), "sigmoid": len(rep.get("sigmoid", []))}}
    for k, feed in feeds_by_input.items():
        b = board_by_input[k]
        y = {n: s.run(None, feed)[0].ravel() for n, s in S.items()}
        rr[k] = {f"{n}_vs_board": stats(v, b, zp) for n, v in y.items()}
        rr[k]["board_std"] = float(b.std()); rr[k]["BL_default_std"] = float(y["BL_default"].std()); rr[k]["ORT_default_std"] = float(y["ORT_default"].std())
        print(f"{label:30s} {k:5s} ORT {fmt(rr[k]['ORT_default_vs_board'])} (opt-off {fmt(rr[k]['ORT_opt_off_vs_board'])}) | "
              f"BL {fmt(rr[k]['BL_default_vs_board'])} (ext {fmt(rr[k]['BL_ext_vs_board'])})  rules {rr['rules']}", flush=True)
    return rr


print("== (a) conditional segments (board int8 in -> board int8 out)")
for label, path, ins, out in COND:
    m = onnx.load(str(path), load_external_data=False)
    shp = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in m.graph.input}
    names = [i.name for i in m.graph.input]
    feeds = {k: {n: np.load(D / f"board_{p}_{k}.npy").reshape(shp[n]) for n, p in zip(names, ins)} for k in INQ}
    board = {k: np.load(D / f"board_{out}_{k}.npy") for k in INQ}
    res["cond " + label] = run_all(path, feeds, qz[m.graph.output[0].name], board, "cond " + label)
print("== (b) cuts from the model input")
for label, path, out in UNC:
    m = onnx.load(str(path), load_external_data=False)
    feeds = {k: {"input": ((q.astype(np.float32) + 128) * np.float32(0.003921568859368563)).reshape(1, 3, 256, 256)} for k, q in INQ.items()}
    board = {k: np.load(D / f"board_{out}_{k}.npy") for k in INQ}
    res["cut " + label] = run_all(path, feeds, qz[m.graph.output[0].name], board, "cut " + label)
print("== (c) full qkvconv model, fixed input (factcheck/board_hex.txt, 512 values)")
be = np.frombuffer(bytes.fromhex(open(R / "factcheck/board_hex.txt").read().strip()), np.int8).astype(np.int64)
bl = M / "qkvconv_bl.onnx"; make_boardlike(SRC, bl)
x = ((FIXED_Q.astype(np.float32) + 128) * np.float32(0.003921568859368563)).reshape(1, 3, 256, 256)
rr = {}
for n, p, opt in (("ORT_default", SRC, True), ("ORT_opt_off", SRC, False), ("BL_default", bl, True), ("BL_ext", bl, "ext")):
    e = np.clip(np.rint(sess(p, opt).run(None, {"input": x})[0].ravel() / 0.003208649344742298) + 6, -128, 127)
    rr[n] = stats(e, be, 6); rr[n]["std"] = float(e.std()); rr[n]["sat"] = [int((e == -128).sum()), int((e == 127).sum())]
print(f"full qkvconv fixed: ORT {fmt(rr['ORT_default'])} (opt-off {fmt(rr['ORT_opt_off'])}) | BL {fmt(rr['BL_default'])} (ext {fmt(rr['BL_ext'])}); "
      f"std board {be.std():.2f} BL {rr['BL_default']['std']:.2f}; sat board {(be == -128).sum()}/{(be == 127).sum()} BL {rr['BL_default']['sat']}")
rr["board_std"] = float(be.std()); res["full qkvconv fixed"] = rr
json.dump(res, open(HERE / "validate_attn.json", "w"), indent=1)
