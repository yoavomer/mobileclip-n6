"""Check G2: the float32 GELU chain written by make_boardlike(gelu_impl='float') == the float64 ideal LUT on all 256 int8 input
codes, for every GELU of the given model, in ORT default / disable_quant_qdq / opt-off. Micro-model per GELU:
int8[256] codes -> DQ(s_in,z_in) -> Div sqrt2 -> Erf -> Add 1 -> Mul x -> Mul 0.5 -> Q(s_o,z_o).  -> printed + check_gelu_float_<tag>.json"""
import sys, json, numpy as np, onnx, onnxruntime as ort
from onnx import helper as h, TensorProto as TP
import boardlike as BL
from common import HERE
src, tag = sys.argv[1], sys.argv[2]
rep = BL.make_boardlike(src, HERE / "models" / f"_chk_{tag}.onnx")
codes = np.arange(-128, 128, dtype=np.int8).reshape(1, 256)
def mk(mode):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    if mode == "noqdq": so.add_session_config_entry("session.disable_quant_qdq", "1")
    if mode == "off": so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return so
res = []
for gg in rep["gelu"]:
    (si, zi), (so_, zo) = gg["in_q"], gg["out_q"]
    G = BL._G(onnx.ModelProto())
    nodes = [h.make_node("DequantizeLinear", ["x", "si", "zi"], ["xf"])]
    gr = h.make_graph(nodes, "g", [h.make_tensor_value_info("x", TP.INT8, [1, 256])], [h.make_tensor_value_info("y", TP.INT8, [1, 256])],
                      [h.make_tensor("si", TP.FLOAT, [], [si]), h.make_tensor("zi", TP.INT8, [], [zi])])
    m = h.make_model(gr, opset_imports=[h.make_opsetid("", 17)]); m.ir_version = 8
    G = BL._G(m); m.graph.node.extend(BL._gelu_float_nodes(G, "xf", so_, zo, "y"))
    lut = BL.gelu_lut(si, zi, so_, zo).astype(np.int64)
    r = {"erf": gg["erf"]}
    for mode in ("default", "noqdq", "off"):
        y = ort.InferenceSession(m.SerializeToString(), mk(mode), providers=["CPUExecutionProvider"]).run(None, {"x": codes})[0].ravel().astype(np.int64)
        r[mode] = int(np.count_nonzero(y != lut))
    res.append(r)
print(tag, "GELUs:", len(res), "| codes differing from the float64 LUT (sum over GELUs): " + ", ".join(f"{m} {sum(r[m] for r in res)}" for m in ("default", "noqdq", "off")))
for r in res:
    if r["default"] or r["noqdq"] or r["off"]: print("  ", r)
json.dump(res, open(HERE / f"check_gelu_float_{tag}.json", "w"), indent=1)
