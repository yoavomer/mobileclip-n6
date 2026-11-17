"""Check S2: the generic softmax final-stage LUT of boardlike.py (mul -> mul_sub2 -> sub2, constants derived from the output Q)
equals the compiled units (hwlib.arith with the trace constants) on the full (e, inv) grid, for the 3 compiles, both blocks.
Also: whole boardlike softmax ONNX subgraph == hwlib.hw_softmax (compiled tables) on the board logits.  -> check_softmax_lut.json"""
import json, numpy as np, onnx
from onnx import helper as h, TensorProto as TP
import hwlib, boardlike as BL
from common import R, HERE, sess
WS = {"qkvc": (str(R / "gws_qkvc/neural_art__network"), 1 / 255, -128), "a2": (str(R / "baseline/gws_a2/neural_art__network"), 1 / 127, 0),
      "p9995": (str(R / "quant/layers/gws_p9995/neural_art__network"), 1 / 127, 0)}
res = {}
g = np.arange(-128, 128, dtype=np.int64); E, I = np.meshgrid(g, g, indexing="ij")
for name, (ws, so, zo) in WS.items():
    for blk in ("742", "811"):
        _, fm = hwlib.unit(ws, f"Softmax_{blk}_mul_expansion"); _, fm2 = hwlib.unit(ws, f"Softmax_{blk}_mul_expansion_mul_sub2_")
        _, fs2 = hwlib.unit(ws, f"Softmax_{blk}_sub2_expansion")
        u = hwlib.arith(ws, fm, E, I, rmode="half_away"); u = hwlib.arith(ws, fm2, u, I, rmode="half_away"); hw = hwlib.arith(ws, fs2, u, I, rmode="half_away")
        mine = BL.softmax_final_lut(so, zo).reshape(256, 256).astype(np.int64)
        valid = (E <= 0) & (I < 0)          # reachable region: exp table outputs <= 0, inverse table outputs < 0
        res[f"{name} {blk}"] = {"params": BL.softmax_final_params(so, zo), "equal_full_grid": bool(np.array_equal(hw, mine)),
                                "n_diff_reachable": int(np.count_nonzero((hw != mine) & valid))}
        print(name, blk, res[f"{name} {blk}"])
# ONNX subgraph vs numpy compiled-table emulation on board logits (qkvconv model, 7.0)
s_in, z_in = 0.09573712199926376, 2
for k in ("fixed", "q_0"):
    lg = np.load(R / f"diag/board_qk_{k}.npy").reshape(1, 16, 64, 64)
    ref, _ = hwlib.hw_softmax(WS["qkvc"][0], lg, "742", "half_away")
    gr = h.make_graph([h.make_node("DequantizeLinear", ["x", "s", "z"], ["xf"]), h.make_node("Softmax", ["xf"], ["p"], axis=-1),
                       h.make_node("QuantizeLinear", ["p", "so", "zo"], ["y"])], "sm",
                      [h.make_tensor_value_info("x", TP.INT8, [1, 16, 64, 64])], [h.make_tensor_value_info("y", TP.INT8, [1, 16, 64, 64])],
                      [h.make_tensor("s", TP.FLOAT, [], [s_in]), h.make_tensor("z", TP.INT8, [], [z_in]),
                       h.make_tensor("so", TP.FLOAT, [], [1 / 255]), h.make_tensor("zo", TP.INT8, [], [-128])])
    m = h.make_model(gr, opset_imports=[h.make_opsetid("", 17)]); m.ir_version = 8
    onnx.save(m, str(HERE / "models/_sm_test.onnx"))
    BL.make_boardlike(HERE / "models/_sm_test.onnx", HERE / "models/_sm_test_bl.onnx")
    for opt in (True, False):
        y = sess(HERE / "models/_sm_test_bl.onnx", opt).run(None, {"x": lg})[0]
        res[f"onnx_vs_numpy {k} opt={opt}"] = int(np.count_nonzero(y.astype(np.int64) != ref))
        print(k, "ONNX boardlike softmax vs numpy compiled emulation: n_diff", res[f"onnx_vs_numpy {k} opt={opt}"])
json.dump(res, open(HERE / "check_softmax_lut.json", "w"), indent=1)
