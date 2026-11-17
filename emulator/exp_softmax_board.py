"""Experiment S1: is the compiler analyst's softmax-expansion emulation closer to the BOARD than ORT's Softmax?
Never validated against the board before (the analyst compared it with ORT only).
Data: diag/board_qk_{fixed,q_0}.npy (board Q.K^T output of cut ops_70_qk, 7.0 block, int8 [1,16,64,64], scale 0.0957371 zp 2)
      diag/board_sm_{fixed,q_0}.npy (board softmax output of cut ops_70_softmax, int8 [1,16,64,64], 1/255, zp -128)
Both cuts of mci0_256_qkvconv.onnx start at the model input with the same inputs, and the board was repeat-identical,
so the logits inside the softmax cut equal the qk cut's output (assumption; checked indirectly: Q.K^T is a pure
function of the board's qkv, which matched between cuts at max 0 LSB in the conditional checks).
Also: generic table generation (exp / inverse LUT from formulas) vs the compiled tables of 3 compiles.
Output: exp_softmax_board.json + printed table."""
import json, math, pathlib, numpy as np, onnxruntime as ort, onnx
from onnx import helper as h, TensorProto as TP
import hwlib
R = pathlib.Path(__file__).resolve().parents[2]; D = R / "diag"; HERE = pathlib.Path(__file__).resolve().parent
WS = {"qkvc": str(R / "gws_qkvc/neural_art__network"), "a2": str(R / "baseline/gws_a2/neural_art__network"),
      "p9995": str(R / "quant/layers/gws_p9995/neural_art__network")}
QJ = {"qkvc": R / "gout_qkvc/mci0_256_qkvconv_OE_3_3_1_Q.json", "a2": R / "baseline/gout_a2/mci0_256_a2_OE_3_3_1_Q.json",
      "p9995": R / "quant/layers/gout_p9995/mci0_256_p9995_OE_3_3_1_Q.json"}
res = {}


def stats(a, b, z):
    a = a.astype(np.int64).ravel(); b = b.astype(np.int64).ravel(); d = np.abs(a - b)
    x, y = a - z, b - z
    return {"cos": round(float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)), 5), "maxdiff": int(d.max()),
            "exact_pct": round(float((d == 0).mean() * 100), 2), "within1_pct": round(float((d <= 1).mean() * 100), 2)}


# ORT softmax on int8 logits: DQ -> Softmax -> Q (1/255, -128), both ORT modes
def ort_softmax(lg, s_in, z_in, s_o, z_o, opt):
    g = h.make_graph([h.make_node("DequantizeLinear", ["x", "s", "z"], ["xf"]), h.make_node("Softmax", ["xf"], ["p"], axis=-1),
                      h.make_node("QuantizeLinear", ["p", "so", "zo"], ["y"])], "sm",
                     [h.make_tensor_value_info("x", TP.INT8, lg.shape)], [h.make_tensor_value_info("y", TP.INT8, lg.shape)],
                     [h.make_tensor("s", TP.FLOAT, [], [s_in]), h.make_tensor("z", TP.INT8, [], [z_in]),
                      h.make_tensor("so", TP.FLOAT, [], [s_o]), h.make_tensor("zo", TP.INT8, [], [z_o])])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)]); m.ir_version = 8
    so = ort.SessionOptions(); so.intra_op_num_threads = 3
    if not opt: so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"]).run(None, {"x": lg})[0]


S_QK, Z_QK = 0.09573712199926376, 2          # [from report] Gemm_739 output (bisect/compiler/README.md, diag/qk_models.py)
for k in ("fixed", "q_0"):
    lg = np.load(D / f"board_qk_{k}.npy").reshape(1, 16, 64, 64)
    sm = np.load(D / f"board_sm_{k}.npy").reshape(1, 16, 64, 64)
    hw, inter = hwlib.hw_softmax(WS["qkvc"], lg, "742")
    r = {"hw_emulation_vs_board": stats(hw, sm, -128)}
    hw2, _ = hwlib.hw_softmax(WS["qkvc"], lg, "742", "half_away")      # S3: ties rounded away from zero in every unit
    r["hw_emulation_half_away_vs_board"] = stats(hw2, sm, -128)
    for opt in (True, False):
        o = ort_softmax(lg, S_QK, Z_QK, 1 / 255, -128, opt)
        r[f"ORT_{'default' if opt else 'opt_off'}_vs_board"] = stats(o, sm, -128)
    r["avg_range"] = [int(inter["avg"].min()), int(inter["avg"].max())]
    res[f"softmax 7.0 board_qk->board_sm {k}"] = r
    print(k, json.dumps(r))

# sub1 over the full grid: is it exactly d = x - max + 127 ?
g = np.arange(-128, 128); X, M = np.meshgrid(g, g, indexing="ij"); valid = X <= M
_, fsub = hwlib.unit(WS["p9995"], "Softmax_742_sub1_expansion")
d = hwlib.arith(WS["p9995"], fsub, X, M)
res["sub1_equals_x_minus_max_plus_127_on_valid_grid"] = bool(np.array_equal(d[valid], (X - M + 127)[valid]))
print("sub1 == x-max+127 (x<=max):", res["sub1_equals_x_minus_max_plus_127_on_valid_grid"])

# generic tables: exp LUT ~ -round(128*exp(s*(d-127))), inverse LUT ~ -round(256/(-avg))
for name, ws in WS.items():
    T, q = hwlib.qjson(QJ[name])
    for blk, gm in (("742", "Gemm_739_out_0"), ("811", "Gemm_808_out_0")):
        s_in, z_in = q(gm)
        lut = hwlib.activ_lut(ws, f"Softmax_{blk}_exp_expansion")
        dd = np.arange(-128, 128)
        cand = {"-round_half_even(128*exp(s*(d-127)))": -np.round(128 * np.exp(s_in * (dd - 127))),
                "-floor(128*exp+0.5)": -np.floor(128 * np.exp(s_in * (dd - 127)) + 0.5)}
        rr = {"s_in": s_in, "z_in": z_in}
        for cn, cv in cand.items():
            cv = np.clip(cv, -128, 127); diff = lut - cv
            rr[cn] = {"n_diff": int(np.count_nonzero(diff)), "max": int(np.abs(diff).max()), "at_d": dd[diff != 0].tolist()[:12]}
        res[f"exp_lut {name} {blk}"] = rr
        print(name, blk, json.dumps(rr))
    inv = hwlib.activ_lut(ws, "Softmax_742_inverse_expansion")
    a = np.arange(-128, 128)
    with np.errstate(divide="ignore"):
        cand = np.where(a < 0, -np.round(256.0 / np.maximum(-a, 1)), -128)
    cand = np.clip(cand, -128, 127); diff = inv - cand
    res[f"inv_lut {name}"] = {"n_diff_vs_-round(256/-avg)": int(np.count_nonzero(diff)), "at_avg": a[diff != 0].tolist()[:20],
                              "lut_at_-128..-120": inv[:9].tolist()}
    print(name, "inv", res[f"inv_lut {name}"])
json.dump(res, open(HERE / "exp_softmax_board.json", "w"), indent=1)
