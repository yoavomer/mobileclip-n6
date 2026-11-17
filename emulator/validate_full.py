"""Experiment V3: full models on the board inputs whose board results are known only as summary numbers:
CRC32 + first 8 int8 values + cos(board, ORT default / opt-off) + std + saturation + cos(board, FP32)
(quant/int16/board_p9995_{A,B}.txt, board_a2_{A,B}.txt; baseline/board_check_run.txt has first16 for a2 fixed).
Inputs rebuilt exactly as quant/int16/make_board2.py: fixed pattern; first 3 of quant/sweep/test200.npy quantized with the
model's input scale; FP32 ref = mci0_256_fp32.onnx on the float input (fixed) / on the original float image (photos).
For each: plain ORT (default, opt-off) and board-like (ORT_ENABLE_ALL = what zs_eval uses; ORT_ENABLE_EXTENDED = no NCHWc).
Also a bound on the unknown cos(board-like, board) from the three known board cosines (projection onto span{ORT def, ORT off, FP32}).
-> validate_full.json"""
import json, re, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh
from common import R, HERE, stats, crc32
from boardlike import make_boardlike
M = HERE / "models"


def S(p, level):
    so = ort.SessionOptions(); so.intra_op_num_threads = 3; so.graph_optimization_level = level
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


L = ort.GraphOptimizationLevel
def parse_log(path):
    txt = open(path).read(); out = {}
    for m in re.finditer(r"C (\S+) t=\d+ us crc32=0x([0-9a-f]+) first8=\[([^\]]*)\]", txt):
        k = m.group(1).replace(".jpg", "")
        if k in out: continue
        d = re.search(rf"D {re.escape(m.group(1))} vs ORT-default: cos=([\d.]+) maxdiff=(\d+) std board=([\d.]+) ORT=([\d.]+) ratio=([\d.]+) sat board=(\d+)/(\d+)", txt)
        o = re.search(rf"D {re.escape(m.group(1))} vs ORT-opt_off: cos=([\d.]+) maxdiff=(\d+)", txt)
        f = re.search(rf"F {re.escape(m.group(1))} board vs FP32: cos=([\d.]+)", txt)
        out[k] = {"crc": int(m.group(2), 16), "first8": [int(v) for v in m.group(3).split(",")], "cos_def": float(d.group(1)),
                  "max_def": int(d.group(2)), "std": float(d.group(3)), "ort_std": float(d.group(4)), "sat": [int(d.group(6)), int(d.group(7))],
                  "cos_off": float(o.group(1)), "max_off": int(o.group(2)), "cos_fp32": float(f.group(1)) if f else None}
    return out


def unit(v): v = np.asarray(v, np.float64); return v / np.linalg.norm(v)


def bound(e, us, cs):
    """cos(e, b) bounds given b.u_k = cs[k] for unit b; e, us arbitrary vectors."""
    U = np.stack([unit(u) for u in us], 1); c = np.asarray(cs); ehat = unit(e)
    Gm = U.T @ U; Gi = np.linalg.inv(Gm)
    p2 = float(c @ Gi @ c); ue = U.T @ ehat
    proj = float(c @ Gi @ ue); e2 = float(ue @ Gi @ ue)
    w = np.sqrt(max(0.0, 1 - p2)) * np.sqrt(max(0.0, 1 - e2))
    return {"low": proj - w, "high": proj + w, "board_in_span": p2, "emu_in_span": e2}


T = np.load(R / "quant/sweep/test200.npy")[:3]; tn = (R / "quant/sweep/test200.txt").read_text().split()[:3]
fixed = (((37 * np.arange(3 * 256 * 256) + 11) % 256) - 128).astype(np.int8)
fp32 = S(R / "mci0_256_fp32.onnx", L.ORT_ENABLE_ALL)
res = {}
for name, src, logs in (("P99.95", R / "quant/layers/mci0_256_p9995.onnx", ["quant/int16/board_p9995_A.txt", "quant/int16/board_p9995_B.txt"]),
                        ("mci0_256_a2", R / "baseline/mci0_256_a2.onnx", ["quant/int16/board_a2_A.txt", "quant/int16/board_a2_B.txt"])):
    mm = onnx.load(str(src)); ini = {x.name: nh.to_array(x) for x in mm.graph.initializer}
    dq = next(n for n in mm.graph.node if mm.graph.output[0].name in n.output)
    qi = next(n for n in mm.graph.node if n.op_type == "QuantizeLinear" and n.input[0] == mm.graph.input[0].name)
    S_IN = np.float32(ini[qi.input[1]]); S_OUT, Z = float(ini[dq.input[1]]), int(ini[dq.input[2]])
    board = {}
    for lg in logs: board.update(parse_log(R / lg))
    bl = M / f"full_{name}_bl.onnx"; rep = make_boardlike(src, bl)
    sess = {"ORT_default": S(src, L.ORT_ENABLE_ALL), "ORT_opt_off": S(src, L.ORT_DISABLE_ALL),
            "BL_all": S(bl, L.ORT_ENABLE_ALL), "BL_ext": S(bl, L.ORT_ENABLE_EXTENDED)}
    inputs = {"fixed": ((fixed.astype(np.float32) + 128) * S_IN).reshape(1, 3, 256, 256)}
    fpin = {"fixed": inputs["fixed"]}
    for i, nm in enumerate(tn):
        q = np.clip(np.rint(T[i].ravel() / S_IN) - 128, -128, 127).astype(np.int8)
        k = nm.replace(".jpg", ""); inputs[k] = ((q.astype(np.float32) + 128) * S_IN).reshape(1, 3, 256, 256); fpin[k] = T[i][None]
    for k, x in inputs.items():
        y = {n: np.clip(np.rint(s.run(None, {"input": x})[0].ravel() / S_OUT) + Z, -128, 127).astype(np.int64) for n, s in sess.items()}
        f = fp32.run(None, {"input": fpin[k]})[0].ravel().astype(np.float64)
        b = board[k]; r = {"board": b}
        # sanity: our plain ORT reproduces the board log's ORT std / board-vs-ORT relation inputs
        r["ORT_default_std_check"] = [float(y["ORT_default"].std()), b["ort_std"]]
        for n, v in y.items():
            c = lambda a, bb: float((a - Z) @ (bb - Z) / np.linalg.norm(a - Z) / np.linalg.norm(bb - Z))
            r[n] = {"crc": crc32(v), "crc_equal_board": crc32(v) == b["crc"], "first8": v[:8].tolist(),
                    "first8_exact": int(np.sum(v[:8] == np.array(b["first8"]))), "first8_maxdiff": int(np.abs(v[:8] - np.array(b["first8"])).max()),
                    "cos_vs_ORT_default": c(v, y["ORT_default"]), "cos_vs_ORT_opt_off": c(v, y["ORT_opt_off"]),
                    "cos_vs_FP32": float(unit(v - Z) @ unit(f)), "std": float(v.std()), "sat": [int((v == -128).sum()), int((v == 127).sum())],
                    "cos_board_bound": bound(v - Z, [y["ORT_default"] - Z, y["ORT_opt_off"] - Z, f], [b["cos_def"], b["cos_off"], b["cos_fp32"]])}
        res[f"{name} {k}"] = r
        bb = r["BL_all"]["cos_board_bound"]; be = r["BL_ext"]["cos_board_bound"]; bo = r["ORT_default"]["cos_board_bound"]
        print(f"{name:11s} {k:13s} board: CRC {b['crc']:08x} first8 {b['first8']} cos(b,ORTdef) {b['cos_def']:.4f} cos(b,off) {b['cos_off']:.4f} cos(b,FP32) {b['cos_fp32']}")
        for n in sess:
            rr = r[n]; bd = rr["cos_board_bound"]
            print(f"   {n:12s} CRC {rr['crc']:08x} {'EQUAL' if rr['crc_equal_board'] else 'diff '} first8 {rr['first8']} ({rr['first8_exact']}/8 exact, max {rr['first8_maxdiff']}) "
                  f"cos(.,ORTdef) {rr['cos_vs_ORT_default']:.4f} cos(.,off) {rr['cos_vs_ORT_opt_off']:.4f} cos(.,FP32) {rr['cos_vs_FP32']:.4f} std {rr['std']:.2f} (board {b['std']:.2f}) "
                  f"| cos(.,board) in [{bd['low']:.3f}, {bd['high']:.3f}]", flush=True)
json.dump(res, open(HERE / "validate_full.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
