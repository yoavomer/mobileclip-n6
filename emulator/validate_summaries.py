"""Experiment V4: board runs of the pre-A2 qkvconv model for which only SUMMARIES exist (no full dump):
 - bisect cuts s0 / s1 / s2 (end of stage 0/1/2; GELUs of the stem and of stages 0-2 inside): evidence/bisect_run_cut_s*.txt
   (CRC, first16, board mean/std, cos vs ORT default / opt-off)  — inputs fixed and q_0
 - full qkvconv model on the constant inputs q=-128 / 0 / 77: evidence/factcheck_board_run.txt (CRC + first16 only)
For each: plain ORT default/opt-off and board-like (ORT_ENABLE_ALL, ORT_ENABLE_EXTENDED): first16 exact/maxdiff, CRC equal?,
cos(., ORT) next to the board's, mean/std next to the board's; for the constants cos over the 16 values.  -> validate_summaries.json"""
import json, re, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh
from common import R, HERE, crc32, FIXED_Q
from boardlike import make_boardlike
M = HERE / "models"; L = ort.GraphOptimizationLevel


def S(p, lv):
    so = ort.SessionOptions(); so.intra_op_num_threads = 3; so.graph_optimization_level = lv
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def cosz(a, b, z): a = np.asarray(a, float) - z; b = np.asarray(b, float) - z; return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


res = {}
for cut in ("s0", "s1", "s2"):
    txt = open(R / f"evidence/bisect_run_cut_{cut}.txt").read()
    src = R / f"bisect/cut_{cut}_int8.onnx"; bl = M / f"cut_{cut}_bl.onnx"; make_boardlike(src, bl)
    m = onnx.load(str(src)); ini = {t.name: nh.to_array(t) for t in m.graph.initializer}
    z = int(ini[next(n for n in m.graph.node if n.output[0] == m.graph.output[0].name).input[2]])
    sess = {"ORT_default": S(src, L.ORT_ENABLE_ALL), "ORT_opt_off": S(src, L.ORT_DISABLE_ALL), "BL_all": S(bl, L.ORT_ENABLE_ALL), "BL_ext": S(bl, L.ORT_ENABLE_EXTENDED)}
    for k, q in (("fixed", FIXED_Q), ("q_0", np.zeros_like(FIXED_Q))):
        c = re.search(rf"C {k} t=\d+ us len=\d+ crc32=0x([0-9a-f]+) first16=\[([^\]]*)\]", txt)
        st = re.search(rf"C {k} board mean=(-?[\d.]+) std=([\d.]+)", txt)
        d = re.search(rf"D {k} vs ORT-default: cos=([\d.]+)", txt); o = re.search(rf"D {k} vs ORT-opt_off: cos=([\d.]+)", txt)
        b = {"crc": int(c.group(1), 16), "first16": [int(v) for v in c.group(2).split(",")], "mean": float(st.group(1)), "std": float(st.group(2)),
             "cos_def": float(d.group(1)), "cos_off": float(o.group(1))}
        x = ((q.astype(np.float32) + 128) * np.float32(0.003921568859368563)).reshape(1, 3, 256, 256)
        y = {n: s.run(None, {"input": x})[0].ravel().astype(np.int64) for n, s in sess.items()}
        r = {"board": b}
        for n, v in y.items():
            f16 = np.array(b["first16"])
            r[n] = {"crc_equal": crc32(v) == b["crc"], "first16_exact": int((v[:16] == f16).sum()), "first16_maxdiff": int(np.abs(v[:16] - f16).max()),
                    "first16_mad": float(np.abs(v[:16] - f16).mean()),
                    "cos_vs_ORT_default": cosz(v, y["ORT_default"], z), "cos_vs_ORT_opt_off": cosz(v, y["ORT_opt_off"], z), "mean": float(v.mean()), "std": float(v.std())}
        res[f"cut_{cut} {k}"] = r
        print(f"cut_{cut} {k:5s} board: first16 {b['first16'][:8]}.. mean {b['mean']:.3f} std {b['std']:.3f} cos(b,ORTdef) {b['cos_def']:.4f} cos(b,off) {b['cos_off']:.4f}")
        for n in sess:
            rr = r[n]
            print(f"   {n:12s} first16 exact {rr['first16_exact']:2d}/16 max {rr['first16_maxdiff']:2d} mean|d| {rr['first16_mad']:.2f} | cos(.,ORTdef) {rr['cos_vs_ORT_default']:.4f} "
                  f"cos(.,off) {rr['cos_vs_ORT_opt_off']:.4f} | mean {rr['mean']:.3f} std {rr['std']:.3f}  CRC {'EQUAL' if rr['crc_equal'] else 'diff'}", flush=True)
# full qkvconv, constant inputs, first16 only
txt = open(R / "evidence/factcheck_board_run.txt", encoding="utf-8", errors="replace").read()
src = R / "mci0_256_qkvconv.onnx"; bl = M / "qkvconv_bl.onnx"; make_boardlike(src, bl)
sess = {"ORT_default": S(src, L.ORT_ENABLE_ALL), "ORT_opt_off": S(src, L.ORT_DISABLE_ALL), "BL_all": S(bl, L.ORT_ENABLE_ALL), "BL_ext": S(bl, L.ORT_ENABLE_EXTENDED)}
for tag, qv in (("q_m128", -128), ("q_0", 0), ("q_77", 77)):
    c = re.search(rf"G{tag} OUT[^\n]*crc32=0x([0-9a-f]+) first16=\[([^\]]*)\]", txt)
    b16 = np.array([int(v) for v in c.group(2).split(",")]); crc = int(c.group(1), 16)
    x = ((np.full(FIXED_Q.shape, qv, np.float32) + 128) * np.float32(0.003921568859368563)).reshape(1, 3, 256, 256)
    r = {"board_first16": b16.tolist()}
    line = f"qkvconv {tag:6s} board first16 {b16[:8].tolist()}.. |"
    for n, s in sess.items():
        e = np.clip(np.rint(s.run(None, {"input": x})[0].ravel() / 0.003208649344742298) + 6, -128, 127).astype(np.int64)
        r[n] = {"crc_equal": crc32(e) == crc, "cos16": cosz(e[:16], b16, 6), "exact16": int((e[:16] == b16).sum()), "max16": int(np.abs(e[:16] - b16).max())}
        line += f" {n} cos16 {r[n]['cos16']:+.3f} exact {r[n]['exact16']}/16 max {r[n]['max16']} |"
    res[f"qkvconv {tag}"] = r; print(line, flush=True)
json.dump(res, open(HERE / "validate_summaries.json", "w"), indent=1)
