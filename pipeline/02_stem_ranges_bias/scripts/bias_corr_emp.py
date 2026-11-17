"""M4b (extension of M4): empirical bias correction of the WHOLE quantized model's mean conv-output error.
For every Conv of the candidate that has an int8-weight / int32-bias QDQ pattern:
   e[c] = mean over images, positions of (conv output of the BOARD-LIKE candidate, float, before its QuantizeLinear)
          - (conv output of the FP32 model)                                       [same tensor name in both models]
   new int32 bias[c] = old[c] - round(alpha * e[c] / bias_scale[c])
This includes the error from input quantization and from all upstream layers (M4 covers only W_q - W).
All convs are corrected at once; `iters` repeats measure -> correct on the corrected model.
Images: calib pool [off, off+N) (default 300..699, disjoint from the proxy images 0..299).
usage: python -I bias_corr_emp.py <src.onnx> <out.onnx> [iters=1] [alpha=1.0] [N=400] [off=300] [scope=all|stem]"""
import sys, json, pathlib, time, copy, os
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib

src, out = sys.argv[1], sys.argv[2]
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 1
alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
N = int(sys.argv[5]) if len(sys.argv) > 5 else 400
OFF = int(sys.argv[6]) if len(sys.argv) > 6 else 300
SCOPE = sys.argv[7] if len(sys.argv) > 7 else "all"   # "stem" = only the 3 patch_embed convs
U = np.load(w2lib.R / "night" / "W0_data" / "calib_pool_u8.npy", mmap_mode="r")[OFF:OFF + N]
fp = onnx.load(str(w2lib.FP32_PRE)); fpo = {o for n in fp.graph.node for o in n.output}


def sess(m, outs):
    m = copy.deepcopy(m)
    for t in outs: m.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
    so = ort.SessionOptions(); so.intra_op_num_threads = 3; so.inter_op_num_threads = 1
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])


def conv_jobs(m):
    init, prod, cons = w2lib.maps(m); J = []
    for n in m.graph.node:
        if n.op_type != "Conv" or len(n.input) < 3: continue
        if SCOPE == "stem" and "/patch_embed/" not in n.name: continue
        bdq = prod.get(n.input[2])
        if bdq is None or bdq.op_type != "DequantizeLinear" or bdq.input[0] not in init: continue
        if n.output[0] not in fpo: continue
        J.append({"node": n.name, "y": n.output[0], "bq": bdq.input[0], "bs": nh.to_array(init[bdq.input[1]]).astype(np.float64)})
    return J


# FP32 per-channel means (once)
m = w2lib.load(src)
J = conv_jobs(m); ys = [j["y"] for j in J]
print(len(J), "convs", flush=True)
sf = sess(fp, ys); F = {y: 0 for y in ys}
t0 = time.time()
for i in range(N):
    o = sf.run(ys, {"input": U[i:i + 1].astype(np.float32) / np.float32(255)})
    for y, v in zip(ys, o): F[y] = F[y] + v[0].reshape(v.shape[1], -1).mean(1).astype(np.float64)
F = {y: v / N for y, v in F.items()}
print(f"fp32 means {time.time() - t0:.0f}s", flush=True)
hist = []
for it in range(iters):
    blp = w2lib.W2 / "tmp" / f"bce_{os.getpid()}.onnx"
    w2lib.BL.make_boardlike(m, str(blp)); bm = onnx.load(str(blp)); blp.unlink()
    have = {o for n in bm.graph.node for o in n.output}
    yy = [y for y in ys if y in have]
    sq = sess(bm, yy); Q = {y: 0 for y in yy}
    for i in range(N):
        o = sq.run(yy, {"input": U[i:i + 1].astype(np.float32) / np.float32(255)})
        for y, v in zip(yy, o): Q[y] = Q[y] + v[0].reshape(v.shape[1], -1).mean(1).astype(np.float64)
    init = {t.name: t for t in m.graph.initializer}
    rec = []
    for j in J:
        if j["y"] not in Q: continue
        e = Q[j["y"]] / N - F[j["y"]]
        t = init[j["bq"]]; b = nh.to_array(t).astype(np.int64)
        corr = np.rint(alpha * e / j["bs"]).astype(np.int64)
        t.CopyFrom(nh.from_array((b - corr).astype(np.int32), j["bq"]))
        rec.append({"node": j["node"], "err_absmax": float(np.abs(e).max()), "err_absmean": float(np.abs(e).mean()), "corr_absmax": int(np.abs(corr).max())})
    hist.append({"iter": it, "convs": len(rec), "mean_abs_err_mean": float(np.mean([r["err_absmean"] for r in rec])), "per_conv_first5": rec[:5]})
    print(f"iter {it}: {len(rec)} convs corrected, mean |err| {hist[-1]['mean_abs_err_mean']:.4g}, {time.time() - t0:.0f}s", flush=True)
ok, zps = w2lib.a2_check(m); assert ok, zps
sha = w2lib.save(m, out)
json.dump({"src": src, "out": out, "scope": SCOPE, "iters": iters, "alpha": alpha, "N": N, "offset": OFF, "sha256": sha, "a2_zps": zps, "hist": hist},
          open(out + ".json", "w"), indent=1)
print("done", sha)
