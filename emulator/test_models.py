"""Task 4: make_boardlike on baseline/mci0_256_a2.onnx and quant/layers/mci0_256_p9995.onnx; ORT runs the result
(same session settings as night/W0_data/zs_eval.py: ORT_ENABLE_ALL, here 3 threads); on quant/eval100.npy (100 COCO images,
float [0,1]): cosine plain vs board-like (float embeddings), and each vs FP32; timing.  -> test_models.json"""
import json, time, numpy as np, onnx, onnxruntime as ort
from common import R, HERE
from boardlike import make_boardlike
X = np.load(R / "quant/eval100.npy")
def S(p):
    so = ort.SessionOptions(); so.intra_op_num_threads = 3; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
def run(s):
    t0 = time.time(); E = np.concatenate([s.run(None, {"input": X[i:i + 1]})[0] for i in range(len(X))]); return E.astype(np.float64), (time.time() - t0) / len(X)
def cos(a, b): return np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
F, _ = run(S(R / "mci0_256_fp32.onnx"))
res = {}
for name, src in (("mci0_256_a2", R / "baseline/mci0_256_a2.onnx"), ("P99.95", R / "quant/layers/mci0_256_p9995.onnx")):
    dst = HERE / "models" / f"test_{name}_bl.onnx"; t0 = time.time(); rep = make_boardlike(src, dst); trw = time.time() - t0
    onnx.checker.check_model(str(dst))
    P, tp = run(S(src)); B, tb = run(S(dst))
    c_pb, c_pf, c_bf = cos(P, B), cos(P, F), cos(B, F)
    r = {"rewrite_s": trw, "ms_per_img_plain": tp * 1000, "ms_per_img_boardlike": tb * 1000,
         "gelu": len(rep["gelu"]), "softmax": sum("skipped" not in x for x in rep["softmax"]), "sigmoid": len(rep["sigmoid"]),
         "matmul_A_zp": rep["a2_zero_points"], "no_qdq_fusion": rep["no_qdq_fusion"],
         "cos_plain_vs_boardlike": {"mean": float(c_pb.mean()), "min": float(c_pb.min()), "max": float(c_pb.max())},
         "cos_plain_vs_fp32_mean": float(c_pf.mean()), "cos_boardlike_vs_fp32_mean": float(c_bf.mean()),
         "boardlike_minus_plain_fp32cos": {"mean": float((c_bf - c_pf).mean()), "sd": float((c_bf - c_pf).std()),
                                            "n_better": int((c_bf > c_pf).sum())}}
    res[name] = r
    print(name, json.dumps(r), flush=True)
json.dump(res, open(HERE / "test_models.json", "w"), indent=1)
