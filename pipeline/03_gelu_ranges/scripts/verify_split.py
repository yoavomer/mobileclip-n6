"""Verify that the W4 split form (stem depthwise conv in 4 channel groups) of a candidate is integer-identical:
max |embedding diff| over N SELECTION images (default 24) in ORT plain (ORT_ENABLE_ALL and ORT_DISABLE_ALL) and in the
board-like conversion (W1 v4) of both models. Also checks A2 on the split model.
usage: verify_split.py <cand.onnx> <cand_splitdw4.onnx> [N=24] -> prints JSON, writes <split>.verify.json"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, onnxruntime as ort, w3lib as L

a, b = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
U = np.load(L.W0 / "selection_u8.npy", mmap_mode="r")[:N]
res = {"cand": a, "split": b, "n_images": N, "cand_sha256": L.sha256_file(a), "split_sha256": L.sha256_file(b)}


def run(bytes_, opt):
    so = ort.SessionOptions(); so.intra_op_num_threads = 3; so.log_severity_level = 3
    if opt == "off": so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(bytes_, so, providers=["CPUExecutionProvider"])
    return np.stack([s.run(None, {"input": U[i:i + 1].astype(np.float32) / np.float32(255)})[0].reshape(-1) for i in range(N)])


ba, bb = pathlib.Path(a).read_bytes(), pathlib.Path(b).read_bytes()
for opt in ("default", "off"):
    res[f"plain_{opt}_max_abs_diff"] = float(np.abs(run(ba, opt) - run(bb, opt)).max())
bla, _ = L.to_boardlike_bytes(L.load(a), "va"); blb, rep = L.to_boardlike_bytes(L.load(b), "vb")
res["boardlike_max_abs_diff"] = float(np.abs(run(bla, "default") - run(blb, "default")).max())
res["boardlike_gelu_rewritten_split"] = sum("skipped" not in g for g in rep["gelu"])
ok, zps = L.a2_check(L.load(b)); res["a2_ok_split"] = ok; res["a2_split"] = {k: list(v) for k, v in zps.items()}
json.dump(res, open(b + ".verify.json", "w"), indent=1); print(json.dumps(res, indent=1))
