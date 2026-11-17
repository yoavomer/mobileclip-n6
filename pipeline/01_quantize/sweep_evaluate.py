"""Percentile sweep evaluation (ORT CPU; default kernels and opt-off).
Metrics vs FP32 (mci0_256_fp32.onnx): (a) embedding cosine mean/median/min; (b) COCO-80 and (c) ImageNet-1000 zero-shot
('a photo of a {name}', FP32 text embeddings): top-1 agreement % and mean top-3 overlap (of 3).
usage: evaluate.py <selection|test> name=model.onnx ...   (test set only with I_HAVE_CHOSEN=1)"""
import os, sys, json, pathlib, numpy as np, onnxruntime as ort
R = pathlib.Path(__file__).resolve().parents[2]; S = R / "quant" / "sweep"
which = sys.argv[1]
if which == "test" and os.environ.get("I_HAVE_CHOSEN") != "1": sys.exit("test set is locked until the value is chosen")
X = np.load(R / "quant" / "eval100.npy") if which == "selection" else np.load(S / "test200.npy")
TEXT = {"coco80": np.load(R / "quant" / "text80.npy"), "imagenet1000": np.load(S / "text_imagenet1000.npy")}


def embed(path, opt):
    so = ort.SessionOptions()
    if not opt: so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    E = np.concatenate([s.run(None, {"input": X[i:i + 1]})[0] for i in range(len(X))]).astype(np.float64)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


fp = S / f"emb_fp32_{which}.npy"
F = np.load(fp) if fp.exists() else embed(R / "mci0_256_fp32.onnx", True)
np.save(fp, F)
FT = {k: np.argsort(-(F @ T.T), axis=1)[:, :3] for k, T in TEXT.items()}
out = S / f"results_{which}.json"
res = json.load(open(out)) if out.exists() else {}
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1)
    for mode in ("default", "opt_off"):
        E = embed(path, mode == "default"); cos = (E * F).sum(1)
        r = {"path": path, "n": len(X), "cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)), "cos_min": float(cos.min())}
        for k, T in TEXT.items():
            t3 = np.argsort(-(E @ T.T), axis=1)[:, :3]
            r[f"{k}_top1"] = float(np.mean(t3[:, 0] == FT[k][:, 0]) * 100)
            r[f"{k}_top3"] = float(np.mean([len(set(a) & set(b)) for a, b in zip(t3, FT[k])]))
        res[f"{name}/{mode}"] = r
        print(f"{which} {name:10s} {mode:7s} cos {r['cos_mean']:.4f}/{r['cos_median']:.4f}/{r['cos_min']:.4f} | "
              f"COCO top1 {r['coco80_top1']:5.1f}% top3 {r['coco80_top3']:.2f} | ImageNet top1 {r['imagenet1000_top1']:5.1f}% top3 {r['imagenet1000_top3']:.2f}")
    json.dump(res, open(out, "w"), indent=1)
