"""Proxy evaluation of existing model files: board-like (or plain) on the first n calib-pool images.
usage: python -I proxy_models.py <log.jsonl> <n> <mode> model1.onnx [model2.onnx ...]"""
import sys, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ev

logp, n, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
for p in sys.argv[4:]:
    name = pathlib.Path(p).stem
    r = ev.run(p, "proxy_" + name, split="calib_pool", mode=mode, n=n, log=False)
    rec = {"name": name, "model": p, "mode": mode, "n": n, "cos_mean": r["cos_vs_fp32_mean"], "agree": r["agreement_top1_with_fp32"],
           "top1": r["top1"], "a2_ok": r["a2_ok"], "sha": r["model_sha256"], "npz": r["npz"], "time": time.strftime("%H:%M:%S")}
    with open(logp, "a") as f: f.write(json.dumps(rec) + "\n")
    print(f"{name:34s} cos {rec['cos_mean']:.4f} agree {rec['agree']:.3f} top1 {rec['top1']:.3f} a2 {rec['a2_ok']}", flush=True)
