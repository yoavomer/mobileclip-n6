"""Proxy search: build each named spec from a source model, evaluate board-like (or plain) on the first n calib-pool
images, log mean cosine vs FP32 (+ top-1 agreement, top-1 vs labels), delete the temporary model.
spec file: JSON list of {"name": ..., "ranges": {tensor: [lo, hi]}}   (optional "src")
usage: python -I search.py <specs.json> <log.jsonl> [n=300] [mode=boardlike]"""
import sys, json, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib, ev
from build_ranges import build

specs = json.load(open(sys.argv[1])); logp = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
mode = sys.argv[4] if len(sys.argv) > 4 else "boardlike"
done = set()
if os.path.exists(logp):
    for l in open(logp):
        try: done.add(json.loads(l)["name"])
        except Exception: pass
for sp in specs:
    if sp["name"] in done: continue
    p = w2lib.W2 / "tmp" / f"srch_{sp['name']}.onnx"
    info = build(sp["ranges"], p, sp.get("src", w2lib.P9995))
    r = ev.run(p, "srch_" + sp["name"], split="calib_pool", mode=mode, n=n, log=False)
    rec = {"name": sp["name"], "ranges": sp["ranges"], "mode": mode, "n": n, "cos_mean": r["cos_vs_fp32_mean"],
           "cos_median": r["cos_vs_fp32_median"], "agree": r["agreement_top1_with_fp32"], "top1": r["top1"],
           "a2_ok": r["a2_ok"], "sha": info["sha256"], "edits": [(e["new_scale"], e["new_zp"]) for e in info["edits"]],
           "npz": r["npz"], "time": time.strftime("%H:%M:%S")}
    with open(logp, "a") as f: f.write(json.dumps(rec) + "\n")
    print(f"{sp['name']:30s} cos {rec['cos_mean']:.4f} agree {rec['agree']:.3f} top1 {rec['top1']:.3f}", flush=True)
    os.remove(p)
