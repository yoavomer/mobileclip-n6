"""Build night/W3_beyond/results.md from the result files (proxy jsonl, SELECTION summary, compile summaries)."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, pathlib
sys.stdout.reconfigure(encoding="utf-8")
W3 = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"\night\W3_beyond"); RS = W3 / "results"
sys.path.insert(0, str(W3 / "scripts")); import w3lib as L


def jl(name):
    p = RS / name
    return [json.loads(l) for l in open(p) if l.strip() and "error" not in json.loads(l)] if p.exists() else []


def proxy_table(rows, base_cos, base_top1, title, sort=True, label=lambda n: n):
    o = [f"#### {title}", "", "| job | cos vs FP32 | Δcos vs base | agree FP32 | top-1 (500 img) |", "|---|---|---|---|---|"]
    if sort: rows = sorted(rows, key=lambda r: -r["cos"])
    for r in rows:
        o.append(f"| {label(r['name'])} | {r['cos']:.4f} | {r['cos'] - base_cos:+.4f} | {100 * r['agree']:.1f} % | {100 * r['top1']:.1f} % |")
    return o + [""]


out = ["# W3 — results tables (all numbers [measured] tonight unless tagged otherwise)", "",
       "Proxy = board-like (W1 v4) on calib-B = calib_pool rows 500..999 (500 images, 500 classes; FP32 top-1 57.0 %).",
       "Δcos = paired difference vs the base on the same images (paired SE of a Δ: 0.0001–0.0008).",
       "SELECTION = 3,000 images, W0 `zs_eval` (80 templates), Wilson CI for accuracies, paired bootstrap (B = 10,000) for differences.", ""]

out += ["## 1. Block sensitivity (fix-one: all activation QDQs inside the block FP32, weights int8)", ""]
for f, t in (("proxy_blocks.jsonl", "base = P99.95 (`quant\\layers\\mci0_256_p9995.onnx`)"), ("proxy_blocks_s3.jsonl", "base = W2's S3 (`models\\ext\\W2_S3_S2t_bcall.onnx`)"),
             ("proxy_w3u_round2.jsonl", "base = W3u (this workstream's best, after the GELU-chain fix)")):
    rows = jl(f)
    if f == "proxy_w3u_round2.jsonl": rows = [r for r in rows if r["name"].startswith(("blk:", "allfloat", "W3t_base"))]
    if not rows: continue
    b = next((r for r in rows if r["name"] in ("base", "W3t_base")), None)
    if b is None:
        b = next((r for r in jl("proxy_w3u_round2.jsonl") if r["name"] == "W3t_base"), None)
    if b is None: continue
    out += proxy_table([r for r in rows if r["name"] != b["name"]], b["cos"], b["top1"], f"{t}: base cos {b['cos']:.4f}, top-1 {100 * b['top1']:.1f} %")

out += ["## 2. Per-tensor sensitivity inside the worst blocks (S3 base; one board tensor bypassed, followers with their leader)", ""]
tt = RS / "table_tensors_s3.md"
if tt.exists(): out += tt.read_text(encoding="utf-8").splitlines() + [""]

b3 = next(r for r in jl("proxy_blocks_s3.jsonl") if r["name"] == "base")
out += ["## 3. Range methods per tensor class (S3 base; non-ACTIV tensors)", ""]
out += proxy_table(jl("proxy_ranges_s3.jsonl"), b3["cos"], b3["top1"], f"base S3 cos {b3['cos']:.4f}")
out += ["## 4. GELU-chain tops (27 non-stem chains; S3 base)", ""]
out += proxy_table(jl("proxy_gchain_s3.jsonl"), b3["cos"], b3["top1"], f"base S3 cos {b3['cos']:.4f}")
fw = jl("proxy_fpw_s3.jsonl")
if fw: out += ["## 5. Weights FP32 (weights + biases original FP32, activations int8; S3 base)", ""] + proxy_table(fw, b3["cos"], b3["top1"], "base S3")
sm = [r for r in jl("proxy_w3u_round2.jsonl") if not r["name"].startswith(("blk:", "allfloat"))]
if sm and any(r["name"] == "W3t_base" for r in sm):
    bw = next(r for r in sm if r["name"] == "W3t_base")
    out += ["## 6. Small extra range changes on top of W3u (W3t_base = W3u itself)", ""] + proxy_table([r for r in sm if r["name"] != "W3t_base"], bw["cos"], bw["top1"], f"base W3t cos {bw['cos']:.4f}")

out += ["## 7. SELECTION (3,000 images)", ""]
p = RS / "selection_summary.md"
if p.exists(): out += p.read_text(encoding="utf-8").splitlines() + [""]
out += ["## 8. Candidates", ""]
p = RS / "candidates.md"
if p.exists(): out += p.read_text(encoding="utf-8").splitlines() + [""]
(W3 / "results.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out[:20]))
