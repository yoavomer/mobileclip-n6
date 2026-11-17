"""markdown table of a proxy jsonl: name | cos | d cos vs base | agree | top1 (sorted by d cos). usage: tables.py <jsonl> [base_name=base] [sort=1]"""
import sys, json
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if "error" not in r]
bn = sys.argv[2] if len(sys.argv) > 2 else "base"
srt = (sys.argv[3] if len(sys.argv) > 3 else "1") == "1"
b = next((r for r in rows if r["name"] == bn), None)
if srt and b: rows.sort(key=lambda r: -(r["cos"] - b["cos"]))
print("| job | cos vs FP32 | Δcos vs base | cos min | agree FP32 | top-1 (500) | Δtop-1 pp |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    d = r["cos"] - b["cos"] if b else float("nan"); dt = 100 * (r["top1"] - b["top1"]) if b else float("nan")
    print(f"| {r['name']} | {r['cos']:.4f} | {d:+.4f} | {r['cos_min']:.3f} | {100 * r['agree']:.1f} % | {100 * r['top1']:.1f} % | {dt:+.1f} |")
