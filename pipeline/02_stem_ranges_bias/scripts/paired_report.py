"""Paired comparisons (W0 paired_diff: bootstrap 95 % CI of the top-1 / top-5 difference on the same 3,000 SELECTION images).
-> results/paired.json + printed table"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ev

rows = [json.loads(l) for l in open(ev.W2 / "results" / "all_results.jsonl") if l.strip()]
npz = {}
for r in rows:
    if r.get("split") == "selection" and r.get("n") == 3000:
        nm, md = r["name"], r["mode"]
        if nm.endswith("__blct"): nm, md = nm[:-6], "blct"
        npz[(nm, md)] = r["npz"]
PAIRS = [("S5_stemS1_othersS2t_bcall", "boardlike", "p9995", "boardlike"), ("S5_stemS1_othersS2t_bcall", "blct", "p9995", "blct"),
         ("S5_stemS1_othersS2t_bcall", "plain", "p9995", "plain"), ("S5_stemS1_othersS2t_bcall", "blct", "S3_S2t_bcall", "blct"),
         ("S5_stemS1_othersS2t_bcall", "boardlike", "S3_S2t_bcall", "boardlike"), ("S5_stemS1_othersS2t_bcall", "blct", "S1_stem_best3", "blct"), ("S5_stemS1_othersS2t_bcall", "blct", "S1tBA_S1t_bcall", "blct"),
         ("S1tBA_S1t_bcall", "blct", "p9995", "blct"),
         ("S3_S2t_bcall", "boardlike", "p9995", "boardlike"), ("S3_S2t_bcall", "blct", "p9995", "blct"),
         ("S3_S2t_bcall", "plain", "p9995", "plain"), ("S1_stem_best3", "boardlike", "p9995", "boardlike"),
         ("S1_stem_best3", "blct", "p9995", "blct"), ("S1_stem_best3", "plain", "p9995", "plain"),
         ("S1t_stem_tableaware", "boardlike", "S1_stem_best3", "blct"), ("S1t_stem_tableaware", "boardlike", "p9995", "boardlike"),
         ("S2t_all30_tableaware", "boardlike", "S1t_stem_tableaware", "boardlike"), ("S3_S2t_bcall", "boardlike", "S2t_all30_tableaware", "boardlike"),
         ("S1tBA_S1t_bcall", "boardlike", "S3_S2t_bcall", "boardlike"), ("S1tBA_S1t_bcall", "boardlike", "p9995", "boardlike"),
         ("B_all_p9995", "boardlike", "p9995", "boardlike"), ("S2_all30_best", "blct", "S2t_all30_tableaware", "boardlike"),
         ("S2t_all30_tableaware", "plain", "p9995", "plain"), ("S1t_stem_tableaware", "plain", "p9995", "plain"),
         ("B_all_p9995", "plain", "p9995", "plain"), ("S1tBA_S1t_bcall", "plain", "p9995", "plain"),
         ("S2_all30_best", "boardlike", "p9995", "boardlike"), ("S2_all30_best", "plain", "p9995", "plain")]
out = []
for a, ma, b, mb in PAIRS:
    if (a, ma) not in npz or (b, mb) not in npz: continue
    d1 = ev.paired(npz[(a, ma)], npz[(b, mb)]); d5 = ev.paired(npz[(a, ma)], npz[(b, mb)], metric="correct5")
    out.append({"a": f"{a} [{ma}]", "b": f"{b} [{mb}]", "a_top1": d1["a_top1"], "b_top1": d1["b_top1"], "diff_top1": d1["diff_top1"],
                "diff_top1_ci95": d1["diff_top1_ci95"], "a_only": d1["a_only"], "b_only": d1["b_only"], "sign_test_p": d1["sign_test_p"],
                "diff_top5": d5["diff_top5"], "diff_top5_ci95": d5["diff_top5_ci95"]})
json.dump(out, open(ev.W2 / "results" / "paired.json", "w"), indent=1)
sys.stdout.reconfigure(encoding="utf-8")
print("| A | B | top-1 A / B | Δ top-1 [95 % CI] | A-only / B-only correct, sign-test p | Δ top-5 [95 % CI] |\n|---|---|---|---|---|---|")
for o in out:
    print(f"| {o['a']} | {o['b']} | {100 * o['a_top1']:.2f} / {100 * o['b_top1']:.2f} % | {100 * o['diff_top1']:+.2f} pp "
          f"[{100 * o['diff_top1_ci95'][0]:+.2f}, {100 * o['diff_top1_ci95'][1]:+.2f}] | {o['a_only']} / {o['b_only']}, p = {o['sign_test_p']:.2g} | "
          f"{100 * o['diff_top5']:+.2f} pp [{100 * o['diff_top5_ci95'][0]:+.2f}, {100 * o['diff_top5_ci95'][1]:+.2f}] |")
