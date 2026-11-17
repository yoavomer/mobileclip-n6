"""SELECTION summary of W3 candidates with the right paired references per mode (paired bootstrap, W0 zs_eval.paired_diff):
  boardlike  vs P99.95 board-like (W0 npz) and vs W2's best S5 board-like (W2 npz)
  blct       (board-like + the candidate's own compiled tables, evaluated as 'plain' of the blct model)
             vs P99.95 blct (W2 npz) and vs S5 blct (W2 npz)
  plain      vs P99.95 plain (W0 npz) and vs S5 plain (W2 npz)
usage: sel_summary.py  -> results/selection_summary.json + markdown table on stdout"""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, pathlib
sys.stdout.reconfigure(encoding="utf-8")
R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
sys.path.insert(0, str(R / "night" / "W0_data")); import zs_eval
W3 = R / "night" / "W3_beyond"; W2R = R / "night" / "W2_stem" / "results"; W0R = R / "night" / "W0_data" / "results"
REF = {"boardlike": {"P99.95": W0R / "p9995_bl_d948cf85eeb1_boardlike_selection_n3000_templates80.npz",
                     "S5": W2R / "S5_stemS1_othersS2t_bcall_7f040d8b9137_boardlike_selection_n3000_templates80.npz"},
       "blct": {"P99.95": W2R / "p9995__blct_4aecec7d4653_plain_selection_n3000_templates80.npz",
                "S5": W2R / "S5_stemS1_othersS2t_bcall__blct_38d15166ed22_plain_selection_n3000_templates80.npz"},
       "plain": {"P99.95": W0R / "p9995_ba18908b45e8_plain_selection_n3000_templates80.npz",
                 "S5": W2R / "S5_stemS1_othersS2t_bcall_477a7df69175_plain_selection_n3000_templates80.npz"}}
rows = [json.loads(l) for l in open(W3 / "results" / "selection_results.jsonl") if l.strip()]
out = []
for r in rows:
    kind = "blct" if r["name"].endswith("__blct") else r["mode"]
    d = {"name": r["name"], "kind": kind, "top1": r["top1"], "top1_ci95": r["top1_ci95"], "top5": r["top5"], "top5_ci95": r["top5_ci95"],
         "share_of_fp32_top1": r["share_of_fp32_top1"], "share_ci95": r["share_of_fp32_top1_ci95"], "cos": r["cos_vs_fp32_mean"],
         "agree": r["agreement_top1_with_fp32"], "a2_ok": r["a2_ok"], "npz": r["npz"], "model_sha256": r["model_sha256"]}
    for k, ref in REF[kind].items():
        if not ref.exists(): continue
        for met in ("correct1", "correct5"):
            p = zs_eval.paired_diff(r["npz"], str(ref), metric=met); key = "top1" if met == "correct1" else "top5"
            d[f"vs_{k}_{key}"] = [p[f"diff_{key}"], p[f"diff_{key}_ci95"], p["a_only"], p["b_only"], p["sign_test_p"]]
    out.append(d)
json.dump(out, open(W3 / "results" / "selection_summary.json", "w"), indent=1)
pc = lambda x: f"{100 * x:.2f}"
print("| model | kind | top-1 % [95 % CI] | top-5 % | share of FP32 top-1 % [CI] | cos | Δtop-1 vs P99.95 pp [CI] | Δtop-1 vs S5 pp [CI] (A-only/B-only) | Δtop-5 vs S5 pp |")
print("|---|---|---|---|---|---|---|---|---|")
for d in out:
    f = lambda k: (f"{100 * d[k][0]:+.2f} [{100 * d[k][1][0]:+.2f}, {100 * d[k][1][1]:+.2f}]" + (f" ({d[k][2]}/{d[k][3]})" if "S5" in k else "")) if k in d else "—"
    print(f"| {d['name']} | {d['kind']} | {pc(d['top1'])} [{pc(d['top1_ci95'][0])}, {pc(d['top1_ci95'][1])}] | {pc(d['top5'])} | "
          f"{pc(d['share_of_fp32_top1'])} [{pc(d['share_ci95'][0])}, {pc(d['share_ci95'][1])}] | {d['cos']:.4f} | {f('vs_P99.95_top1')} | {f('vs_S5_top1')} | {f('vs_S5_top5').split(' (')[0]} |")
