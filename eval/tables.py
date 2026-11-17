"""Print the markdown result tables for NOTES.md from results/results_selection.json and results/paired_selection.json."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import json, pathlib
W = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"\night\W0_data")
R = json.loads((W / "results" / "results_selection.json").read_text())
P = json.loads((W / "results" / "paired_selection.json").read_text()) if (W / "results" / "paired_selection.json").exists() else {}
pc = lambda x: f"{100 * x:.2f}"
ci = lambda c: f"[{100 * c[0]:.2f}, {100 * c[1]:.2f}]"
print("| model | mode | text | top-1 % [95 % CI] | top-5 % [95 % CI] | share of FP32 top-1 % [CI] | Δtop-1 vs FP32 pp [paired CI] | top-1 agree w/ FP32 % | cos vs FP32 mean / median / min | A2 zp 0 | ms/img |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for k, r in R.items():
    share = f"{100 * r['share_of_fp32_top1']:.1f}" + (f" [{100 * r['share_of_fp32_top1_ci95'][0]:.1f}, {100 * r['share_of_fp32_top1_ci95'][1]:.1f}]" if "share_of_fp32_top1_ci95" in r else "")
    d = (f"{100 * r['diff_top1_vs_fp32']:+.2f} [{100 * r['diff_top1_vs_fp32_ci95'][0]:+.2f}, {100 * r['diff_top1_vs_fp32_ci95'][1]:+.2f}]"
         if "diff_top1_vs_fp32" in r else "—")
    a2 = r["a2_check_src"]; a2s = ("yes" if a2.get("ok") else "NO") if a2.get("applicable") else "n/a"
    print(f"| {r['name']} | {r['mode']} | {r['text']} | {pc(r['top1'])} {ci(r['top1_ci95'])} | {pc(r['top5'])} {ci(r['top5_ci95'])} | "
          f"{share} | {d} | {100 * r['agreement_top1_with_fp32']:.1f} | {r['cos_vs_fp32_mean']:.4f} / {r['cos_vs_fp32_median']:.4f} / "
          f"{r['cos_vs_fp32_min']:.4f} | {a2s} | {r.get('ms_per_image', 'cached')} |")
if P:
    print()
    print("| comparison | metric | A % | B % | A − B pp | paired bootstrap 95 % CI | A-only / B-only correct | sign-test p |")
    print("|---|---|---|---|---|---|---|---|")
    for k, d in P.items():
        m = "top1" if d["metric"] == "correct1" else "top5"
        print(f"| {d['a']} − {d['b']} | {m} | {pc(d['a_' + m])} | {pc(d['b_' + m])} | {100 * d['diff_' + m]:+.2f} | "
              f"[{100 * d['diff_' + m + '_ci95'][0]:+.2f}, {100 * d['diff_' + m + '_ci95'][1]:+.2f}] | {d['a_only']} / {d['b_only']} | {d['sign_test_p']:.2g} |")
