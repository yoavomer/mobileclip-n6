"""results/candidates.md: candidate | change | plain top-1 | board-like top-1 | bl + compiled tables top-1 | share of FP32 (bl) |
compiles | epochs | SW MACs | hyperRAM | mapping. Reads results/selection_summary.json and compile/<cand>/summary.json."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, pathlib
sys.stdout.reconfigure(encoding="utf-8")
W3 = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"\night\W3_beyond")
S = {(d["name"].replace("__blct", ""), d["kind"]): d for d in json.load(open(W3 / "results" / "selection_summary.json"))}
CANDS = [("P99.95 (reference) [W0]", None, "—", {"plain": (40.43, None), "boardlike": (41.17, None), "blct": (41.37, None)},
          {"exit": 0, "epochs": {"HW": 215, "SW": 64, "hybrid": 18, "noop": 15}, "sw_macs": 8388608, "hyperRAM": "1.500 MB", "map": "reference"}),
         ("W2 S5 (W2's best, base of W3) [W2]", "W2_S5", "W2: stem ranges S1 + 27 GELU-input ranges S2t + bias correction (all 95 convs)",
          {"plain": (50.77, None), "boardlike": (53.03, None), "blct": (52.87, None)},
          {"exit": 0, "epochs": {"HW": 215, "SW": 64, "hybrid": 18, "noop": 15}, "sw_macs": 8388608, "hyperRAM": "1.500 MB", "map": "220/220 same [from W2]"}),
         ("W3a_S5_cl2", "W3a_S5_cl2", "S5 + cl2 GELU-chain tops/input lo on 27 non-stem chains (not table-aware)", None, None),
         ("W3t_S5_cl2t", "W3t_S5_cl2t", "S5 + table-aware cl2 (27 non-stem chains)", None, None),
         ("**W3u_S5_cl2t_inner** (best)", "W3u_S5_cl2t_inner", "W3t + inner GELU Mul QDQ made consistent (board-like model and compiled rel byte-identical to W3t)", None, None),
         ("W3u_S5_cl2t_inner_splitdw4", "W3u_S5_cl2t_inner_splitdw4", "W3u + W4 split rewrite (stem dw conv in 4 groups)", None, None)]
o = ["| candidate | change | plain top-1 % | board-like top-1 % [CI] | bl + compiled tables top-1 % | share of FP32 (bl) % | Δ bl vs P99.95 pp [CI] | Δ bl vs S5 pp [CI] | compiles | epochs HW/SW/hyb/noop | SW MACs | hyperRAM | mapping | A2 |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, key, change, fixed, comp in CANDS:
    def g(kind):
        if fixed: return f"{fixed[kind][0]:.2f}"
        d = S.get((key, kind))
        if d: return f"{100 * d['top1']:.2f}" + (f" [{100 * d['top1_ci95'][0]:.2f}, {100 * d['top1_ci95'][1]:.2f}]" if kind == "boardlike" else "")
        if key.endswith("_splitdw4"):
            if kind == "plain": return "= W3u (ORT outputs identical, max diff 0 on 24 images; not run on 3,000)"
            if kind == "blct": return "not read (same Q params as W3u, so the same tables are expected [inference])"
        if key == "W3u_S5_cl2t_inner" and kind in ("boardlike", "blct"):          # byte-identical board-like / blct model as W3t
            d = S.get(("W3t_S5_cl2t", kind))
            if d: return f"{100 * d['top1']:.2f}" + (f" [{100 * d['top1_ci95'][0]:.2f}, {100 * d['top1_ci95'][1]:.2f}] (= W3t, identical model)" if kind == "boardlike" else " (= W3t)")
        return "—"
    bl = S.get((key, "boardlike")) or (S.get(("W3t_S5_cl2t", "boardlike")) if key == "W3u_S5_cl2t_inner" else None)
    share = f"{100 * bl['share_of_fp32_top1']:.1f}" if bl and not fixed else ("68.1" if name.startswith("P99") else ("87.8" if fixed else "—"))
    dp = f"{100 * bl['vs_P99.95_top1'][0]:+.2f} [{100 * bl['vs_P99.95_top1'][1][0]:+.2f}, {100 * bl['vs_P99.95_top1'][1][1]:+.2f}]" if bl and "vs_P99.95_top1" in bl and not fixed else "—"
    ds = f"{100 * bl['vs_S5_top1'][0]:+.2f} [{100 * bl['vs_S5_top1'][1][0]:+.2f}, {100 * bl['vs_S5_top1'][1][1]:+.2f}]" if bl and "vs_S5_top1" in bl and not fixed else "—"
    if comp is None:
        cs = W3 / "compile" / key / "summary.json"
        comp = json.load(open(cs)) if cs.exists() else None
        if comp: comp["map"] = (comp.get("compare_mapping_vs_p9995", [""])[0].split("|")[-1].strip() + "; maptype " + ("OK" if comp.get("maptype_ok") else "CHANGED"))
    if comp and comp.get("exit") == 0:
        e = comp["epochs"]; cc = f"yes | {e['HW']}/{e['SW']}/{e['hybrid']}/{e['noop']} | {comp.get('sw_macs', 0):,} | {comp.get('hyperRAM')} | {comp['map']}"
    else: cc = "pending | | | |"
    a2 = "0/0/0/0"
    o.append(f"| {name} | {change} | {g('plain')} | {g('boardlike')} | {g('blct')} | {share} | {dp} | {ds} | {cc} | {a2} |")
(W3 / "results" / "candidates.md").write_text("\n".join(o) + "\n", encoding="utf-8")
print("\n".join(o))
