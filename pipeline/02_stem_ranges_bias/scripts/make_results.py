"""Collect SELECTION results (results/all_results.jsonl) + compile summaries into results/results.json and results.md.
Modes: plain, boardlike (W1 v4, ideal tables), blct (W1 v4 with the candidate's own compiled GELU/Sigmoid tables;
evaluated as a ready-made board-like model, name '<cand>__blct').
Paired differences vs P99.95 in the same mode (bootstrap 95 % CI, W0 paired_diff)."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ev

W2 = ev.W2
CANDS = json.load(open(W2 / "results" / "candidates.json"))   # [{"name", "method", "model", "compile"}]
rows = [json.loads(l) for l in open(W2 / "results" / "all_results.jsonl") if l.strip()]
sel = {}
for r in rows:
    if r.get("split") != "selection" or r.get("n") != 3000: continue
    name = r["name"]; mode = r["mode"]
    if name.endswith("__blct"): name, mode = name[:-6], "blct"
    sel[(name, mode)] = r                                   # last one wins
ref = {m: sel.get(("p9995", m)) for m in ("plain", "boardlike", "blct")}
out = []
for c in CANDS:
    e = {"name": c["name"], "method": c["method"], "model": c["model"]}
    for m in ("plain", "boardlike", "blct"):
        r = sel.get((c["name"], m))
        if r is None: continue
        d = {"top1": r["top1"], "top1_ci95": r["top1_ci95"], "top5": r["top5"], "share": r["share_of_fp32_top1"],
             "share_ci95": r.get("share_of_fp32_top1_ci95"), "agree": r["agreement_top1_with_fp32"], "cos": r["cos_vs_fp32_mean"],
             "npz": r["npz"]}
        if ref.get(m) and c["name"] != "p9995":
            p = ev.paired(r["npz"], ref[m]["npz"]); d["vs_p9995"] = [p["diff_top1"], p["diff_top1_ci95"], p["a_only"], p["b_only"], p["sign_test_p"]]
            p5 = ev.paired(r["npz"], ref[m]["npz"], metric="correct5"); d["vs_p9995_top5"] = [p5["diff_top5"], p5["diff_top5_ci95"]]
        e[m] = d
    cs = c.get("compile")
    if cs and (W2 / cs / "summary.json").exists():
        e["compile"] = json.load(open(W2 / cs / "summary.json"))
    out.append(e)
json.dump(out, open(W2 / "results" / "results.json", "w"), indent=1)

pc = lambda x: f"{100 * x:.2f} %"
def cell(e, m):
    d = e.get(m)
    if not d: return "—"
    s = f"{pc(d['top1'])} [{100 * d['top1_ci95'][0]:.1f}, {100 * d['top1_ci95'][1]:.1f}]"
    if "vs_p9995" in d: s += f"; Δ {100 * d['vs_p9995'][0]:+.2f} pp [{100 * d['vs_p9995'][1][0]:+.1f}, {100 * d['vs_p9995'][1][1]:+.1f}]"
    return s
L = ["| candidate | method | plain top-1 [95 % CI]; Δ vs P99.95 [paired CI] | board-like top-1 | board-like + compiled tables top-1 | top-5 (plain / bl / bl+ct) | share of FP32 top-1 (plain / bl / bl+ct) | agreement with FP32 (plain / bl / bl+ct) | compiles? | epochs HW/SW/hyb/no-op | SW MACs | hyperRAM | mapping vs P99.95 / vs a2 |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for e in out:
    t5 = " / ".join(pc(e[m]["top5"]) if m in e else "—" for m in ("plain", "boardlike", "blct"))
    sh = " / ".join(f"{100 * e[m]['share']:.1f} %" if m in e else "—" for m in ("plain", "boardlike", "blct"))
    ag = " / ".join(f"{100 * e[m]['agree']:.1f} %" if m in e else "—" for m in ("plain", "boardlike", "blct"))
    c = e.get("compile")
    if e["name"] == "p9995": comp = ("yes (earlier, quant/layers/gout_p9995) [from report]", "215/64/18/15", "8,388,608", "1.500 MB", "— / same 220, diff 0 [from report]")
    elif c is None: comp = ("not compiled", "—", "—", "—", "—")
    elif "epochs" not in c:
        err = " ".join(c.get("error", []))
        short = ("atonn assertion ContainersBits() <= 8, E103" if "ContainersBits" in err else err[-100:])
        comp = (f"**no** ({short})", "—", "—", "—", "—")
    else:
        ep = c["epochs"]; mp = lambda k: (c[k][0].split("|")[-1].strip().replace("ops 220: same mapping", "same").replace(", not found in full 0", "").replace("different", "diff") if c.get(k) else "?")
        comp = ("yes", f"{ep['HW']}/{ep['SW']}/{ep['hybrid']}/{ep['noop']}", f"{c.get('sw_macs', '?'):,}", c.get("hyperRAM", "?"),
                f"{mp('mapping_vs_p9995')} / {mp('mapping_vs_a2')}")
    L.append(f"| {e['name']} | {e['method']} | {cell(e, 'plain')} | {cell(e, 'boardlike')} | {cell(e, 'blct')} | {t5} | {sh} | {ag} | "
             + " | ".join(comp) + " |")
(W2 / "results" / "results_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8"); print("\n".join(L))
