"""Assemble night/W2_stem/results.md from results/results_table.md (make_results.py), results/paired.json
(paired_report.py), best/SHA256SUMS.txt and the compile summaries."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import json, pathlib, subprocess, sys
W2 = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"\night\W2_stem")
PY = r"python"
for s in ("make_results.py", "paired_report.py"):
    subprocess.run([PY, "-I", str(W2 / "scripts" / s)], check=True, capture_output=True)
table = (W2 / "results" / "results_table.md").read_text(encoding="utf-8")
paired = json.load(open(W2 / "results" / "paired.json"))
sha = (W2 / "best" / "SHA256SUMS.txt").read_text(encoding="utf-8")
res = {e["name"]: e for e in json.load(open(W2 / "results" / "results.json"))}
b = res["S5_stemS1_othersS2t_bcall"]
pc = lambda x: f"{100 * x:.2f} %"
L = ["# W2 — results (SELECTION, 3,000 ImageNetV2 images, 80-template text; nothing on the board)", "",
     "Tags: accuracies [measured] (W0 `zs_eval` via `scripts\\ev.py`, ORT 3 threads); compile columns [from report]",
     "(stedgeai via `night\\tools\\compile.sh`, `scripts\\compile_report.py`, `bisect\\compare_mapping.py`). Modes: **plain** = ORT",
     "default kernels; **board-like** = W1 `boardlike.py` v4 (ideal GELU / Sigmoid tables); **bl + compiled tables** = the",
     "same emulator with the candidate's own compiled GELU / Sigmoid tables (`scripts\\bl_ct.py`; the most faithful number,",
     "see NOTES E6, E11). Δ = paired bootstrap 95 % CI vs P99.95 in the same mode. A2 (4 attention-operand zp = 0) verified",
     "in every candidate ONNX (and in the compiled Q.json of S3 and S5).", "",
     "## Best candidate: S5 (by board-like SELECTION top-1)", "",
     f"- **ONNX:** `night\\W2_stem\\best\\S5_stemS1_othersS2t_bcall.onnx` (= `models\\S5_stemS1_othersS2t_bcall.onnx`)",
     f"- **Compiled:** `night\\W2_stem\\best\\S5_stemS1_othersS2t_bcall_network_rel.bin` (= `compile\\S5_stemS1_othersS2t_bcall\\gout\\network_rel.bin`)",
     "- **What it is:** P99.95 A2 + new int8 range (scale, zp) on the 30 GELU-input tensors (3 stem conv outputs: GELU-aware",
     "  local-error optimum; 27 others: table-aware choice) + bias correction of the weight-rounding mean error in all 95 int8",
     "  convs (int32 biases only). No graph change, no weight change, no retraining.",
     f"- board-like top-1 **{pc(b['boardlike']['top1'])}** [{100 * b['boardlike']['top1_ci95'][0]:.2f}, {100 * b['boardlike']['top1_ci95'][1]:.2f}], "
     f"Δ vs P99.95 {100 * b['boardlike']['vs_p9995'][0]:+.2f} pp [{100 * b['boardlike']['vs_p9995'][1][0]:+.2f}, {100 * b['boardlike']['vs_p9995'][1][1]:+.2f}]; "
     f"top-5 {pc(b['boardlike']['top5'])}; share of FP32 top-1 {100 * b['boardlike']['share']:.1f} %; agreement with FP32 {100 * b['boardlike']['agree']:.1f} %",
     f"- bl + compiled tables top-1 **{pc(b['blct']['top1'])}** [{100 * b['blct']['top1_ci95'][0]:.2f}, {100 * b['blct']['top1_ci95'][1]:.2f}], "
     f"Δ {100 * b['blct']['vs_p9995'][0]:+.2f} pp [{100 * b['blct']['vs_p9995'][1][0]:+.2f}, {100 * b['blct']['vs_p9995'][1][1]:+.2f}]; plain top-1 {pc(b['plain']['top1'])} "
     f"(Δ {100 * b['plain']['vs_p9995'][0]:+.2f} pp [{100 * b['plain']['vs_p9995'][1][0]:+.2f}, {100 * b['plain']['vs_p9995'][1][1]:+.2f}])",
     "- FP32 on the same images: top-1 60.43 %, top-5 86.63 % [from W0].",
     "- Statistically tied with **S3** (all GELU tables exact): S5 − S3 = +0.63 pp [−0.57, +1.83] with compiled tables.",
     "", "SHA256 (`best\\SHA256SUMS.txt`):", "```", sha.strip(), "```", "",
     "## All candidates", "", table, "",
     "Reference P99.95 compile [from report, REPORT.md]: 215 / 64 / 18 / 15 epochs, SW MACs 8,388,608, hyperRAM 1.500 MB;",
     "mci0_256_a2: 207 / 64 / 18 / 15, same SW MACs / hyperRAM / mapping. `B_all_p9995` and `S1tBA_S1t_bcall` were not",
     "compiled (bias-only / range-only variants of compiled candidates; S5 and S3 cover both changes).", "",
     "## Paired comparisons (same 3,000 images)", "",
     "| A | B | top-1 A / B | Δ top-1 [95 % CI] | A-only / B-only correct, sign-test p | Δ top-5 [95 % CI] |", "|---|---|---|---|---|---|"]
for o in paired:
    L.append(f"| {o['a']} | {o['b']} | {100 * o['a_top1']:.2f} / {100 * o['b_top1']:.2f} % | {100 * o['diff_top1']:+.2f} pp "
             f"[{100 * o['diff_top1_ci95'][0]:+.2f}, {100 * o['diff_top1_ci95'][1]:+.2f}] | {o['a_only']} / {o['b_only']}, p = {o['sign_test_p']:.2g} | "
             f"{100 * o['diff_top5']:+.2f} pp [{100 * o['diff_top5_ci95'][0]:+.2f}, {100 * o['diff_top5_ci95'][1]:+.2f}] |")
(W2 / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("written", len(L), "lines")
