"""tool checks: (1) removing only the GELU-inner QDQs before board-like = no change; (2) my board-like base = W0's
boardlike result for P99.95 (embeddings of calib rows are not in W0 -> compare via SELECTION first 20 images)."""
import sys, json, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, w3lib as L
m = L.load()
ch = L.gelu_chains(m)
print("chains", len(ch), "inner per chain", sorted(set(len(c["inner"]) for c in ch)), "with output", sum(c["output"] is not None for c in ch))
inner = [t for c in ch for t in c["inner"]]
U = np.load(L.W0 / "selection_u8.npy", mmap_mode="r")[:20]
t0 = time.time(); b0, rep0 = L.to_boardlike_bytes(m, "base"); print("convert", round(time.time() - t0, 1), "s")
t0 = time.time(); E0 = L.embed(b0, U, 1); print("base 20 img 1 thread", round(time.time() - t0, 1), "s")
b1, rep1 = L.to_boardlike_bytes(L.bypass(m, inner), "inner")
E1 = L.embed(b1, U, 1)
print("gelu rewritten base / inner-removed:", sum("skipped" not in g for g in rep0["gelu"]), sum("skipped" not in g for g in rep1["gelu"]))
print("inner-removed vs base: max|dE|", float(np.abs(E1 - E0).max()), "sha equal", L.sha256_bytes(b0) == L.sha256_bytes(b1))
W0E = np.load(L.W0 / "results" / "p9995_bl_d948cf85eeb1_boardlike_selection_n3000_templates80.npz")["E"][:20]
print("my base vs W0 board-like P99.95 npz: max|dE|", float(np.abs(E0 - W0E).max()), "min cos", float((E0 * W0E).sum(1).min()))
print("W0 bl sha d948cf85 vs mine", L.sha256_bytes(b0)[:12])
allbut = [t for t in L.Info(m).actQ if t not in L.ATT]
b2, _ = L.to_boardlike_bytes(L.bypass(m, allbut), "allbut")
E2 = L.embed(b2, U, 1)
F = np.load(L.W0 / "results" / "fp32ref_selection.npz")["E"][:20]
print("all activations float (weights int8, A2 4 quantized): cos vs FP32", float((E2 * F).sum(1).mean()), "base", float((E0 * F).sum(1).mean()))
