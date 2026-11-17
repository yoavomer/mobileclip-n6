"""Experiment V1: stem cuts (P99.95) — board vs plain ORT vs board-like (make_boardlike) vs R_lut.
Models stem/C0.onnx, stem/C2.onnx (float in, float out = DQ of the cut tensor); int8 = round(y/scale)+zp (stem/cuts.json).
Inputs stem/inputs_float.npy (order stem/inputs.txt). Board dumps stem/board/C*_<input>.npy (complete, CRC-verified).
C0: also reported excluding bytes 8..3187 (non-model data, REPORT 'Open issue: C0').  -> validate_stem.json"""
import json, numpy as np
from common import R, HERE, sess, stats, fmt, q8
from boardlike import make_boardlike
cuts = json.load(open(R / "stem/cuts.json")); names = [n.replace(".jpg", "") for n in (R / "stem/inputs.txt").read_text().split()]
X = np.load(R / "stem/inputs_float.npy")
out = {}
for cut in ("C0", "C2"):
    s, z = cuts[cut]["scale"], cuts[cut]["zp"]
    bl = HERE / "models" / f"{cut}_bl.onnx"
    rep = make_boardlike(R / f"stem/{cut}.onnx", bl)
    S = {"ORT_default": sess(R / f"stem/{cut}.onnx", True), "ORT_opt_off": sess(R / f"stem/{cut}.onnx", False),
         "boardlike_default": sess(bl, True), "boardlike_ext": sess(bl, "ext")}
    for i, nm in enumerate(names):
        y = {k: q8(v.run(None, {"input": X[i]})[0], s, z).ravel() for k, v in S.items()}
        rlut = np.load(R / f"stem/refs/{cut}_{nm}_R_lut.npy").astype(np.int64)
        rec = {"boardlike_default_vs_R_lut": stats(y["boardlike_default"], rlut, z),
               "boardlike_default_vs_ext": stats(y["boardlike_default"], y["boardlike_ext"], z)}
        bf = R / f"stem/board/{cut}_{nm}.npy"
        if bf.exists():
            b = np.load(bf).astype(np.int64)
            for k in S: rec[f"{k}_vs_board"] = stats(y[k], b, z)
            rec["R_lut_vs_board"] = stats(rlut, b, z)
            if cut == "C0":
                mask = np.ones(b.size, bool); mask[8:3188] = False
                for k in S: rec[f"{k}_vs_board_excl8_3187"] = stats(y[k], b, z, mask)
        out[f"{cut} {nm}"] = rec
        line = f"{cut} {nm:13s} BL vs R_lut {fmt(rec['boardlike_default_vs_R_lut'])}"
        if bf.exists():
            line += f" | board: ORT {fmt(rec['ORT_default_vs_board'])}  BL {fmt(rec['boardlike_default_vs_board'])}  BL-ext {fmt(rec['boardlike_ext_vs_board'])}"
            if cut == "C0": line += f" | excl: ORT {fmt(rec['ORT_default_vs_board_excl8_3187'])} BL {fmt(rec['boardlike_default_vs_board_excl8_3187'])}"
        print(line, flush=True)
json.dump(out, open(HERE / "validate_stem.json", "w"), indent=1)
