"""Experiment E1 (task 3, slow numpy reference): integer emulation of the whole stem (patch_embed.0 -> .1 -> .2) of the
compiled P99.95 model, chained block by block from the board's int8 input, compared with the board's C0 / C2 dumps.
Per block [from code: network_blob_trace.c via the helpers of stem/decode_stem.py, same decoding as stem/emulate_mid.py]:
   acc = sum w * (q - z_in) (exact int)  ->  x16 = rshift(acc, shift_o of the last CONV pipe)
   ->  u8 = sat8(rshift(A[c] * x16 + C[c], shift_o of the ARITH off_bias unit))   (8-bit, ONNX conv-output Q)
   ->  y = GELU_LUT[u8]  (compiled ACTIV table, = ideal GELU-once for these 3 tables)
Rounding of the right shifts: 'half_up' (analyst's assumption) vs 'half_away' (ties away from zero: what the softmax
units do on the board, NOTES.md S3). Compared also: board-like ONNX (make_boardlike, = R_lut) and plain ORT.
-> stem_int_emulator.json"""
import re, json, numpy as np, onnx, torch
from onnx import numpy_helper as nh
from common import R, HERE, stats, fmt
import hwlib
src = open(R / "stem/decode_stem.py").read().split("m = onnx.load")[0]
ns = {"__file__": str(R / "stem/decode_stem.py")}; exec(src, ns)            # P (params pool bytes), struct(), name_of(), T
P, struct, name_of, T = ns["P"], ns["struct"], ns["name_of"], ns["T"]
WS = str(R / "quant/layers/gws_p9995/neural_art__network")
m = onnx.load(str(R / "quant/layers/mci0_256_p9995.onnx")); g = m.graph
ini = {x.name: nh.to_array(x) for x in g.initializer}; prod = {o: n for n in g.node for o in n.output}
BL = {"patch_embed.0": ("Conv2D_30", "Conv2D_30", "Div_33"), "patch_embed.1": ("Conv2D_45", "Conv2D_45", "Div_48"),
      "patch_embed.2": ("Conv2D_59", "Conv2D_59_ca_pipe_3", "Div_62")}
meta = {}
for blk, (cv, last, gel) in BL.items():
    c = next(n for n in g.node if n.op_type == "Conv" and n.name.startswith(f"/model/patch_embed/{blk}/"))
    xdq, wdq = prod[c.input[0]], prod[c.input[1]]
    ar = struct(re.search(r"static const \w+ (" + cv + r"_off_bias_\d+_init\d+) = ", T).group(1))
    meta[blk] = dict(zx=int(ini[xdq.input[2]]), w=ini[wdq.input[0]].astype(np.float64),
                     stride=[a.ints for a in c.attribute if a.name == "strides"][0], pads=[a.ints for a in c.attribute if a.name == "pads"][0],
                     group=([a.i for a in c.attribute if a.name == "group"] or [1])[0],
                     A=np.frombuffer(P, np.int16, 64, int(ar["A_vector"].split("@")[-1].rstrip("}"))).astype(np.int64),
                     C=np.frombuffer(P, np.int32, 64, int(ar["C_vector"].split("@")[-1].rstrip("}"))).astype(np.int64),
                     sh_c=int(struct(name_of(last))["shift_o"]), sh_a=int(ar["shift_o"]),
                     lut={rm: hwlib.activ_lut(WS, gel, rm) for rm in ("half_up", "half_away")})


def rshift(v, n, rm):
    if n == 0: return v
    if rm == "half_up": return (v + (1 << (n - 1))) >> n
    a = (np.abs(v) + (1 << (n - 1))) >> n; return np.where(v < 0, -a, a)


def block(q, md, rm):
    pd = md["pads"]
    xp = np.pad(q - md["zx"], ((0, 0), (0, 0), (pd[0], pd[2]), (pd[1], pd[3]))).astype(np.float64)
    acc = np.rint(torch.nn.functional.conv2d(torch.from_numpy(xp), torch.from_numpy(md["w"]), stride=tuple(md["stride"]), groups=md["group"]).numpy()).astype(np.int64)
    x16 = np.clip(rshift(acc, md["sh_c"], rm), -32768, 32767)
    u8 = np.clip(rshift(md["A"][None, :, None, None] * x16 + md["C"][None, :, None, None], md["sh_a"], rm), -128, 127)
    return md["lut"][rm][u8 + 128]


cuts = json.load(open(R / "stem/cuts.json")); names = [n.replace(".jpg", "") for n in (R / "stem/inputs.txt").read_text().split()]
X = np.load(R / "stem/inputs_float.npy"); S_IN = cuts["input_scale"]
res = {}
for i, nm in enumerate(names):
    q0 = np.clip(np.rint(X[i] / S_IN) - 128, -128, 127).astype(np.int64)
    for rm in ("half_up", "half_away"):
        y0 = block(q0, meta["patch_embed.0"], rm); y1 = block(y0, meta["patch_embed.1"], rm); y2 = block(y1, meta["patch_embed.2"], rm)
        for cut, y in (("C0", y0), ("C2", y2)):
            bf = R / f"stem/board/{cut}_{nm}.npy"
            if not bf.exists(): continue
            b = np.load(bf).astype(np.int64); z = cuts[cut]["zp"]
            r = stats(y.ravel(), b, z)
            if cut == "C0":
                mask = np.ones(b.size, bool); mask[8:3188] = False; r["excl8_3187"] = stats(y.ravel(), b, z, mask)
            rl = np.load(R / f"stem/refs/{cut}_{nm}_R_lut.npy").astype(np.int64)
            r["R_lut_vs_board"] = stats(rl, b, z)
            res[f"{cut} {nm} {rm}"] = r
            print(f"{cut} {nm:13s} {rm:9s} int-emulator vs board {fmt(r)}" + (f" (excl {fmt(r['excl8_3187'])})" if cut == "C0" else "")
                  + f" | R_lut (= board-like ONNX) vs board {fmt(r['R_lut_vs_board'])}", flush=True)
json.dump(res, open(HERE / "stem_int_emulator.json", "w"), indent=1)
