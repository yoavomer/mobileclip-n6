"""Does a candidate's compiled stem compute what the board-like ONNX assumes? (NPU fixed-point check, no board)
Integer emulation of the three compiled stem blocks (CONV -> ARITH requant to 8 bit -> ACTIV GELU table), with the
unit constants decoded from <gws>/neural_art__network/network_blob_trace.c, the same model as W1's
stem_int_emulator.py (which is 100 % bit-exact vs the board on the P99.95 stem cuts) [from code: W1 NOTES E1].
Rounding of right shifts: ties away from zero (board-observed, W1 NOTES S3).
Compared with the board-like ONNX of the same candidate (W1 v4) on calib-pool images:
  per block, conditional (ORT's int8 block input fed in): 8-bit conv value (GELU input) and block output, % exact;
  chained from the int8 model input: stem output (patch_embed.2 GELU output), % exact and cosine on q - zp.
usage: python -I stem_int_check.py <candidate.onnx> <gws dir> <tag> [n_images=8] [tables.json]  -> results/stem_int_check_<tag>.json"""
import sys, re, json, pathlib
import numpy as np, onnx, onnxruntime as ort, torch
from onnx import numpy_helper as nh
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib
sys.path.insert(0, str(w2lib.R / "night" / "W1_emulator"))
import hwlib  # noqa: E402

cand, gws, tag = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
NI = int(sys.argv[4]) if len(sys.argv) > 4 else 8
WS = gws / "neural_art__network" if (gws / "neural_art__network").exists() else gws
T = (WS / "network_blob_trace.c").read_text()
arr = re.search(r"__ec_params_memory_pool_base\[\d+\] = \{(.*?)\};", T, re.S).group(1)
P = np.array([int(x, 0) for x in re.findall(r"0x[0-9a-fA-F]+|\d+", re.sub(r"/\*.*?\*/", "", arr, flags=re.S))], dtype=np.uint64).tobytes()


def struct(name):
    m = re.search(r"static const \w+ " + re.escape(name) + r" = \{(.*?)\n    \};", T, re.S)
    out = {}
    for k, v in re.findall(r"\.(\w+) = (.*?),\n", m.group(1) + ",\n"):
        off = re.search(r"\+ (\d+)\)", v)
        out[k] = ("@%d" % int(off.group(1))) if off else v.strip()
    return out


def name_of(prefix):
    return re.search(r"static const \w+ (" + re.escape(prefix) + r"_init\d+) = ", T).group(1)


m = onnx.load(cand); g = m.graph
ini = {x.name: nh.to_array(x) for x in g.initializer}; prod = {o: n for n in g.node for o in n.output}
qn = {n.input[0]: n for n in g.node if n.op_type == "QuantizeLinear"}
# stedgeai layer names of the 3 stem blocks (same graph -> same numbering as in the P99.95 compile; checked below)
BL = {"patch_embed.0": ("Conv2D_30", "Conv2D_30", "Div_33"), "patch_embed.1": ("Conv2D_45", "Conv2D_45", "Div_48"),
      "patch_embed.2": ("Conv2D_59", "Conv2D_59_ca_pipe_3", "Div_62")}
meta = {}
for blk, (cv, last, gel) in BL.items():
    c = next(n for n in g.node if n.op_type == "Conv" and n.name.startswith(f"/model/patch_embed/{blk}/"))
    xdq, wdq = prod[c.input[0]], prod[c.input[1]]
    ar = struct(re.search(r"static const \w+ (" + cv + r"_off_bias_\d+_init\d+) = ", T).group(1))
    qc = qn[c.output[0]]; qo = qn[f"/model/patch_embed/{blk}/activation/Mul_1_output_0"]
    meta[blk] = dict(zx=int(ini[xdq.input[2]]), w=ini[wdq.input[0]].astype(np.float64),
                     stride=[a.ints for a in c.attribute if a.name == "strides"][0], pads=[a.ints for a in c.attribute if a.name == "pads"][0],
                     group=([a.i for a in c.attribute if a.name == "group"] or [1])[0],
                     A=np.frombuffer(P, np.int16, 64, int(ar["A_vector"].split("@")[-1].rstrip("}"))).astype(np.int64),
                     C=np.frombuffer(P, np.int32, 64, int(ar["C_vector"].split("@")[-1].rstrip("}"))).astype(np.int64),
                     sh_c=int(struct(name_of(last))["shift_o"]), sh_a=int(ar["shift_o"]),
                     lut=hwlib.activ_lut(str(WS), gel, "half_away"),
                     conv_q=qc.output[0], out_q=qo.output[0], conv_qp=(float(ini[qc.input[1]]), int(ini[qc.input[2]])),
                     out_qp=(float(ini[qo.input[1]]), int(ini[qo.input[2]])), in_q=xdq.input[0])


def rshift(v, n):
    if n == 0: return v
    a = (np.abs(v) + (1 << (n - 1))) >> n; return np.where(v < 0, -a, a)


def block(q, md):
    pd = md["pads"]
    xp = np.pad(q - md["zx"], ((0, 0), (0, 0), (pd[0], pd[2]), (pd[1], pd[3]))).astype(np.float64)
    acc = np.rint(torch.nn.functional.conv2d(torch.from_numpy(xp), torch.from_numpy(md["w"]), stride=tuple(md["stride"]), groups=md["group"]).numpy()).astype(np.int64)
    x16 = np.clip(rshift(acc, md["sh_c"]), -32768, 32767)
    u8 = np.clip(rshift(md["A"][None, :, None, None] * x16 + md["C"][None, :, None, None], md["sh_a"]), -128, 127)
    return u8, md["lut"][u8 + 128]


# board-like ONNX of the candidate with the int8 stem tensors as extra outputs
blp = w2lib.W2 / "tmp" / f"sic_{tag}_bl.onnx"
TBL = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None   # optional: compiled tables (bl_ct.py *.tables.json)
w2lib.BL.make_boardlike(cand, str(blp), tables=TBL)
bm = onnx.load(str(blp))
names = ["input_QuantizeLinear_Output"] + [meta[b][k] for b in BL for k in ("conv_q", "out_q")]
have = {o for n in bm.graph.node for o in n.output}
assert all(t in have for t in names), [t for t in names if t not in have]
for t in names: bm.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
so = ort.SessionOptions(); so.intra_op_num_threads = 3; so.inter_op_num_threads = 1
s = ort.InferenceSession(bm.SerializeToString(), so, providers=["CPUExecutionProvider"])
blp.unlink()
U = np.load(w2lib.R / "night" / "W0_data" / "calib_pool_u8.npy", mmap_mode="r")[:NI]
acc = {}
def add(k, a, b, z):
    a = a.astype(np.int64).ravel(); b = b.astype(np.int64).ravel(); d = np.abs(a - b)
    r = acc.setdefault(k, {"n": 0, "exact": 0, "le1": 0, "max": 0, "xy": 0.0, "xx": 0.0, "yy": 0.0})
    r["n"] += a.size; r["exact"] += int((d == 0).sum()); r["le1"] += int((d <= 1).sum()); r["max"] = max(r["max"], int(d.max()))
    x, y = a - z, b - z; r["xy"] += float(x @ y); r["xx"] += float(x @ x); r["yy"] += float(y @ y)
for i in range(NI):
    x = U[i:i + 1].astype(np.float32) / np.float32(255)
    o = dict(zip(names, s.run(names, {"input": x})))
    q = o["input_QuantizeLinear_Output"].astype(np.int64)
    qc = q
    for b in BL:
        md = meta[b]
        u8c, yc = block(o[md["in_q"]].astype(np.int64) if b != "patch_embed.0" else q, md)   # conditional
        add(f"{b} conv8 (conditional)", u8c, o[md["conv_q"]], md["conv_qp"][1])
        add(f"{b} out (conditional)", yc, o[md["out_q"]], md["out_qp"][1])
        u8, qc = block(qc, md)                                                               # chained
        add(f"{b} out (chained from input)", qc, o[md["out_q"]], md["out_qp"][1])
res = {"candidate": cand, "gws": str(gws), "n_images": NI,
       "units": {b: {"sh_conv": meta[b]["sh_c"], "sh_arith": meta[b]["sh_a"], "conv_q": meta[b]["conv_qp"], "out_q": meta[b]["out_qp"],
                     "A_range": [int(meta[b]["A"].min()), int(meta[b]["A"].max())]} for b in BL},
       "int_emulator_vs_boardlike": {k: {"exact_pct": 100 * r["exact"] / r["n"], "within1_pct": 100 * r["le1"] / r["n"], "maxdiff": r["max"],
                                         "cos": r["xy"] / np.sqrt(r["xx"] * r["yy"])} for k, r in acc.items()}}
json.dump(res, open(w2lib.W2 / "results" / f"stem_int_check_{tag}.json", "w"), indent=1)
print(json.dumps(res["units"]))
for k, r in res["int_emulator_vs_boardlike"].items():
    print(f"{k:38s} exact {r['exact_pct']:6.2f} %  <=1 {r['within1_pct']:6.2f} %  max {r['maxdiff']:3d}  cos {r['cos']:.6f}")
