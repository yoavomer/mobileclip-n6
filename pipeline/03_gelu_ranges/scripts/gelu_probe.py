"""GELU table probe: which ACTIV_FUNC table does the compiler build for a given (conv-output Q, GELU-output Q)?
Builds a tiny QDQ model: input [1,64,8,8] float -> K parallel copies of the P99.95 patch_embed.2 block
(input Q/DQ -> Conv 1x1 int8 (real pe2 weights / bias) -> conv-output Q/DQ (probe params) -> the exact ORT GELU QDQ chain
(Div, Erf, Q/DQ, Add, Q/DQ, Mul, Q/DQ, Mul 0.5) -> GELU-output Q/DQ (probe params)) -> K outputs.
It is compiled with night/tools/compile.sh like any candidate; the K compiled GELU tables are then read with W1's
hwlib.activ_lut (Div_<k> nodes in graph order) and compared with the ideal GELU-once table.
usage: python -I gelu_probe.py build <probe.onnx> <params.json>    params = [[s_in, z_in, s_out, z_out], ...]
       python -I gelu_probe.py read  <probe.onnx> <params.json> <gws dir> <out.json>"""
import sys, json, copy, re, pathlib
import numpy as np, onnx
from onnx import numpy_helper as nh, helper as oh, TensorProto as TP
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w3lib as w2lib  # W3 copy of W2 gelu_probe.py (W2 files untouched)

SRC_IN = "/model/patch_embed/patch_embed.1/activation/Mul_1_output_0"          # float tensor before pe1 output Q
SRC_OUT = "/model/patch_embed/patch_embed.2/activation/Mul_1_output_0_DequantizeLinear_Output"
CONV_OUT = "/model/patch_embed/patch_embed.2/reparam_conv/Conv_output_0"
GOUT = "/model/patch_embed/patch_embed.2/activation/Mul_1_output_0"


def build(out, params):
    m = onnx.shape_inference.infer_shapes(w2lib.load(w2lib.BASE))
    sub = onnx.utils.Extractor(m).extract_model([SRC_IN], [SRC_OUT])
    g = sub.graph
    init = {t.name: t for t in g.initializer}
    nodes, inits, outs = [], [], []
    for k, (s_in, z_in, s_out, z_out) in enumerate(params):
        ren = lambda n: n if n == SRC_IN else f"p{k}_{n}"
        for t in g.initializer:
            c = copy.deepcopy(t); c.name = ren(t.name); inits.append(c)
        for n in g.node:
            c = copy.deepcopy(n); c.name = f"p{k}_{n.name}"
            c.input[:] = [ren(i) if i else i for i in n.input]; c.output[:] = [ren(o) for o in n.output]
            nodes.append(c)
        # new Q params for the conv output and the GELU output of this branch
        for t, (s, z) in ((CONV_OUT, (s_in, z_in)), (GOUT, (s_out, z_out))):
            q = next(n for n in nodes if n.op_type == "QuantizeLinear" and n.input[0] == ren(t))
            sn, zn = f"p{k}_{t}_probe_s", f"p{k}_{t}_probe_z"
            inits += [nh.from_array(np.array(s, np.float32), sn), nh.from_array(np.array(z, np.int8), zn)]
            for n in [q] + [n for n in nodes if n.op_type == "DequantizeLinear" and n.input[0] == q.output[0]]:
                n.input[1] = sn; n.input[2] = zn
        outs.append(oh.make_tensor_value_info(ren(SRC_OUT), TP.FLOAT, [1, 64, 8, 8]))
    used = {i for n in nodes for i in n.input}
    inits = [t for t in inits if t.name in used]
    gg = oh.make_graph(nodes, "gelu_probe", [oh.make_tensor_value_info(SRC_IN, TP.FLOAT, [1, 64, 8, 8])], outs, inits)
    mm = oh.make_model(gg, opset_imports=[oh.make_opsetid("", 17)], ir_version=m.ir_version)
    onnx.checker.check_model(mm)
    onnx.save(mm, out)
    return len(nodes)


def read(params, gws):
    sys.path.insert(0, str(w2lib.R / "night" / "W1_emulator"))
    import hwlib, boardlike as BL
    ws = pathlib.Path(gws); ws = ws / "neural_art__network" if (ws / "neural_art__network").exists() else ws
    eps, _ = hwlib.load_ws(str(ws))
    acts = sorted({c["node"] for l in eps.values() for c in l if c["struct"] == "LL_Activacc_InitTypeDef" and c["node"].startswith("Div_")},
                  key=lambda s: int(re.search(r"_(\d+)", s).group(1)))
    assert len(acts) == len(params), (len(acts), len(params))
    # stedgeai numbers the K parallel branches in REVERSE order (Div_<k> ascending = last branch first) [measured on
    # probe1: reversed pairing gives 0 differing codes vs the full-model compile on the validation branches]
    acts = acts[::-1]
    res = []
    for node, (s_in, z_in, s_out, z_out) in zip(acts, params):
        lut = hwlib.activ_lut(str(ws), node, "half_away")
        ide = np.asarray(BL.gelu_lut(s_in, z_in, s_out, z_out), np.int64)
        _, f, R0, R1 = hwlib.activ_tables(str(ws), node)
        res.append({"params": [s_in, z_in, s_out, z_out], "node": node, "rows": int(len(R1)), "bwidth": int(hwlib.ival(f, "bwidth")),
                    "n_diff": int((lut != ide).sum()), "lut": lut.tolist(), "ideal": ide.tolist()})
    return res


if __name__ == "__main__":
    if sys.argv[1] == "build":
        print(build(sys.argv[2], json.load(open(sys.argv[3]))), "nodes")
    else:
        r = read(json.load(open(sys.argv[3])), sys.argv[4])
        json.dump(r, open(sys.argv[5], "w"))
        for x in r: print(x["node"], x["params"], "rows", x["rows"], "bwidth", x["bwidth"], "diff codes", x["n_diff"])
