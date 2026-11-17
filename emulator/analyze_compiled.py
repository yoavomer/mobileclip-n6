"""Experiment G1: are ALL activation units (30 GELU, 3 Sigmoid, softmax exp/inv) of a compiled model built the same way as
the stem GELUs ("8-bit input at the producer's ONNX Q -> exact function -> one rounding at the ONNX output Q")?
Per ACTIV unit [from report: c_info.json; from code: network_blob_trace.c via bisect/compiler/trace_parse.py]:
  epoch, HW unit, the preceding unit in the same epoch (what streams into it) and its output width,
  whether the ACTIV node has an input BUFFER (c_info node inputs) or is fed by a stream,
  ACTIV inbytes_f / outbytes_f / operation,
  the input Q (OE: Div input DQ -> Q -> producer tensor, Q.json) and output Q (final Mul 0.5 / Sigmoid output),
  the table evaluated with the analyst's inferred ACTIV_FUNC model vs (a) the ideal function quantised once at those
  ONNX scales, (b) the ONNX QDQ chain as ORT computes it (Erf/Add/Mul QDQ, opt-off arithmetic in float64).
usage: analyze_compiled.py <name> <gws/neural_art__network dir> <OE.onnx> <Q.json>  -> analyze_<name>.json"""
import sys, json, math, pathlib
import numpy as np, onnx
from onnx import numpy_helper as nh
import hwlib
from trace_parse import ival
name, ws, oe_path, qj_path = sys.argv[1:5]
HERE = pathlib.Path(__file__).resolve().parent
ci = json.load(open(pathlib.Path(ws) / "c_info.json"))
B = {b["id"]: b for b in ci["buffers"]}
T, q = hwlib.qjson(qj_path)
oe = onnx.load(oe_path, load_external_data=False); g = oe.graph
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
init = {t.name: t for t in g.initializer}


def init_val(t):
    try: return float(nh.to_array(init[t]).ravel()[0])
    except Exception: return None


# ---- c_info: every leaf unit node, in epoch order
leaf = []
for ep in ci["graphs"][0]["nodes"]:
    for sub in ep.get("subgraph_nodes", []):
        if sub.get("subgraph_nodes"):
            for s in sub["subgraph_nodes"]: leaf.append((sub["name"], s))
        else: leaf.append((ep["name"], sub))
eps, _ = hwlib.load_ws(ws)


def trace_cfg(node):
    for e, lst in eps.items():
        for c in lst:
            if c["node"] == node and c["struct"] in ("LL_Arithacc_InitTypeDef", "LL_Convacc_InitTypeDef", "LL_Activacc_InitTypeDef", "LL_Poolacc_InitTypeDef"):
                return e, c
    return None, None


erf = np.vectorize(math.erf)
x = np.arange(-128, 128)
qz = lambda v, s, z: np.clip(np.round(v / s) + z, -128, 127)


def oe_gelu(div):
    """OE GELU pattern: DQ(in) -> Div -> Erf -> Add -> Q/DQ -> Mul(x) -> Q/DQ -> Mul(0.5) -> Q"""
    dq = prod[div.input[0]]; qin = prod[dq.input[0]]; src = qin.input[0]
    erf_n = cons[div.output[0]][0]; add = cons[erf_n.output[0]][0]
    def nextq(t):
        c = cons[t][0]; assert c.op_type == "QuantizeLinear", c.op_type; return c
    qa = nextq(add.output[0]); dqa = cons[qa.output[0]][0]; mul = cons[dqa.output[0]][0]
    qm = nextq(mul.output[0]); dqm = cons[qm.output[0]][0]; mul1 = cons[dqm.output[0]][0]
    return dict(src=src, add=add.output[0], mul=mul.output[0], out=mul1.output[0],
                div_c=init_val(div.input[1]), add_c=[init_val(i) for i in add.input if i in init],
                half=[init_val(i) for i in mul1.input if i in init])


rows = []
for i, (epname, s) in enumerate(leaf):
    if not str(s.get("attributes", {}).get("HWUnit", "")).startswith("ACTIV"): continue
    nd = s["name"]; e, c = trace_cfg(nd); f = c["fields"]
    prev = leaf[i - 1][1] if i > 0 and leaf[i - 1][0] == epname else None
    pe, pc = trace_cfg(prev["name"]) if prev else (None, None)
    r = {"node": nd, "type": s["type"], "epoch": e, "hwunit": s["attributes"]["HWUnit"], "operation": ival(f, "operation"),
         "inbytes_f": ival(f, "inbytes_f"), "outbytes_f": ival(f, "outbytes_f"), "outbytes_o": ival(f, "outbytes_o"),
         "rom1_rows": ival(f, "ROM1_nbytes") // 6, "bwidth": ival(f, "bwidth"),
         "input_buffers": [B[b]["name"] + f" ({B[b]['format']}, {B[b]['nbits']} bit)" for b in s.get("inputs", [])],
         "prev_unit": None if prev is None else f'{prev["name"]} {prev["type"]} {prev["attributes"].get("HWUnit")}',
         "prev_outbytes_o": None if pc is None else ival(pc["fields"], "outbytes_o"),
         "output_buffers": [B[b]["name"] + f" ({B[b]['format']}, {B[b]['nbits']} bit, " + str(B[b].get("intq", {}).get("scales", ["?"])[0]) + "/" +
                            str(B[b].get("intq", {}).get("offsets", ["?"])[0]) + ")" for b in s.get("outputs", [])]}
    lut = hwlib.activ_lut(ws, nd)
    if s["type"] == "Gelu":
        div = next(n for n in g.node if n.name == nd)
        gg = oe_gelu(div)
        (si, zi), (so, zo) = q(gg["src"]), q(gg["out"])
        (sa, za), (sm, zm) = q(gg["add"]), q(gg["mul"])
        xf = si * (x - zi)
        ideal = qz(0.5 * xf * (1 + erf(xf / math.sqrt(2))), so, zo)
        # ONNX QDQ chain as quantized by ORT: Div (no Q) -> Erf -> Q(erf) -> Add 1 -> Q(add) -> Mul x -> Q(mul) -> Mul .5 -> Q(out)
        # (the OE graph has dropped the Erf Q; the original ONNX has one: we take its scale from the ONNX model when given)
        chain = None
        r.update(src=gg["src"], in_q=[si, zi], out_q=[so, zo], add_q=[sa, za], mul_q=[sm, zm], div_const=gg["div_c"],
                 add_const=gg["add_c"], half_const=gg["half"])
        d1 = lut - ideal
        r["hw_vs_ideal_once"] = {"n_diff": int(np.count_nonzero(d1)), "max": int(np.abs(d1).max()),
                                 "at_q": x[d1 != 0].tolist()[:16]}
    elif s["type"] in ("Sigmoid", "Logistic") or nd.startswith("Sigmoid"):
        sg = next(n for n in g.node if n.name == nd)
        dq = prod[sg.input[0]]; src = prod[dq.input[0]].input[0]
        (si, zi), (so, zo) = q(src), q(sg.output[0])
        ideal = qz(1 / (1 + np.exp(-si * (x - zi))), so, zo)
        d1 = lut - ideal
        r.update(src=src, in_q=[si, zi], out_q=[so, zo], hw_vs_ideal_once={"n_diff": int(np.count_nonzero(d1)), "max": int(np.abs(d1).max()),
                                                                           "at_q": x[d1 != 0].tolist()[:16]})
    rows.append(r)
json.dump(rows, open(HERE / f"analyze_{name}.json", "w"), indent=1)
ng = sum(r["type"] == "Gelu" for r in rows)
print(f"{name}: {len(rows)} ACTIV units, {ng} Gelu")
for r in rows:
    hv = r.get("hw_vs_ideal_once")
    print(f'{r["node"]:34s} {r["type"]:8s} ep{r["epoch"]:<4} {r["operation"]} in{r["inbytes_f"]}B out{r["outbytes_o"]}B prev=[{r["prev_unit"]} out{r["prev_outbytes_o"]}B] '
          f'inbuf={r["input_buffers"]} in_q={r.get("in_q")} out_q={r.get("out_q")} '
          + (f'HW-vs-ideal: {hv["n_diff"]}/256 max {hv["max"]}' if hv else ""))
