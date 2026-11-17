"""Board-like WITH the compiled GELU / Sigmoid tables of a specific compile ("bl+ct").
Copy of W2's method (night/W2_stem/scripts/bl_ct.py, E6-E8 of W2's NOTES; W2's files are not touched): the 30 compiled
GELU ACTIV nodes (Div_<k>, sorted by k = graph order) and the 3 Sigmoid nodes are paired with the ONNX chains in graph
order; the pairing is checked (<= 51 of 256 codes may differ from the ideal table of that chain); each table is read with
W1's hwlib.activ_lut(ws, node, "half_away") (bit-exact vs the board on the P99.95 stem cuts [from W1/W2 notes]).
Softmax: W1's generic model (bit-exact) is kept.
API: tables_for(cand_onnx, gws) -> (tables, report);  blct_bytes(cand_onnx, gws) -> (board-like bytes, report)
CLI: blct.py <cand.onnx> <gws dir>  -> prints JSON report (codes differing per table)"""
import sys, re, json, copy, pathlib, os, time
import numpy as np, onnx
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w3lib as L
sys.path.insert(0, str(L.R / "night" / "W1_emulator"))
import hwlib  # noqa: E402  (W1, read-only import)


def _bl_module():
    L.boardlike_fn()
    return sys.modules[f"w1_boardlike_{L._BL['sha'][:12]}"]


def chains_q(m, BL):
    G = BL._G(copy.deepcopy(m)); out = []
    for f in BL._find_gelus(G):
        (s_i, z_i), (s_o, z_o) = G.qparams(f["dq_in"]), G.qparams(f["q_out"])
        out.append(dict(x=G.prod[f["x_q"]].input[0], s_in=float(np.ravel(s_i)[0]), z_in=int(np.ravel(z_i)[0]),
                        s_out=float(np.ravel(s_o)[0]), z_out=int(np.ravel(z_o)[0])))
    sigs = []
    for n in [n for n in m.graph.node if n.op_type == "Sigmoid"]:
        dq = G.prod.get(n.input[0]); q = G.cons.get(n.output[0], [])[0]
        (s_i, z_i), (s_o, z_o) = G.qparams(dq), G.qparams(q)
        sigs.append((float(np.ravel(s_i)[0]), int(np.ravel(z_i)[0]), float(np.ravel(s_o)[0]), int(np.ravel(z_o)[0])))
    return out, sigs


def tables_for(cand, gws):
    BL = _bl_module()
    ws = pathlib.Path(gws); ws = ws / "neural_art__network" if (ws / "neural_art__network").exists() else ws
    eps, _ = hwlib.load_ws(str(ws))
    acts = sorted({c["node"] for l in eps.values() for c in l if c["struct"] == "LL_Activacc_InitTypeDef"})
    num = lambda s: int(re.search(r"_(\d+)", s).group(1))
    gel = sorted([a for a in acts if a.startswith("Div_")], key=num)
    sig = sorted([a for a in acts if a.startswith("Sigmoid_")], key=num)
    m = L.load(cand) if not isinstance(cand, onnx.ModelProto) else cand
    chains, sigs = chains_q(m, BL)
    assert len(gel) == len(chains) == 30, (len(gel), len(chains))
    assert len(sig) == len(sigs) == 3, (len(sig), len(sigs))
    T = {"gelu": [], "sigmoid": [], "softmax": []}; rep = []
    for node, c in zip(gel, chains):
        lut = hwlib.activ_lut(str(ws), node, "half_away")
        ide = np.asarray(BL.gelu_lut(c["s_in"], c["z_in"], c["s_out"], c["z_out"]), np.int64)
        nd = int((lut != ide).sum()); assert nd <= 51, (node, c["x"], nd)
        T["gelu"].append({"node": node, "in_q": [c["s_in"], c["z_in"]], "out_q": [c["s_out"], c["z_out"]], "lut": lut.tolist()})
        rep.append({"node": node, "tensor": c["x"], "codes_differing_from_ideal": nd, "max_abs": int(np.abs(lut - ide).max())})
    for node, (s_i, z_i, s_o, z_o) in zip(sig, sigs):
        lut = hwlib.activ_lut(str(ws), node, "half_away")
        ide = np.asarray(BL.sigmoid_lut(s_i, z_i, s_o, z_o), np.int64)
        nd = int((lut != ide).sum()); assert nd <= 51, (node, nd)
        T["sigmoid"].append({"node": node, "in_q": [s_i, z_i], "out_q": [s_o, z_o], "lut": lut.tolist()})
        rep.append({"node": node, "codes_differing_from_ideal": nd, "max_abs": int(np.abs(lut - ide).max())})
    return T, rep


def blct_bytes(cand, gws):
    BL = _bl_module()
    T, rep = tables_for(cand, gws)
    L.TMP.mkdir(parents=True, exist_ok=True)
    dst = L.TMP / f"blct_{os.getpid()}_{time.time_ns()}.onnx"
    try:
        r = BL.make_boardlike(str(cand), str(dst), tables=T)
        b = dst.read_bytes()
    finally:
        try: os.remove(dst)
        except OSError: pass
    used = sum(1 for g in r["gelu"] if str(g.get("lut", "")).startswith("compiled")) + \
        sum(1 for g in r["sigmoid"] if str(g.get("lut", "")).startswith("compiled"))
    return b, {"compiled_tables_used": used, "tables": rep, "a2_ok": r["a2_ok"],
               "total_codes_differing": sum(x["codes_differing_from_ideal"] for x in rep)}


if __name__ == "__main__":
    T, rep = tables_for(sys.argv[1], sys.argv[2])
    print(json.dumps({"total_codes_differing": sum(x["codes_differing_from_ideal"] for x in rep),
                      "nonzero": [x for x in rep if x["codes_differing_from_ideal"]]}, indent=1))
