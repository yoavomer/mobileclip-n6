"""Board-like model WITH the compiled GELU / Sigmoid tables of a specific compile ("bl+ct").
W1's boardlike v4 assumes every GELU unit computes the ideal GELU rounded once. The compiler builds each ACTIV_FUNC
table as a piecewise polynomial; when it uses few segments the table is off by +-1 LSB on some codes. W1 supports
make_boardlike(..., tables=...) for an exact compiled model; this script builds that `tables` dict without writing into
W1's folder: the 30 compiled GELU nodes (Div_<k>, sorted by k = graph order) and 3 Sigmoid nodes are paired with the
ONNX chains in graph order, the pairing is checked (>= 80 % of the 256 codes equal to the ideal table of that chain),
and each table is read with W1's hwlib.activ_lut(ws, node, "half_away") (the analyst's inferred ACTIV model, which is
bit-exact vs the board on the P99.95 stem cuts). Softmax tables: W1's generic model (bit-exact) is kept.
usage: python -I bl_ct.py <candidate.onnx> <gws dir> <out_boardlike.onnx>  -> prints JSON (per-table code differences)"""
import sys, json, re, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib
sys.path.insert(0, str(w2lib.R / "night" / "W1_emulator"))
import hwlib, boardlike as BL  # noqa: E402


def tables_for(cand, gws):
    ws = pathlib.Path(gws); ws = ws / "neural_art__network" if (ws / "neural_art__network").exists() else ws
    eps, _ = hwlib.load_ws(str(ws))
    acts = sorted({c["node"] for l in eps.values() for c in l if c["struct"] == "LL_Activacc_InitTypeDef"})
    num = lambda s: int(re.search(r"_(\d+)", s).group(1))
    gel = sorted([a for a in acts if a.startswith("Div_")], key=num)
    sig = sorted([a for a in acts if a.startswith("Sigmoid_")], key=num)
    m = w2lib.load(cand)
    chains = w2lib.gelu_chains(m)
    assert len(gel) == len(chains) == 30, (len(gel), len(chains))
    G = BL._G(m)
    sigs = []
    for n in [n for n in m.graph.node if n.op_type == "Sigmoid"]:
        dq = G.prod.get(n.input[0]); q = G.cons.get(n.output[0], [])[0]
        (s_i, z_i), (s_o, z_o) = G.qparams(dq), G.qparams(q)
        sigs.append((float(np.ravel(s_i)[0]), int(np.ravel(z_i)[0]), float(np.ravel(s_o)[0]), int(np.ravel(z_o)[0])))
    assert len(sig) == len(sigs) == 3
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


if __name__ == "__main__":
    cand, gws, out = sys.argv[1:4]
    T, rep = tables_for(cand, gws)
    json.dump(T, open(out + '.tables.json', 'w'))   # keep the compiled tables (gws folders are deleted to save disk)
    r = BL.make_boardlike(cand, out, tables=T)
    used = sum(1 for g in r["gelu"] if str(g.get("lut", "")).startswith("compiled")) + sum(1 for g in r["sigmoid"] if str(g.get("lut", "")).startswith("compiled"))
    print(json.dumps({"out": out, "compiled_tables_used": used, "a2_ok": r["a2_ok"], "tables": rep,
                      "total_codes_differing": sum(x["codes_differing_from_ideal"] for x in rep)}))
