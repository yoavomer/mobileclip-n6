"""W1 board-like rewrite of a QDQ ONNX model (MobileCLIP-S0 on the OpenMV N6 / ST Neural-ART, stedgeai v4.0.0).

    from boardlike import make_boardlike
    info = make_boardlike("model.onnx", "model_boardlike.onnx")      # returns a dict (what was rewritten)

The result is a normal ONNX model (same input/output names) that ORT runs (any optimization level). It changes only
the ops whose on-board arithmetic is known from evidence (evidence per rule: NOTES.md; how to use: USAGE.md):

 R0 no QDQ fusion: every DequantizeLinear is replaced by the same float32 arithmetic ((float(x)-zp)*scale; constants
          pre-computed), so ORT cannot fuse DQ->op->Q into QLinearConv/QLinearMul/QGemm/... and computes with "opt-off"
          arithmetic even under ORT_ENABLE_ALL. (Board is closer to opt-off; ORT-default fusions differ from the board
          by up to 34 LSB on the layer-scale Mul, 2 LSB with opt-off.)
 R1 GELU  (Div sqrt2 -> Erf -> Add 1 -> Mul x -> Mul 0.5, with QDQ pairs inside): the board has ONE ACTIV_FUNC unit,
          8-bit input = the int8 tensor at the chain INPUT's Q (conv output / SE-Mul output), 8-bit output at the chain
          OUTPUT's Q (Mul_1). No rounding inside. Rewrite: DQ(x_q) -> exact float32 GELU -> Q(out) (gelu_impl="float",
          default, == float64 LUT on all 256 codes of all 30 GELUs of P99.95 and a2) or y_q = LUT[x_q] (gelu_impl="lut").
 R2 Softmax over 64 keys (attention, --expand-softmax): the board runs an int8 pipeline
          d = x - max + 127 ; e = EXP[d] (= -round(128 exp(s_in (d-127)))) ; avg = round_half_away(sum e / 64) ;
          inv = INV_LUT[avg] (compiled table, ~ -round(256/-avg)) ; u = e*inv ; u2 = u + 128 inv ;
          out = sat8(round_half_away(((A u2) >> 7) + B inv + (C << Cs)) >> sh)   (A,B,C,sh from the output Q).
          Bit-exact vs the board on 2 x 65,536 softmax values of the qkvconv model (board logits in).
          Rewrite: integer ONNX ops (Cast/ReduceMax/Gather/ReduceSum/Div) + one 256x256 int8 LUT for the last stage.
 R3 attention MatMul of two activations (SW integer conv on the board): the operand that stedgeai passes as the conv
          WEIGHTS (= ONNX input A) is used without its zero point. Rewrite: DQ of input A with zp 0.
          (No-op on A2 models, whose A operands have zp 0 already; kept so the pre-A2 board dumps can be reproduced.)
 R4 Sigmoid (DQ -> Sigmoid -> Q): ACTIV_FUNC table = ideal sigmoid quantized once. Rewrite: y_q = LUT[x_q] (same as
          ORT up to float32 rounding).

Everything else is left to ORT (convs, BN, adds, muls, pools, reshapes ...). NOT modelled: the NPU's 16-bit requant
coefficients / conv accumulator shifts (the remaining 2-4 % non-exact values in the stem), the board's non-model data
in the C0 output buffer, anything of the hybrid/SW Split/Transpose kernels (assumed exact).
Optional tables=compiled_tables(...) replaces the ideal GELU/Sigmoid/exp/inverse LUTs by the compiled ACTIV tables
(analyst's inferred table model; exact compiled model only; NOT board-validated beyond the stem, default off).
"""
VERSION = "W1-boardlike 2026-11-08 v4 (R0 no-QDQ-fusion, R1 float GELU, R2 softmax, R3 wzp, R4 sigmoid)"
import math, os, copy
import numpy as np
import onnx
from onnx import helper as oh, numpy_helper as nh, TensorProto as TP

SQRT2 = math.sqrt(2.0)
# Inverse table of the softmax expansion, index avg+128 (avg = int8 mean of the exp codes). Evaluated from the compiled
# ACTIV table with ties rounded away from zero; identical in the qkvconv, a2 and P99.95 compiles (both blocks) and
# bit-exact with the board's softmax output [measured, exp_softmax_board.py / validate_attn.py]. Ideal would be
# -round(256/-avg); the table differs from that at avg -80 and -74 (-4 instead of -3) and at avg -128 (-128).
INV_LUT = [-128, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -4, -3, -3, -3, -3, -3, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -6, -6, -6, -6, -6, -6, -6, -7, -7, -7, -7, -7, -8, -8, -8, -8, -9, -9, -9, -9, -10, -10, -11, -11, -12, -12, -13, -13, -14, -15, -16, -17, -18, -20, -21, -23, -26, -28, -32, -37, -43, -51, -64, -85, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128]


def _inv_lut(): return np.asarray(INV_LUT, np.int64)


def _round_away(r, sh):
    """right shift by sh with ties rounded away from zero (board behaviour, see NOTES.md S3)"""
    a = (np.abs(r) + (1 << (sh - 1))) >> sh
    return np.where(r < 0, -a, a)


def _q(v, s, z): return np.clip(np.round(v / s) + z, -128, 127)


def _erf(a): return np.array([math.erf(t) for t in np.ravel(a)]).reshape(np.shape(a))


def gelu_lut(s_in, z_in, s_out, z_out):
    x = s_in * (np.arange(-128, 128) - z_in)
    return _q(0.5 * x * (1.0 + _erf(x / SQRT2)), s_out, z_out).astype(np.int8)


def sigmoid_lut(s_in, z_in, s_out, z_out):
    x = s_in * (np.arange(-128, 128) - z_in)
    return _q(1.0 / (1.0 + np.exp(-x)), s_out, z_out).astype(np.int8)


def softmax_exp_lut(s_in):
    d = np.arange(-128, 128)
    return np.clip(-np.round(128.0 * np.exp(s_in * (d - 127))), -128, 127).astype(np.int64)


def softmax_final_params(s_out, z_out):
    """sub2 unit constants. Observed: (1/255,-128) -> A 32640 B -32640 C -32768 Cs 6 sh 14; (1/127,0) -> A 32512 B -32512
    C 0 sh 15 [from code: network_blob_trace.c of qkvconv / a2 / P99.95]. Generic rule reproducing both: A = round(2^(sh-7)/s_out)
    with the largest sh such that A <= 32767; B = -A; C << Cs = z_out * 2^sh."""
    inv_s = 1.0 / s_out
    sh = 7
    while round(inv_s * 2 ** (sh + 1 - 7)) <= 32767: sh += 1
    A = int(round(inv_s * 2 ** (sh - 7)))
    return A, -A, int(z_out) << sh, sh


def softmax_final_lut(s_out, z_out):
    """LUT2[(e+128)*256 + (inv+128)] -> int8 output, exact integer emulation of mul -> mul_sub2 -> sub2."""
    A, B, CC, sh = softmax_final_params(s_out, z_out)
    e = np.arange(-128, 128, dtype=np.int64)[:, None]; inv = np.arange(-128, 128, dtype=np.int64)[None, :]
    u = np.clip(e * inv, -32768, 32767)                                    # ARITH_MUL, 16-bit out
    u2 = ((16384 * u) >> 7) + 16384 * inv                                  # mul_sub2: A=B=16384, Ax 7, shift_o 7
    u2 = np.clip(_round_away(u2, 7), -32768, 32767)                        # exact (multiple of 128)
    r = ((A * u2) >> 7) + B * inv + CC                                      # sub2
    return np.clip(_round_away(r, sh), -128, 127).astype(np.int8).ravel()


# --------------------------------------------------------------------------------------------------------- graph utils
class _G:
    def __init__(self, m):
        self.m = m; self.g = m.graph
        self.init = {t.name: t for t in self.g.initializer}
        self.refresh()
        self.k = 0

    def refresh(self):
        self.prod = {o: n for n in self.g.node for o in n.output}
        self.cons = {}
        for n in self.g.node:
            for i in n.input: self.cons.setdefault(i, []).append(n)
        self.outs = {o.name for o in self.g.output}

    def const(self, name):
        """value of a constant tensor: initializer, Constant node, or DequantizeLinear of constants."""
        if name in self.init: return nh.to_array(self.init[name])
        p = self.prod.get(name)
        if p is None: return None
        if p.op_type == "Constant":
            return nh.to_array(next(a.t for a in p.attribute if a.name == "value"))
        if p.op_type == "DequantizeLinear":
            x = self.const(p.input[0]); s = self.const(p.input[1])
            z = self.const(p.input[2]) if len(p.input) > 2 and p.input[2] else 0
            if x is None or s is None: return None
            return (x.astype(np.float64) - np.asarray(z, np.float64)) * np.asarray(s, np.float64)
        if p.op_type == "Identity": return self.const(p.input[0])
        return None

    def qparams(self, node):
        s = self.const(node.input[1]); z = self.const(node.input[2]) if len(node.input) > 2 and node.input[2] else np.array(0, np.uint8)
        return s, z

    def name(self, base):
        self.k += 1; return f"bl_{base}_{self.k}"

    def add_init(self, arr, base):
        n = self.name(base); t = nh.from_array(arr, n); self.g.initializer.append(t); self.init[n] = t; return n


def _scalar(v):
    v = np.asarray(v); assert v.size == 1, f"per-channel Q not supported here ({v.shape})"; return v.ravel()[0]


def _skip_qdq(G, t):
    """follow t through Q -> DQ pairs (single consumer each); return the float tensor name after them."""
    while True:
        c = G.cons.get(t, [])
        if len(c) == 1 and c[0].op_type == "QuantizeLinear":
            c2 = G.cons.get(c[0].output[0], [])
            if len(c2) == 1 and c2[0].op_type == "DequantizeLinear" and c[0].output[0] not in G.outs:
                t = c2[0].output[0]; continue
        return t


def _is_const(G, name, val, tol):
    v = G.const(name)
    return v is not None and np.asarray(v).size == 1 and abs(float(np.asarray(v).ravel()[0]) - val) <= tol


def _lut_nodes(G, x_q, lut_int8, out_name, base):
    """y = Gather(lut, int32(x_q) + 128) with output name out_name."""
    # Gather with negative indices (ONNX/ORT: index -k counts from the end): table rolled so that T[q] = lut[q + 128]
    t_lut = G.add_init(np.roll(np.asarray(lut_int8, np.int8), 128), base + "_lut")
    a = G.name(base + "_i32")
    return [oh.make_node("Cast", [x_q], [a], to=TP.INT32, name=G.name(base + "_cast")),
            oh.make_node("Gather", [t_lut, a], [out_name], axis=0, name=G.name(base + "_gather"))]


# ------------------------------------------------------------------------------------------------------------- rules
def _find_gelus(G):
    found = []
    for erf in [n for n in G.g.node if n.op_type == "Erf"]:
        div = G.prod.get(erf.input[0])
        if div is None or div.op_type != "Div" or not _is_const(G, div.input[1], SQRT2, 1e-3): continue
        xf = div.input[0]; dq_in = G.prod.get(xf)
        if dq_in is None or dq_in.op_type != "DequantizeLinear": continue
        t = _skip_qdq(G, erf.output[0]); c = G.cons.get(t, [])
        if len(c) != 1 or c[0].op_type != "Add": continue
        add = c[0]; other = [i for i in add.input if i != t]
        if len(other) != 1 or not _is_const(G, other[0], 1.0, 1e-2): continue
        t = _skip_qdq(G, add.output[0]); muls = []
        ok = True
        for _ in range(2):
            c = G.cons.get(t, [])
            if len(c) != 1 or c[0].op_type != "Mul": ok = False; break
            mu = c[0]; o = [i for i in mu.input if i != t]
            if len(o) != 1: ok = False; break
            po = G.prod.get(o[0])
            if o[0] == xf or (po is not None and po.op_type == "DequantizeLinear" and po.input[0] == dq_in.input[0]): muls.append("x")
            elif _is_const(G, o[0], 0.5, 1e-2): muls.append("half")
            else: ok = False; break
            last = mu; t = _skip_qdq(G, mu.output[0]) if _ == 0 else mu.output[0]
        if not ok or sorted(muls) != ["half", "x"]: continue
        c = G.cons.get(last.output[0], [])
        if len(c) != 1 or c[0].op_type != "QuantizeLinear": continue
        q_out = c[0]
        if G.const(dq_in.input[1]) is None: continue
        found.append(dict(erf=erf, dq_in=dq_in, x_q=dq_in.input[0], q_out=q_out))
    return found


def _rule_gelu(G, tables=None, impl="lut"):
    rep = []
    gel = _find_gelus(G)
    new = []
    for i, f in enumerate(gel):
        s_in, z_in = G.qparams(f["dq_in"]); s_o, z_o = G.qparams(f["q_out"])
        s_in, z_in, s_o, z_o = float(_scalar(s_in)), int(_scalar(z_in)), float(_scalar(s_o)), int(_scalar(z_o))
        if np.asarray(G.const(f["q_out"].input[2])).dtype != np.int8 or np.asarray(G.const(f["dq_in"].input[2])).dtype != np.int8:
            rep.append({"erf": f["erf"].name, "skipped": "chain input/output is not int8 (e.g. INT16 QDQ): not modelled"}); continue
        lut = gelu_lut(s_in, z_in, s_o, z_o); src = "ideal"
        if tables is not None and i < len(tables["gelu"]):
            tq = tables["gelu"][i]
            if np.isclose(tq["in_q"][0], s_in, rtol=1e-6) and tq["in_q"][1] == z_in and np.isclose(tq["out_q"][0], s_o, rtol=1e-6) and tq["out_q"][1] == z_o:
                lut = np.asarray(tq["lut"], np.int8); src = "compiled:" + tq["node"]
            else: src = "ideal (compiled table scale mismatch)"
        out = f["q_out"].output[0]
        G.g.node.remove(f["q_out"])
        if impl == "float" and src == "ideal":
            new.append((f["dq_in"].output[0], _gelu_float_nodes(G, f["dq_in"].output[0], s_o, z_o, out)))
            src = "float32 exact-GELU chain (no inner QDQ)"
        else:
            new.append((f["x_q"], _lut_nodes(G, f["x_q"], lut, out, "gelu")))
        rep.append({"erf": f["erf"].name, "in_q": [s_in, z_in], "out_q": [s_o, z_o], "lut": src})
    return new, rep


def _gelu_float_nodes(G, xf, s_o, z_o, out_name):
    """xf (float, = DQ of the int8 chain input) -> 0.5*x*(1+erf(x/sqrt2)) in float32 -> Q(s_o, z_o) named out_name."""
    c2 = G.add_init(np.array(SQRT2, np.float32), "g_sqrt2"); c1 = G.add_init(np.array(1.0, np.float32), "g_one")
    ch = G.add_init(np.array(0.5, np.float32), "g_half")
    cs = G.add_init(np.array(s_o, np.float32), "g_so"); cz = G.add_init(np.array(z_o, np.int8), "g_zo")
    t = [G.name("g_" + k) for k in ("div", "erf", "add", "mul", "mul1")]
    nodes = [oh.make_node("Div", [xf, c2], [t[0]]), oh.make_node("Erf", [t[0]], [t[1]]), oh.make_node("Add", [t[1], c1], [t[2]]),
             oh.make_node("Mul", [xf, t[2]], [t[3]]), oh.make_node("Mul", [t[3], ch], [t[4]]),
             oh.make_node("QuantizeLinear", [t[4], cs, cz], [out_name])]
    for nd in nodes: nd.name = G.name("gf_" + nd.op_type)
    return nodes


def _rule_sigmoid(G, tables=None):
    rep, new = [], []
    sig = [n for n in G.g.node if n.op_type == "Sigmoid"]
    for i, n in enumerate(sig):
        dq = G.prod.get(n.input[0]); c = G.cons.get(n.output[0], [])
        if dq is None or dq.op_type != "DequantizeLinear" or len(c) != 1 or c[0].op_type != "QuantizeLinear" or n.output[0] in G.outs: continue
        q_out = c[0]
        s_in, z_in = G.qparams(dq); s_o, z_o = G.qparams(q_out)
        if np.asarray(z_in).dtype != np.int8 or np.asarray(z_o).dtype != np.int8:
            rep.append({"sigmoid": n.name, "skipped": "not int8"}); continue
        s_in, z_in, s_o, z_o = float(_scalar(s_in)), int(_scalar(z_in)), float(_scalar(s_o)), int(_scalar(z_o))
        lut = sigmoid_lut(s_in, z_in, s_o, z_o); src = "ideal"
        if tables is not None and i < len(tables["sigmoid"]):
            tq = tables["sigmoid"][i]
            if np.isclose(tq["in_q"][0], s_in, rtol=1e-6) and tq["in_q"][1] == z_in:
                lut = np.asarray(tq["lut"], np.int8); src = "compiled:" + tq["node"]
        out = q_out.output[0]; G.g.node.remove(q_out)
        new.append((dq.input[0], _lut_nodes(G, dq.input[0], lut, out, "sigm")))
        rep.append({"sigmoid": n.name, "in_q": [s_in, z_in], "out_q": [s_o, z_o], "lut": src})
    return new, rep


def _rule_softmax(G, tables=None, shapes=None):
    rep, new = [], []
    for i, n in enumerate([n for n in G.g.node if n.op_type == "Softmax"]):
        axis = next((oh.get_attribute_value(a) for a in n.attribute if a.name == "axis"), -1)
        dq = G.prod.get(n.input[0]); c = G.cons.get(n.output[0], [])
        if dq is None or dq.op_type != "DequantizeLinear" or len(c) != 1 or c[0].op_type != "QuantizeLinear":
            rep.append({"softmax": n.name, "skipped": "no DQ->Softmax->Q pattern"}); continue
        shp = (shapes or {}).get(n.input[0])
        if shp is None or len(shp) == 0 or shp[-1] != 64 or axis not in (-1, len(shp) - 1):
            rep.append({"softmax": n.name, "skipped": f"shape {shp} axis {axis}: only last-axis softmax over 64 is modelled"}); continue
        q_out = c[0]
        s_in, z_in = G.qparams(dq); s_o, z_o = G.qparams(q_out)
        if np.asarray(z_in).dtype != np.int8 or np.asarray(z_o).dtype != np.int8:
            rep.append({"softmax": n.name, "skipped": "not int8"}); continue
        s_in, s_o, z_o = float(_scalar(s_in)), float(_scalar(s_o)), int(_scalar(z_o))
        e_lut = softmax_exp_lut(s_in); inv = _inv_lut(); src = "formula"
        if tables is not None and i < len(tables["softmax"]):
            tq = tables["softmax"][i]
            if np.isclose(tq["s_in"], s_in, rtol=1e-6):
                e_lut = np.asarray(tq["exp_lut"], np.int64); inv = np.asarray(tq["inv_lut"], np.int64); src = "compiled"
        fin = softmax_final_lut(s_o, z_o)
        x_q = dq.input[0]; out = q_out.output[0]; G.g.node.remove(q_out)
        b = "smx"
        t_e = G.add_init(e_lut.astype(np.int32), b + "_exp"); t_i = G.add_init(inv.astype(np.int32), b + "_inv")
        t_f = G.add_init(fin, b + "_fin")
        c127 = G.add_init(np.array(127, np.int32), b + "_c127"); c128 = G.add_init(np.array(128, np.int32), b + "_c128")
        assert e_lut.max() <= 0, "exp table with positive codes: pooling sign assumption broken"
        cav = G.add_init(np.array(32, np.int32), b + "_cav"); c64 = G.add_init(np.array(64, np.int32), b + "_c64")
        c0 = G.add_init(np.array(0, np.int32), b + "_c0")
        cmin = G.add_init(np.array(-128, np.int32), b + "_cmin"); cmax = G.add_init(np.array(127, np.int32), b + "_cmax")
        c256 = G.add_init(np.array(256, np.int32), b + "_c256"); ax = G.add_init(np.array([-1], np.int64), b + "_ax")
        T = {k: G.name(b + "_" + k) for k in ("x", "mx", "d0", "d", "di", "e", "s", "s2", "a0", "a1", "a2", "a3", "ai", "inv", "ei", "ii", "e256", "idx")}
        nodes = [oh.make_node("Cast", [x_q], [T["x"]], to=TP.INT32),
                 oh.make_node("ReduceMax", [T["x"]], [T["mx"]], axes=[-1], keepdims=1),
                 oh.make_node("Sub", [T["x"], T["mx"]], [T["d0"]]),
                 oh.make_node("Add", [T["d0"], c127], [T["d"]]),              # d = x - max + 127  (sub1 unit, exact on the grid)
                 oh.make_node("Add", [T["d"], c128], [T["di"]]),
                 oh.make_node("Gather", [t_e, T["di"]], [T["e"]], axis=0),    # exp table
                 oh.make_node("ReduceSum", [T["e"], ax], [T["s"]], keepdims=1),
                 oh.make_node("Sub", [cav, T["s"]], [T["s2"]]),                # 32 - sum >= 0 (sum <= 0)
                 oh.make_node("Div", [T["s2"], c64], [T["a0"]]),               # (32 - sum) div 64  (non-negative int div)
                 oh.make_node("Sub", [c0, T["a0"]], [T["a1"]]),                # avg = round_half_away(sum/64)
                 oh.make_node("Max", [T["a1"], cmin], [T["a2"]]),
                 oh.make_node("Min", [T["a2"], cmax], [T["a3"]]),
                 oh.make_node("Add", [T["a3"], c128], [T["ai"]]),
                 oh.make_node("Gather", [t_i, T["ai"]], [T["inv"]], axis=0),  # inverse table
                 oh.make_node("Add", [T["e"], c128], [T["ei"]]),
                 oh.make_node("Add", [T["inv"], c128], [T["ii"]]),
                 oh.make_node("Mul", [T["ei"], c256], [T["e256"]]),
                 oh.make_node("Add", [T["e256"], T["ii"]], [T["idx"]]),
                 oh.make_node("Gather", [t_f, T["idx"]], [out], axis=0)]      # mul -> mul_sub2 -> sub2 as one LUT
        for k, nd in enumerate(nodes): nd.name = G.name(b + f"_n{k}")
        new.append((x_q, nodes))
        rep.append({"softmax": n.name, "s_in": s_in, "out_q": [s_o, z_o], "tables": src, "final_params": list(softmax_final_params(s_o, z_o))})
    return new, rep


def _rule_matmul_wzp(G):
    rep = []
    for n in [n for n in G.g.node if n.op_type == "MatMul"]:
        pa, pb = G.prod.get(n.input[0]), G.prod.get(n.input[1])
        if pa is None or pb is None or pa.op_type != "DequantizeLinear" or pb.op_type != "DequantizeLinear": continue
        if G.const(pa.input[0]) is not None or G.const(pb.input[0]) is not None: continue   # one operand constant: HW Gemm/conv
        z = G.const(pa.input[2]) if len(pa.input) > 2 and pa.input[2] else np.zeros(1)
        zv = int(_scalar(z))
        r = {"matmul": n.name, "A_operand": pa.input[0], "A_zp": zv}
        if zv != 0:
            zn = G.add_init(np.zeros_like(np.asarray(z)), "wzp0")
            nd = oh.make_node("DequantizeLinear", [pa.input[0], pa.input[1], zn], [G.name("wzp0_out")], name=G.name("wzp0_dq"))
            n.input[0] = nd.output[0]
            G.g.node.insert(list(G.g.node).index(n), nd)
            r["rewritten"] = True
        else: r["rewritten"] = False
        rep.append(r)
    return rep


def _toposort_insert(G, new):
    """insert each new node list right after the producer of its input tensor (or at the top)."""
    for x_q, nodes in new:
        p = G.prod.get(x_q)
        idx = (list(G.g.node).index(p) + 1) if p is not None else 0
        for k, nd in enumerate(nodes): G.g.node.insert(idx + k, nd)
        for nd in nodes:
            for o in nd.output: G.prod[o] = nd


def _dce(G):
    G.refresh()
    need = set(o.name for o in G.g.output); keep = []
    for n in reversed(list(G.g.node)):
        if any(o in need for o in n.output):
            keep.append(n); need.update(i for i in n.input if i)
    removed = len(G.g.node) - len(keep)
    keep.reverse()
    del G.g.node[:]; G.g.node.extend(keep)
    used = set(i for n in G.g.node for i in n.input)
    inits = [t for t in G.g.initializer if t.name in used or t.name in {o.name for o in G.g.output}]
    del G.g.initializer[:]; G.g.initializer.extend(inits)
    return removed


def _rule_no_qdq_fusion(G):
    """R0: replace every standard-domain DequantizeLinear by plain float ops with the SAME float32 arithmetic
    ((float(x) - zp) * scale): constant inputs -> float initializer computed in numpy exactly like ORT's DQ kernel,
    activation inputs -> Cast(float) -> Sub(zp) -> Mul(scale). ORT's QDQ fusions (QLinearConv/QLinearMul/QGemm/...) then
    cannot trigger, so ORT computes with 'opt-off' arithmetic at ANY optimization level (the board is closer to opt-off:
    see NOTES.md, rule R0). QuantizeLinear nodes are kept (one rounding per ONNX Q, like the board)."""
    n_const = n_act = 0
    for n in [n for n in G.g.node if n.op_type == "DequantizeLinear" and n.domain in ("", "ai.onnx")]:
        x, s = G.const(n.input[0]) if n.input[0] in G.init else None, G.const(n.input[1])
        z = G.const(n.input[2]) if len(n.input) > 2 and n.input[2] else None
        if s is None or (len(n.input) > 2 and n.input[2] and z is None): continue
        axis = next((oh.get_attribute_value(a) for a in n.attribute if a.name == "axis"), 1)
        if x is not None:
            xi = np.asarray(x)
            sv = np.asarray(s, np.float32); zv = np.zeros_like(sv, np.int32) if z is None else np.asarray(z).astype(np.int32)
            if sv.size > 1:
                shp = [1] * xi.ndim; shp[axis % xi.ndim] = sv.size; sv = sv.reshape(shp); zv = zv.reshape(shp)
            val = ((xi.astype(np.int64) - zv).astype(np.float32) * sv).astype(np.float32)
            G.g.initializer.append(nh.from_array(val, n.output[0])); G.init[n.output[0]] = G.g.initializer[-1]
            G.g.node.remove(n); n_const += 1
        else:
            sv = np.asarray(s, np.float32)
            if sv.size != 1: continue
            zf = 0.0 if z is None else float(np.asarray(z).ravel()[0])
            cz = G.add_init(np.array(zf, np.float32), "dq_zp"); cs = G.add_init(sv.reshape(()), "dq_s")
            a, b = G.name("dq_f"), G.name("dq_c")
            new = [oh.make_node("Cast", [n.input[0]], [a], to=TP.FLOAT, name=G.name("dq_cast")),
                   oh.make_node("Sub", [a, cz], [b], name=G.name("dq_sub")),
                   oh.make_node("Mul", [b, cs], [n.output[0]], name=G.name("dq_mul"))]
            i = list(G.g.node).index(n); G.g.node.remove(n)
            for k, nd in enumerate(new): G.g.node.insert(i + k, nd)
            n_act += 1
    return {"const_dq_folded": n_const, "activation_dq_replaced": n_act}


def _tables_from_analysis(ws, analysis_json):
    import json, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import hwlib
    rows = json.load(open(analysis_json))
    t = {"gelu": [], "sigmoid": [], "softmax": []}
    for r in rows:
        if r["type"] == "Gelu": t["gelu"].append({"node": r["node"], "in_q": r["in_q"], "out_q": r["out_q"], "lut": hwlib.activ_lut(ws, r["node"], "half_away").tolist()})
        elif r["type"] == "Sigmoid": t["sigmoid"].append({"node": r["node"], "in_q": r["in_q"], "out_q": r["out_q"], "lut": hwlib.activ_lut(ws, r["node"], "half_away").tolist()})
    return t


DEFAULT_RULES = ("gelu", "softmax", "matmul_wzp", "sigmoid", "no_qdq_fusion")


def make_boardlike(src_onnx_path, dst_onnx_path, rules=DEFAULT_RULES, tables=None, verbose=False, gelu_impl="float"):
    """Rewrite src -> dst (see module doc). rules: subset of gelu / softmax / matmul_wzp / sigmoid / no_qdq_fusion.
    tables: optional dict from compiled_tables() (exact compiled model only). Returns a report dict."""
    m = src_onnx_path if isinstance(src_onnx_path, onnx.ModelProto) else onnx.load(str(src_onnx_path))
    m = copy.deepcopy(m) if isinstance(src_onnx_path, onnx.ModelProto) else m
    try:
        mi = onnx.shape_inference.infer_shapes(m)
        shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}
    except Exception:
        shapes = {}
    G = _G(m)
    rep = {"version": VERSION, "src": "ModelProto" if isinstance(src_onnx_path, onnx.ModelProto) else str(src_onnx_path),
           "dst": str(dst_onnx_path), "rules": list(rules), "gelu_impl": gelu_impl}
    new = []
    if "matmul_wzp" in rules: rep["matmul_wzp"] = _rule_matmul_wzp(G); G.refresh()
    if "gelu" in rules:
        n, r = _rule_gelu(G, tables, gelu_impl); new += n; rep["gelu"] = r; G.refresh()
    if "sigmoid" in rules:
        n, r = _rule_sigmoid(G, tables); new += n; rep["sigmoid"] = r; G.refresh()
    if "softmax" in rules:
        n, r = _rule_softmax(G, tables, shapes); new += n; rep["softmax"] = r; G.refresh()
    _toposort_insert(G, new)
    rep["removed_nodes"] = _dce(G)
    if "no_qdq_fusion" in rules:
        G.refresh(); rep["no_qdq_fusion"] = _rule_no_qdq_fusion(G); _dce(G)
    # A2 check (reported, not enforced): zp of the A operands of activation x activation MatMuls in the SOURCE model
    rep["a2_zero_points"] = [r["A_zp"] for r in rep.get("matmul_wzp", [])]
    rep["a2_ok"] = len(rep["a2_zero_points"]) == 4 and all(z == 0 for z in rep["a2_zero_points"])
    del m.graph.value_info[:]
    m.producer_name = "W1 boardlike"; m.doc_string = VERSION
    onnx.checker.check_model(m)
    onnx.save(m, str(dst_onnx_path))
    if verbose:
        print(f"make_boardlike: {sum('skipped' not in r for r in rep.get('gelu', []))} GELU, {sum('skipped' not in r for r in rep.get('sigmoid', []))} Sigmoid, "
              f"{sum('skipped' not in r for r in rep.get('softmax', []))} Softmax, "
              f"{sum(r['rewritten'] for r in rep.get('matmul_wzp', []))}/{len(rep.get('matmul_wzp', []))} MatMul A-zp rewritten, "
              f"{rep['removed_nodes']} nodes removed -> {dst_onnx_path}")
    return rep


def compiled_tables(ws, analysis_json, qjson_path):
    """Optional: ACTIV tables of a compiled model (inferred evaluation model), keyed in graph order, for make_boardlike(tables=...).
    analysis_json = output of analyze_compiled.py for the same compile; qjson_path = its OE Q.json."""
    import json, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import hwlib
    t = _tables_from_analysis(ws, analysis_json)
    T, q = hwlib.qjson(qjson_path)
    for blk, gm in (("742", "Gemm_739_out_0"), ("811", "Gemm_808_out_0")):
        t["softmax"].append({"s_in": q(gm)[0], "exp_lut": hwlib.activ_lut(ws, f"Softmax_{blk}_exp_expansion", "half_away").tolist(),
                             "inv_lut": hwlib.activ_lut(ws, f"Softmax_{blk}_inverse_expansion", "half_away").tolist()})
    return t


if __name__ == "__main__":
    import sys, json
    r = make_boardlike(sys.argv[1], sys.argv[2], verbose=True)
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 6 else f"{len(v)} items") for k, v in r.items()}, indent=1))
