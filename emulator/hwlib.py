"""W1 helper: read NPU unit configs from a stedgeai workspace trace (network_blob_trace.c) and evaluate them in numpy.
Re-implements (does not modify) the compiler analyst's models in bisect/compiler/emulate_activ.py / emulate_eltwise.py:
  ACTIV_FUNC table model (INFERRED by the analyst, validated vs ideal GELU/Sigmoid):
    u = q + 128; outer = u >> (8 - bwidth); ROM0 entry e: start = e & 0x3F, k = e >> 6 (2^k inner rows);
    inner = (u >> (8 - bwidth - k)) & (2^k - 1); row = start + inner;
    y = (A*q*q + (B*q << shift_b) + (C << shift_c)) >> shift_o (round half up), saturated to int8.
  ARITH model (ll_aton.h field docs): AFFINE r = (A*x >> Ax) + (B*y >> By) + (C << Cs) (or + BC when combinebc);
    MUL r = x*y; out = round_half_up(r / 2^shift_o), saturated to outbytes_o.
Only reads files. [from code]"""
import os, re, sys, json, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bisect", "compiler"))
from trace_parse import parse, ival, params_off  # noqa: E402  (read-only import of the analyst's parser)


@functools.lru_cache(maxsize=8)
def load_ws(ws):
    """ws = <gws>/neural_art__network directory (or the gws dir itself)."""
    p = os.path.join(ws, "network_blob_trace.c")
    if not os.path.exists(p): p = os.path.join(ws, "neural_art__network", "network_blob_trace.c")
    return parse(p)


def find(ws, node, struct):
    eps, _ = load_ws(ws)
    for e, lst in eps.items():
        for c in lst:
            if c["node"] == node and c["struct"] == struct: return e, c
    raise KeyError((node, struct))


def unit(ws, node):
    e, c = find(ws, node, "LL_Arithacc_InitTypeDef"); return e, c["fields"]


def _poff(v):
    m = re.search(r"\+\s*(\d+)\)\)\)", v); return int(m.group(1))


def activ_tables(ws, node):
    eps, params = load_ws(ws)
    e, c = find(ws, node, "LL_Activacc_InitTypeDef"); f = c["fields"]
    R0 = np.frombuffer(params, np.uint8, ival(f, "ROM0_nbytes"), _poff(f["ROM0_vector"][0]))
    R1 = np.frombuffer(params, "<i2", ival(f, "ROM1_nbytes") // 2, _poff(f["ROM1_vector"][0])).reshape(-1, 3).astype(np.int64)
    return e, f, R0, R1


def _rshift_round(r, so, rmode):
    if rmode == "half_up": return (r + (1 << (so - 1))) >> so
    # half away from zero (ties of negative values go toward -inf): board-observed on the softmax inverse table + pool
    a = (np.abs(r) + (1 << (so - 1))) >> so
    return np.where(np.asarray(r) < 0, -a, a)


def activ_eval(f, R0, R1, rmode="half_up"):
    """256-entry table: index q+128 -> int8 output (analyst's inferred model, 'offset' segment mapping)."""
    bw = ival(f, "bwidth"); sb, sc, so = ival(f, "shift_b"), ival(f, "shift_c"), ival(f, "shift_o")
    out = np.zeros(256, np.int64)
    for qq in range(-128, 128):
        u = qq + 128
        e = int(R0[u >> (8 - bw)]); k = e >> 6; start = e & 0x3F
        inner = (u >> (8 - bw - k)) & ((1 << k) - 1) if k else 0
        A, B, C = R1[start + inner]
        r = A * qq * qq + ((B * qq) << sb) + (C << sc)
        out[qq + 128] = max(-128, min(127, int(_rshift_round(r, so, rmode))))
    return out


def activ_lut(ws, node, rmode="half_up"):
    e, f, R0, R1 = activ_tables(ws, node); return activ_eval(f, R0, R1, rmode)


def _vec(params, f, k, n, idx):
    o = params_off(f, k)
    if o is None: return None
    bits = int(f["vec_precision"][0].strip("{}").split(",")[idx])
    return np.frombuffer(params, dtype={8: "<i1", 16: "<i2", 32: "<i4"}[bits], count=n, offset=o).astype(np.int64)


def arith(ws, f, x, y=None, ch=None, nch=1, rmode="half_up"):
    _, params = load_ws(ws)
    op = f["operation"][0]
    sel = (lambda v: v[ch]) if ch is not None else (lambda v: v[0])
    A = _vec(params, f, "A_vector", nch, 0); B = _vec(params, f, "B_vector", nch, 1); C = _vec(params, f, "C_vector", nch, 2)
    A = sel(A) if A is not None else ival(f, "A_scalar")
    ax, by, cs = ival(f, "Ax_shift", 0), ival(f, "By_shift", 0), ival(f, "C_shift", 0)
    if op == "ARITH_MUL":
        r = x * y
    else:
        r = (A * x) >> ax
        if ival(f, "combinebc", 0):
            BC = sel(C) if C is not None else ((ival(f, "B_scalar") << 16) | (ival(f, "C_scalar") & 0xFFFF))
            r = r + BC
        else:
            Bv = sel(B) if B is not None else ival(f, "B_scalar")
            Cv = sel(C) if C is not None else ival(f, "C_scalar")
            if ival(f, "dualinput", 0) and y is not None: r = r + ((Bv * y) >> by)
            r = r + (Cv << cs)
    sh = ival(f, "shift_o", 0)
    if sh: r = _rshift_round(r, sh, rmode)
    lim = 127 if ival(f, "outbytes_o") == 1 else 32767
    return np.clip(r, -lim - 1, lim)


def softmax_consts(ws, blk):
    """All constants of one expanded softmax (blk = '742' / '811') as plain ints, for comparison across compiles."""
    out = {}
    for nm in ("sub1", "mul", "mul_sub2", "sub2"):
        node = {"sub1": f"Softmax_{blk}_sub1_expansion", "mul": f"Softmax_{blk}_mul_expansion",
                "mul_sub2": f"Softmax_{blk}_mul_expansion_mul_sub2_", "sub2": f"Softmax_{blk}_sub2_expansion"}[nm]
        _, f = unit(ws, node)
        out[nm] = {k: ival(f, k) for k in ("operation", "A_scalar", "B_scalar", "C_scalar", "Ax_shift", "By_shift", "C_shift",
                                           "shift_o", "combinebc", "dualinput", "outbytes_o", "inbytes_x", "inbytes_y")}
    eps, _ = load_ws(ws)
    for nm in ("global_maxpool", "red_avg"):
        node = f"Softmax_{blk}_{nm}_expansion"
        for e, lst in eps.items():
            for c in lst:
                if c["node"] == node and "Pool" in c["struct"]:
                    out[nm] = {k: v[0] for k, v in c["fields"].items() if "addr" not in k and "vector" not in k}
    out["exp_lut"] = activ_lut(ws, f"Softmax_{blk}_exp_expansion").tolist()
    out["inv_lut"] = activ_lut(ws, f"Softmax_{blk}_inverse_expansion").tolist()
    return out


def pool_round(v, mul, sh): return (v * mul + (1 << (sh - 1))) >> sh


def hw_softmax(ws, logits, blk, rmode="half_up"):
    """Compiler analyst's softmax-expansion model (bisect/compiler/emulate_softmax.py) on int8 logits [..., keys].
    rmode='half_away': ties rounded away from zero in every unit (W1 finding, see NOTES.md S3)."""
    x = logits.astype(np.int64)
    mx = x.max(-1, keepdims=True)
    _, fsub = unit(ws, f"Softmax_{blk}_sub1_expansion")
    d = arith(ws, fsub, x, np.broadcast_to(mx, x.shape), rmode=rmode)
    e = activ_lut(ws, f"Softmax_{blk}_exp_expansion", rmode)[d + 128]
    nk = x.shape[-1]
    assert nk == 64, nk
    avg = np.clip(_rshift_round(e.sum(-1, keepdims=True) * 16384, 20, rmode), -128, 127)
    inv = activ_lut(ws, f"Softmax_{blk}_inverse_expansion", rmode)[avg + 128]
    _, fm = unit(ws, f"Softmax_{blk}_mul_expansion"); _, fm2 = unit(ws, f"Softmax_{blk}_mul_expansion_mul_sub2_")
    _, fs2 = unit(ws, f"Softmax_{blk}_sub2_expansion")
    invb = np.broadcast_to(inv, e.shape)
    u = arith(ws, fm, e, invb, rmode=rmode); u = arith(ws, fm2, u, invb, rmode=rmode); out = arith(ws, fs2, u, invb, rmode=rmode)
    return out, dict(d=d, e=e, avg=avg, inv=inv)


def qjson(path):
    T = json.load(open(path))["tensors"]
    def q(t):
        p = T[t]["format"]["params"]; return (None if p[0] is None else float(p[0][0])), (None if p[1] is None else int(p[1][0]))
    return T, q
