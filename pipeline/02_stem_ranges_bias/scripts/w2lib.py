"""W2 helpers: GELU-chain discovery, per-tensor QDQ edits, A2 check, int8 range helpers."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, copy, math, pathlib, hashlib
import numpy as np
import onnx
from onnx import numpy_helper as nh, helper as oh

R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
W2 = R / "night" / "W2_stem"
P9995 = R / "quant" / "layers" / "mci0_256_p9995.onnx"
FP32_PRE = R / "mci0_256_fp32_pre.onnx"
FP32 = R / "mci0_256_fp32.onnx"
sys.path.insert(0, str(R / "night" / "W1_emulator"))
import boardlike as BL  # noqa: E402

STEM = [f"/model/patch_embed/patch_embed.{k}/reparam_conv/Conv_output_0" for k in range(3)]
ATT = [f"/model/network.7/network.7.{b}/token_mixer/{s}" for b in "01" for s in ("Softmax_output_0", "Mul_output_0")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p=P9995):
    return onnx.load(str(p))


def maps(m):
    g = m.graph
    init = {t.name: t for t in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    return init, prod, cons


def gelu_chains(m):
    """-> list of dict(x=float chain-input tensor (conv out / SE Mul out), q=its QuantizeLinear node,
    s_in, z_in, s_out, z_out (chain output = Mul_1 Q), q_out node name, out tensor) in graph order."""
    G = BL._G(copy.deepcopy(m))
    res = []
    init, prod, cons = maps(m)
    for f in BL._find_gelus(G):
        qnode = prod[f["x_q"]]
        x = qnode.input[0]
        s_in, z_in = G.qparams(f["dq_in"]); s_o, z_o = G.qparams(f["q_out"])
        res.append(dict(x=x, q_name=qnode.name, s_in=float(np.ravel(s_in)[0]), z_in=int(np.ravel(z_in)[0]),
                        s_out=float(np.ravel(s_o)[0]), z_out=int(np.ravel(z_o)[0]), q_out=f["q_out"].name,
                        out=f["q_out"].input[0], erf=f["erf"].name, x_producer=prod[x].op_type if x in prod else None))
    return res


def get_qparams(m, tensor):
    init, prod, cons = maps(m)
    qs = [n for n in cons.get(tensor, []) if n.op_type == "QuantizeLinear"]
    assert len(qs) == 1, (tensor, len(qs))
    q = qs[0]
    return float(nh.to_array(init[q.input[1]]).ravel()[0]), int(nh.to_array(init[q.input[2]]).ravel()[0])


def set_qparams(m, tensor, scale, zp, tag="w2"):
    """In place: give the (single) int8 QuantizeLinear of `tensor` and all DequantizeLinear nodes of its output new
    scale/zp initializers (new unique names, so nothing shared is touched). Returns (old_scale, old_zp)."""
    init, prod, cons = maps(m)
    qs = [n for n in cons.get(tensor, []) if n.op_type == "QuantizeLinear"]
    assert len(qs) == 1, (tensor, len(qs))
    q = qs[0]
    zt = nh.to_array(init[q.input[2]])
    assert zt.dtype == np.int8, zt.dtype
    old = (float(nh.to_array(init[q.input[1]]).ravel()[0]), int(zt.ravel()[0]))
    base = tensor.strip("/").replace("/", "_")
    sn, zn = f"{base}_{tag}_scale", f"{base}_{tag}_zp"
    k = 0
    while sn in init or zn in init:
        k += 1; sn, zn = f"{base}_{tag}{k}_scale", f"{base}_{tag}{k}_zp"
    m.graph.initializer.append(nh.from_array(np.array(scale, np.float32), sn))
    m.graph.initializer.append(nh.from_array(np.array(zp, np.int8), zn))
    dqs = [n for n in cons.get(q.output[0], []) if n.op_type == "DequantizeLinear"]
    assert len(dqs) >= 1
    for n in [q] + dqs:
        n.input[1] = sn; n.input[2] = zn
    # drop now-unused initializers (only if no one else uses them)
    used = {i for n in m.graph.node for i in n.input}
    keep = [t for t in m.graph.initializer if t.name in used]
    del m.graph.initializer[:]; m.graph.initializer.extend(keep)
    return old


def range_to_qp(lo, hi):
    """asymmetric int8 range [lo, hi] (lo<=0<=hi) -> (scale, zp) as ORT computes it (qmin=-128, qmax=127)."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    s = (hi - lo) / 255.0
    z = int(round(-128 - lo / s))
    z = max(-128, min(127, z))
    return s, z


def qp_to_range(s, z):
    return (-128 - z) * s, (127 - z) * s


def a2_check(m_or_path):
    """zero points of the 4 attention-operand QuantizeLinear nodes; -> (ok, list)"""
    m = m_or_path if isinstance(m_or_path, onnx.ModelProto) else onnx.load(str(m_or_path))
    init, prod, cons = maps(m)
    zps = []
    for t in ATT:
        qs = [n for n in cons.get(t, []) if n.op_type == "QuantizeLinear"]
        for q in qs:
            zps.append(int(nh.to_array(init[q.input[2]]).ravel()[0]))
        if not qs:
            zps.append(None)
    return (len(zps) == 4 and all(z == 0 for z in zps)), zps


def gelu(x):
    from math import sqrt
    import numpy as _np
    try:
        from scipy.special import erf  # not installed; fallback below
    except Exception:
        erf = None
    if erf is None:
        return 0.5 * x * (1.0 + _erf_np(x / sqrt(2.0)))
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def _erf_np(x):
    """vectorised erf, |error| < 1.2e-7 (Numerical Recipes erfc Chebyshev), float64"""
    x = np.asarray(x, np.float64)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (
        0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0, 1.0 - r, r - 1.0)


def fq(x, s, z):
    """fake-quant int8 with round-half-to-even (ORT QuantizeLinear)"""
    q = np.clip(np.rint(x / s) + z, -128, 127)
    return (q - z) * s


def save(m, path):
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return sha256(path)
