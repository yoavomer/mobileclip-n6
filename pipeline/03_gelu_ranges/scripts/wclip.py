"""Per-channel weight range (clipping) search for int8 conv weights (post-training, int8, zp 0 kept).
The P99.95 weights are per-output-channel symmetric max-abs (codes -127..127, zp 0) [from code]. Here, per output channel c,
alpha_c in a grid (1.0 .. 0.5) scales the clip: s_c = alpha_c * max|w_c| / 127, q = clip(round(w / s_c), -127, 127),
chosen to minimise sum_j m2_j * (q s - w)_j^2 with m2_j = E[x_j^2] of the conv input channel j (from the FP32 input; 1 if
not given) summed over the kernel taps (diagonal approximation of the output MSE).
The int32 bias is re-expressed in the new bias scale (s_in * s_c): new = round(old_int * old_scale / new_scale), so the
float bias (incl. any earlier bias correction) is kept up to rounding.
API: clip_conv(m, conv_node_name, m2=None, alphas=...) -> report dict; edits m in place."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import numpy as np, onnx, pathlib, sys
from onnx import numpy_helper as nh
sys.path.insert(0, os.environ.get("MCLIP_ROOT", ".") + r"\quant\layers")
import sens  # read-only: fp32_init (original FP32 weights, with the qkv transposition)

ALPHAS = np.round(np.concatenate([np.arange(1.0, 0.70, -0.01), np.arange(0.70, 0.49, -0.02)]), 3)


def _maps(m):
    ini = {t.name: t for t in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}
    return ini, prod


def clip_conv(m, conv_name, m2=None, alphas=ALPHAS, only_if_better=True):
    ini, prod = _maps(m)
    conv = next(n for n in m.graph.node if n.name == conv_name)
    wdq = prod[conv.input[1]]; bdq = prod[conv.input[2]] if len(conv.input) > 2 else None
    wq = nh.to_array(ini[wdq.input[0]]); ws = nh.to_array(ini[wdq.input[1]]).astype(np.float64)
    wz = nh.to_array(ini[wdq.input[2]])
    assert wq.dtype == np.int8 and (wz == 0).all() and ws.ndim == 1 and ws.size == wq.shape[0]
    W = sens.fp32_init(wdq.input[0], wq.shape).astype(np.float64)
    C = wq.shape[0]; Wf = W.reshape(C, wq.shape[1], -1)
    wt = np.ones(wq.shape[1]) if m2 is None else np.asarray(m2, np.float64)
    if wt.size != wq.shape[1]: wt = np.ones(wq.shape[1])
    amax = np.abs(Wf).reshape(C, -1).max(1)
    new_s = ws.copy(); new_q = wq.copy().reshape(C, -1); e_old = np.zeros(C); e_new = np.zeros(C); a_best = np.ones(C)
    for c in range(C):
        w = Wf[c]
        def err(s):
            q = np.clip(np.round(w / s), -127, 127); return float((wt[:, None] * (q * s - w) ** 2).sum()), q
        e0, _ = err(ws[c]); e_old[c] = e0; best = (e0, ws[c], wq.reshape(C, -1)[c], 1.0)
        if amax[c] == 0: e_new[c] = e0; continue
        for a in alphas:
            s = a * amax[c] / 127.0; e, q = err(s)
            if e < best[0] * (1 - 1e-9): best = (e, s, q.reshape(-1).astype(np.int8), a)
        e_new[c], new_s[c], new_q[c], a_best[c] = best[0], best[1], best[2], best[3]
    # write weights + scale (new initializer names)
    wn, sn = wdq.input[0] + "__w3clip", wdq.input[1] + "__w3clip"
    m.graph.initializer.extend([nh.from_array(new_q.reshape(wq.shape).astype(np.int8), wn), nh.from_array(new_s.astype(np.float32), sn)])
    wdq.input[0] = wn; wdq.input[1] = sn
    rep = {"conv": conv_name, "C": int(C), "n_clipped": int((a_best < 1).sum()), "alpha_min": float(a_best.min()),
           "alpha_mean": float(a_best.mean()), "err_old": float(e_old.sum()), "err_new": float(e_new.sum())}
    if bdq is not None:
        bq = nh.to_array(ini[bdq.input[0]]).astype(np.float64); bs = nh.to_array(ini[bdq.input[1]]).astype(np.float64)
        s_in = bs / ws                                    # bias scale = s_in * s_w  (per channel, s_in constant)
        nbs = s_in * new_s.astype(np.float32).astype(np.float64)
        nb = np.clip(np.round(bq * bs / nbs), -2 ** 31, 2 ** 31 - 1).astype(np.int32)
        bn, bsn = bdq.input[0] + "__w3clip", bdq.input[1] + "__w3clip"
        m.graph.initializer.extend([nh.from_array(nb, bn), nh.from_array(nbs.astype(np.float32), bsn)])
        bdq.input[0] = bn; bdq.input[1] = bsn
        rep["s_in_spread"] = float(s_in.max() / s_in.min() - 1)
    return rep
