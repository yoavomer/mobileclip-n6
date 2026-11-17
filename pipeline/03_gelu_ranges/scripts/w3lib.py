"""W3 library: graph edits on the QDQ P99.95-family models + board-like proxy evaluation.

Edits (on the QDQ source model; board-like conversion with W1's make_boardlike happens AFTER the edit):
  bypass(m, tensors)    remove the activation Q->DQ pair(s) of each tensor (consumers read the float tensor).
                        * RELU_FUSED (3 SE Relus that ORT folded into the Q clamp): a Relu is re-inserted.
                        * GELU chains: if any tensor of a chain (input, Erf/Add/Mul inner, Mul_1 output) is bypassed,
                          the chain's INNER QDQs (Erf/Add/Mul outputs) are removed too. On the board a GELU is ONE unit
                          (int8 in at the chain-input Q, exact GELU, int8 out at the chain-output Q), and W1's R1 rule
                          removes the inner QDQs anyway when both ends are quantized; without this a partially bypassed
                          chain would keep inner roundings that the board does not have.
                        * the 4 A2 attention operands are never bypassed (assert).
  set_q(m, t, s, zp)    give tensor t's QuantizeLinear (and its DequantizeLinear(s)) new scalar int8 params.
Spec strings (parts joined by '+'):  base | byp=T1|T2 | bypblock=B1|B2 | q=T@scale@zp | rng=T@lo@hi
Proxy: board-like (W1 v4) embeddings on a fixed image set vs FP32: cosine mean/min, top-1 agreement, top-1 vs labels.
"""
import os, re, sys, copy, json, time, types, hashlib, pathlib, collections
import numpy as np, onnx
from onnx import numpy_helper as nh, helper as oh

R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
W3 = R / "night" / "W3_beyond"
TMP = W3 / "tmp"
BASE = R / "quant" / "layers" / "mci0_256_p9995.onnx"
FP32 = R / "mci0_256_fp32.onnx"
W0 = R / "night" / "W0_data"
ATT = [f"/model/network.7/network.7.{b}/token_mixer/{s}" for b in "01" for s in ("Softmax_output_0", "Mul_output_0")]
RELU_FUSED = {"/model/network.3/proj/proj.0/se/act/Relu_output_0", "/model/network.5/proj/proj.0/se/act/Relu_output_0",
              "/model/conv_exp/se/Relu_output_0"}
SQRT2 = 2 ** 0.5


def safe(name):
    """file-name-safe version of a job name (Windows: ':' would create an NTFS alternate data stream)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def sha256_bytes(b): return hashlib.sha256(b).hexdigest()


def sha256_file(p): return sha256_bytes(pathlib.Path(p).read_bytes())


def load(path=BASE): return onnx.load(str(path))


# ------------------------------------------------------------------------------------------------ graph inspection
class Info:
    def __init__(self, m):
        g = m.graph
        self.ini = {t.name: t for t in g.initializer}
        self.prod = {o: n for n in g.node for o in n.output}
        self.cons = collections.defaultdict(list)
        for n in g.node:
            for i in n.input: self.cons[i].append(n)
        self.actQ = {n.input[0]: n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] not in self.ini}

    def qparams(self, t):
        q = self.actQ[t]
        return float(nh.to_array(self.ini[q.input[1]])), int(nh.to_array(self.ini[q.input[2]]))


def block_of(t):
    s = re.search(r"/model/(patch_embed|conv_exp|head)", t)
    if s: return s.group(1)
    s = re.search(r"(network\.\d+\.\d+)", t) or re.search(r"(network\.\d+)", t)
    if s: return s.group(1)
    if t == "input": return "input"
    if t == "embedding_QuantizeLinear_Input": return "head"
    if t in ("gemm_input_reshape_arg", "gemm_output_reshape_arg"): return "network.7.0"
    if t in ("gemm_input_reshape_arg_token_1", "gemm_output_reshape_arg_token_4"): return "network.7.1"
    return "?"


def blocks(m):
    I = Info(m); d = collections.OrderedDict()
    for t in I.actQ:
        d.setdefault(block_of(t), []).append(t)
    return d


def gelu_chains(m):
    """-> list of dict(input=chain input tensor, inner=[Erf/Add/Mul outputs], output=Mul_1 output tensor) on the QDQ graph."""
    I = Info(m); out = []
    for erf in [n for n in m.graph.node if n.op_type == "Erf"]:
        div = I.prod[erf.input[0]]
        dq = I.prod.get(div.input[0])
        x = dq.input[0] if dq is not None and dq.op_type == "DequantizeLinear" else div.input[0]
        if x in I.prod and I.prod[x].op_type == "QuantizeLinear": x = I.prod[x].input[0]
        base = erf.output[0].rsplit("/", 1)[0]
        inner = [t for t in (base + "/Erf_output_0", base + "/Add_output_0", base + "/Mul_output_0") if t in I.actQ]
        o = base + "/Mul_1_output_0"
        out.append(dict(input=x, inner=inner, output=o if o in I.actQ else None, base=base))
    return out


# ------------------------------------------------------------------------------------------------------ edits
def bypass(m, tensors):
    m = copy.deepcopy(m); g = m.graph
    tensors = list(dict.fromkeys(tensors))
    for t in tensors: assert t not in ATT, f"A2 operand {t} must stay quantized"
    chains = gelu_chains(m); ts = set(tensors)
    for c in chains:
        if ts & ({c["input"], c["output"]} | set(c["inner"])):
            for t in c["inner"]:
                if t not in ts: tensors.append(t); ts.add(t)
    I = Info(m)
    drop, add = [], []
    for t in tensors:
        q = I.actQ[t]
        p = I.prod.get(t)
        if t in RELU_FUSED and p is not None and p.op_type != "Relu":
            k = list(p.output).index(t); p.output[k] = t + "__prerelu"
            add.append((p, oh.make_node("Relu", [t + "__prerelu"], [t], name=t + "__relu")))
        for dq in I.cons[q.output[0]]:
            assert dq.op_type == "DequantizeLinear"
            for c in I.cons[dq.output[0]]:
                for k, i in enumerate(c.input):
                    if i == dq.output[0]: c.input[k] = t
            for o in g.output:
                if o.name == dq.output[0]: o.name = t
            drop.append(dq)
        drop.append(q)
    ids = {id(n) for n in drop}; after = {id(p): r for p, r in add}
    keep = []
    for n in g.node:
        if id(n) in ids: continue
        keep.append(n)
        if id(n) in after: keep.append(after[id(n)])
    del g.node[:]; g.node.extend(keep)
    return m


def set_q(m, t, scale, zp, copy_model=True):
    """new scalar int8 (scale, zp) for activation tensor t: new initializers for its Q and all its DQs."""
    if copy_model: m = copy.deepcopy(m)
    I = Info(m); q = I.actQ[t]
    zp = int(np.clip(zp, -128, 127))
    tag = re.sub(r"[^A-Za-z0-9]", "_", t)[-60:]
    sn, zn = f"w3_s_{tag}_{len(m.graph.initializer)}", f"w3_z_{tag}_{len(m.graph.initializer)}"
    m.graph.initializer.extend([nh.from_array(np.array(scale, np.float32), sn), nh.from_array(np.array(zp, np.int8), zn)])
    q.input[1] = sn; q.input[2] = zn
    for dq in I.cons[q.output[0]]:
        dq.input[1] = sn; dq.input[2] = zn
    return m


def range_to_q(lo, hi):
    """int8 asymmetric (s, zp) covering [lo, hi] (lo<=0<=hi enforced like ORT), zp rounded; ORT convention."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    s = (hi - lo) / 255.0
    if s == 0: s = 1.0
    zp = int(np.clip(np.round(-128 - lo / s), -128, 127))
    return np.float32(s), zp


def q_to_range(s, zp): return ((-128 - zp) * s, (127 - zp) * s)


def apply_spec(m, spec):
    blk = None
    for part in spec.split("+"):
        part = part.strip()
        if not part or part == "base": continue
        kind, arg = part.split("=", 1)
        if kind == "byp": m = bypass(m, arg.split("|"))
        elif kind == "bypblock":
            blk = blk or blocks(m)
            ts = [t for b in arg.split("|") for t in blk[b] if t not in ATT]
            m = bypass(m, ts)
        elif kind == "fpw":      # weights of the given blocks (or 'all') -> original FP32 (biases stay int32), diagnostic only
            m = float_weights(m, arg.split("|"))
        elif kind == "q":
            for item in arg.split("|"):
                t, s, z = item.split("@"); m = set_q(m, t, float(s), int(z))
        elif kind == "rng":
            for item in arg.split("|"):
                t, lo, hi = item.split("@"); s, z = range_to_q(float(lo), float(hi)); m = set_q(m, t, s, z)
        else: raise ValueError(part)
    if "q=" in spec or "rng=" in spec:
        fix_bias(m)
    return m


def a2_check(m):
    I = Info(m); r = {}
    for t in ATT:
        if t not in I.actQ: r[t] = None; continue
        r[t] = I.qparams(t)
    ok = all(v is not None and v[1] == 0 for v in r.values())
    return ok, r


# -------------------------------------------------------------------------------------------------- board-like
_BL = {}


def boardlike_fn():
    p = R / "night" / "W1_emulator" / "boardlike.py"
    src = p.read_bytes(); sha = sha256_bytes(src)
    if _BL.get("sha") != sha:
        if str(p.parent) not in sys.path: sys.path.insert(0, str(p.parent))
        mod = types.ModuleType(f"w1_boardlike_{sha[:12]}"); mod.__file__ = str(p)
        sys.modules[mod.__name__] = mod
        exec(compile(src, str(p), "exec"), mod.__dict__)
        _BL.clear(); _BL.update(sha=sha, fn=mod.make_boardlike, version=getattr(mod, "VERSION", None))
    return _BL["fn"], _BL["sha"], _BL["version"]


def to_boardlike_bytes(m, tag="x"):
    fn, sha, ver = boardlike_fn()
    TMP.mkdir(parents=True, exist_ok=True)
    dst = TMP / f"bl_{safe(tag)}_{os.getpid()}_{time.time_ns()}.onnx"
    try:
        rep = fn(copy.deepcopy(m), str(dst))
        b = dst.read_bytes()
    finally:
        try: os.remove(dst)
        except OSError: pass
    return b, rep


# ------------------------------------------------------------------------------------------------------ eval
def session(b, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = int(threads); so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; so.log_severity_level = 3
    return ort.InferenceSession(b, so, providers=["CPUExecutionProvider"])


def embed(b, U, threads=1):
    s = session(b, threads)
    outs = [o.name for o in s.get_outputs()]
    on = "embedding" if "embedding" in outs else outs[0]
    E = np.empty((len(U), 512), np.float32)
    for i in range(len(U)):
        E[i] = s.run([on], {"input": U[i:i + 1].astype(np.float32) / np.float32(255)})[0].reshape(-1)
    E = E.astype(np.float64); E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E.astype(np.float32)


def proxy_set(name="calibB"):
    """fixed screening set. calibB = calib_pool rows 500..999 (500 images, 500 classes); calibration statistics use rows 0..499."""
    if name == "calibB":
        U = np.load(W0 / "calib_pool_u8.npy", mmap_mode="r")[500:1000]
        y = np.load(W0 / "calib_pool_labels.npy")[500:1000]
    else: raise ValueError(name)
    return np.ascontiguousarray(U), y


def fp32_ref(name="calibB", threads=3):
    f = W3 / "results" / f"fp32ref_{name}.npz"
    sha = sha256_file(FP32)
    if f.exists():
        z = np.load(f)
        if str(z["sha"]) == sha: return z["E"]
    U, y = proxy_set(name)
    E = embed(FP32.read_bytes(), U, threads)
    np.savez(f, E=E, sha=np.array(sha), labels=y)
    return E


_T = None


def text():
    global _T
    if _T is None: _T = np.load(W0 / "text_imagenet.npy").astype(np.float64)
    return _T


def metrics(E, F, y):
    T = text()
    P = (E.astype(np.float64) @ T.T).argmax(1); PF = (F.astype(np.float64) @ T.T).argmax(1)
    c = (E.astype(np.float64) * F.astype(np.float64)).sum(1)
    return {"cos": float(c.mean()), "cos_min": float(c.min()), "agree": float((P == PF).mean()),
            "top1": float((P == y).mean()), "fp32_top1": float((PF == y).mean()), "n": int(len(y))}


def prune(m):
    """drop initializers no node uses (after edits that create new ones)."""
    used = {i for n in m.graph.node for i in n.input}
    keep = [t for t in m.graph.initializer if t.name in used]
    del m.graph.initializer[:]; m.graph.initializer.extend(keep)
    return m


MOVE_OPS = ("Reshape", "Transpose", "Split", "Squeeze", "Unsqueeze", "Flatten", "AveragePool", "Concat")


def followers(m):
    """leader tensor -> list of follower tensors: outputs of data-movement ops (Reshape/Transpose/Split/Squeeze/AveragePool..)
    whose Q params equal those of their (quantized) input. On the board such ops need no requantization; to keep that,
    a range change of a leader is applied to all its followers (transitively)."""
    I = Info(m); par = {}
    for t, q in I.actQ.items():
        p = I.prod.get(t)
        if p is None or p.op_type not in MOVE_OPS: continue
        src = p.input[0]
        dq = I.prod.get(src)
        if dq is None or dq.op_type != "DequantizeLinear": continue
        qn = I.prod.get(dq.input[0])
        if qn is None or qn.op_type != "QuantizeLinear": continue
        s = qn.input[0]
        if s in I.actQ and I.qparams(s) == I.qparams(t): par[t] = s
    root = lambda t: root(par[t]) if t in par else t
    out = collections.defaultdict(list)
    for t in par: out[root(t)].append(t)
    return dict(out)


def tclass(t, m=None, _c={}):
    """board-relevant tensor class (W3 grouping)."""
    if m is not None and "chains" not in _c:
        ch = gelu_chains(m); _c["chains"] = ch
        _c["gin"] = {c["input"] for c in ch}; _c["gout"] = {c["output"] for c in ch}; _c["inner"] = {x for c in ch for x in c["inner"]}
    if t in ATT: return "A2"
    if t in _c.get("inner", ()): return "GELU_inner"
    if t in _c.get("gin", ()): return "GELU_in"
    if t in _c.get("gout", ()): return "GELU_out"
    if t == "input": return "input"
    if "/se/" in t:
        if t.endswith(("fc2/Conv_output_0", "expand/Conv_output_0")): return "SE_sig_in"
        if "Sigmoid" in t or t.endswith("se/Reshape_output_0"): return "SE_sig_out"
        return "SE_other"
    if re.search(r"network\.\d+\.\d+/Mul(_1)?_output_0$", t): return "layer_scale"
    if re.search(r"network\.\d+\.\d+/Add(_1)?_output_0$", t): return "residual"
    if "token_mixer/reparam_conv" in t: return "token_mixer"
    if "convffn/conv/conv" in t: return "ffn_dw"
    if "convffn/fc2" in t: return "ffn_fc2"
    if "lkb_reparam" in t: return "lkb"
    if t.startswith("/model/network.7") or t.startswith("gemm_"): return "attention"
    if t.startswith("/model/head") or t == "embedding_QuantizeLinear_Input": return "head"
    if t.startswith("/model/conv_exp"): return "conv_exp"
    if "network.6" in t: return "rep_cpe"
    return "other"


def fix_bias(m, tol=1e-6):
    """keep bias_scale == s_in * s_w for every Conv/Gemm with an int32 bias DQ (the convention of the quantizer, which the
    compiler may rely on): if a range change touched a conv INPUT, its bias is re-expressed in the new bias scale
    (float bias kept up to 0.5 new LSB). Returns the number of biases changed. Edits m in place."""
    I = Info(m); k = 0
    for c in m.graph.node:
        if c.op_type not in ("Conv", "Gemm") or len(c.input) < 3: continue
        bdq, wdq, xdq = I.prod.get(c.input[2]), I.prod.get(c.input[1]), I.prod.get(c.input[0])
        if not all(p is not None and p.op_type == "DequantizeLinear" for p in (bdq, wdq, xdq)): continue
        if bdq.input[0] not in I.ini or wdq.input[1] not in I.ini or xdq.input[1] not in I.ini: continue
        bs = nh.to_array(I.ini[bdq.input[1]]).astype(np.float64); ws = nh.to_array(I.ini[wdq.input[1]]).astype(np.float64)
        xs = float(nh.to_array(I.ini[xdq.input[1]]))
        want = (np.float32(xs) * ws.astype(np.float32)).astype(np.float32)
        if np.abs(bs / want.astype(np.float64) - 1).max() <= tol: continue
        bq = nh.to_array(I.ini[bdq.input[0]]).astype(np.float64)
        nb = np.clip(np.round(bq * bs / want.astype(np.float64)), -2 ** 31, 2 ** 31 - 1).astype(np.int32)
        tag = f"w3b_{len(m.graph.initializer)}"
        m.graph.initializer.extend([nh.from_array(nb, bdq.input[0] + tag), nh.from_array(want.reshape(bs.shape), bdq.input[1] + tag)])
        bdq.input[0] = bdq.input[0] + tag; bdq.input[1] = bdq.input[1] + tag
        k += 1
    return k


def weight_dqs(m):
    """[(weight initializer name, its DQ node, consumer node)] for every int8 weight DequantizeLinear (not biases)."""
    I = Info(m); out = []
    for n in m.graph.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in I.ini:
            a = nh.to_array(I.ini[n.input[0]])
            if a.dtype == np.int8 and a.ndim >= 2:
                c = I.cons[n.output[0]]
                out.append((n.input[0], n, c[0] if c else None))
    return out


def float_weights(m, blks):
    sys.path.insert(0, str(R / "quant" / "layers")); import sens
    m = copy.deepcopy(m); I = Info(m); drop = []
    for w, dq, c in weight_dqs(m):
        if c is None: continue
        b = block_of(c.name if c.name.startswith("/model") else (c.output[0]))
        if "all" not in blks and b not in blks: continue
        shape = nh.to_array(I.ini[w]).shape
        nm = w + "__fp32"; m.graph.initializer.append(nh.from_array(sens.fp32_init(w, shape), nm))
        for cc in I.cons[dq.output[0]]:
            for k, i in enumerate(cc.input):
                if i == dq.output[0]: cc.input[k] = nm
            # the int32 bias of this conv/gemm -> its original FP32 value too (an int32 bias may hold a bias correction
            # for the int8 weights, which would be wrong with FP32 weights)
            if cc.op_type in ("Conv", "Gemm") and len(cc.input) > 2:
                bdq = I.prod.get(cc.input[2])
                if bdq is not None and bdq.op_type == "DequantizeLinear" and bdq.input[0] in I.ini:
                    bn = bdq.input[0] + "__fp32"
                    m.graph.initializer.append(nh.from_array(sens.fp32_init(bdq.input[0], nh.to_array(I.ini[bdq.input[0]]).shape), bn))
                    cc.input[2] = bn; drop.append(bdq)
        drop.append(dq)
    ids = {id(n) for n in drop}; keep = [n for n in m.graph.node if id(n) not in ids]
    del m.graph.node[:]; m.graph.node.extend(keep)
    return m
