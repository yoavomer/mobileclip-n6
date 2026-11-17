"""W5 shared PC helpers: data lists, preprocessing pipelines, ORT sessions (plain / board-like), cached embeddings.
All ORT sessions: CPU, ORT_ENABLE_ALL (default, like zs_eval), intra_op 2 threads, inter_op 1."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, hashlib, pathlib, time
import numpy as np

R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
H = R / "night" / "W5_demo"
W0 = R / "night" / "W0_data"
IMG_ROOT = R / "data" / "imagenetv2_mf" / "imagenetv2-matched-frequency-format-val"
RES = H / "results"; TMP = H / "tmp"
for p in (str(H / "pc"), str(W0), str(R / "night" / "W1_emulator")):
    if p not in sys.path:
        sys.path.insert(0, p)
import omv_sim  # noqa: E402

MODELS = {
    "fp32": R / "mci0_256_fp32.onnx",
    "a2": R / "baseline" / "mci0_256_a2.onnx",
    "p9995": R / "quant" / "layers" / "mci0_256_p9995.onnx",
}
THREADS = 2


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_list(name):
    """name in calib_pool / unused / selection (NEVER test) -> list of (relpath, label)."""
    assert name in ("calib_pool", "unused", "selection"), name
    out = []
    for l in (W0 / f"{name}_files.txt").read_text().splitlines():
        if l.strip():
            a, b = l.split("\t"); out.append((a, int(b)))
    return out


def load_photo(rel):
    from PIL import Image
    return Image.open(IMG_ROOT / rel).convert("RGB")


# ---------------- pipelines: photo -> uint8 planes [3,256,256] (model float input = u8/255) ----------------
def pipe_ref(img):
    return omv_sim.reference_u8(img)


def pipe_565only(img):
    """reference geometry, then RGB888 -> RGB565 (truncation) -> 8 bit by OpenMV bit replication."""
    u = omv_sim.reference_u8(img).transpose(1, 2, 0)
    return omv_sim.rgb565_to_rgb888(omv_sim.rgb888_to_rgb565(u)).transpose(2, 0, 1).copy()


def make_pipe_cam(framesize, rounding="trunc"):
    def f(img):
        fr = omv_sim.simulate_frame(img, framesize, rounding)
        q, planes, info = omv_sim.board_input(fr, 1 / 255, -128)
        return planes
    f.__name__ = f"cam_{framesize}_{rounding}"
    return f


def pipe_qvga_pil(img):
    """decomposition: the same 240x240 scene square as cam_QVGA, 8-bit (no RGB565), upsampled 240 -> 256 by PIL bilinear."""
    from PIL import Image
    iw, ih = img.size; s = min(iw, ih)
    sq = img.crop(((iw - s) // 2, (ih - s) // 2, (iw - s) // 2 + s, (ih - s) // 2 + s)).resize((240, 240), Image.BILINEAR)
    return np.asarray(sq.resize((256, 256), Image.BILINEAR), dtype=np.uint8).transpose(2, 0, 1).copy()


PIPES = {"ref": pipe_ref, "565only": pipe_565only, "cam_QVGA_pil": pipe_qvga_pil,
         "cam_QVGA": make_pipe_cam("QVGA"), "cam_HVGA": make_pipe_cam("HVGA"), "cam_VGA": make_pipe_cam("VGA"),
         "cam_QVGA_round": make_pipe_cam("QVGA", "round")}


def planes_cached(pipe, items, tag):
    """-> uint8 [N,3,256,256] for the (relpath, label) items, cached in tmp/planes_<pipe>_<tag>.npy"""
    f = TMP / f"planes_{pipe}_{tag}.npy"
    if f.exists():
        U = np.load(f)
        if len(U) == len(items):
            return U
    U = np.empty((len(items), 3, 256, 256), np.uint8)
    fn = PIPES[pipe]
    for i, (rel, _) in enumerate(items):
        U[i] = fn(load_photo(rel))
    np.save(f, U)
    return U


# ---------------- models ----------------
_SESS = {}


def model_path(name):
    """name: fp32 | a2 | p9995 | a2_bl | p9995_bl | <path to onnx> [+ '_bl']"""
    if name in MODELS:
        return MODELS[name], False
    if name.endswith("_bl"):
        base = name[:-3]
        return (MODELS[base] if base in MODELS else pathlib.Path(base)), True
    return pathlib.Path(name), False


def session(name):
    if name in _SESS:
        return _SESS[name]
    import onnxruntime as ort
    src, bl = model_path(name)
    info = {"name": name, "src": str(src), "src_sha256": sha256(src), "boardlike": bl}
    path = src
    if bl:
        from boardlike import make_boardlike
        TMP.mkdir(exist_ok=True)
        dst = TMP / f"bl_{pathlib.Path(src).stem}_{info['src_sha256'][:8]}.onnx"
        rep = make_boardlike(str(src), str(dst))
        info.update(boardlike_version=rep.get("version"), a2_ok_boardlike=rep.get("a2_ok"),
                    boardlike_py_sha256=sha256(R / "night" / "W1_emulator" / "boardlike.py"))
        path = dst
    so = ort.SessionOptions(); so.intra_op_num_threads = THREADS; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    _SESS[name] = (s, info)
    return _SESS[name]


def a2_check(name):
    import zs_eval
    src, _ = model_path(name)
    if name.startswith("fp32"):
        return {"applicable": False}
    r = zs_eval.check_a2(str(src))
    return {"ok": r.get("ok"), "zps": {"/".join(k.split("/")[-3:]): [q["zp"] for q in v]
                                       for k, v in r.get("operands", {}).items()}}


def embed_u8(name, U, cache_tag=None):
    """L2-normalized embeddings [N,512] float32 of u8 planes (float input = u8/255); cached by (model sha, tag)."""
    s, info = session(name)
    f = None
    if cache_tag:
        import re
        f = TMP / f"emb_{re.sub(r'[^A-Za-z0-9_.-]', '_', name)[-40:]}_{info['src_sha256'][:8]}_{cache_tag}.npy"
        if f.exists():
            E = np.load(f)
            if len(E) == len(U):
                return E
    iname = s.get_inputs()[0].name
    outs = [o.name for o in s.get_outputs()]
    oname = "embedding" if "embedding" in outs else outs[0]
    E = np.empty((len(U), 512), np.float32)
    for i in range(len(U)):
        x = U[i:i + 1].astype(np.float32) / np.float32(255)
        E[i] = s.run([oname], {iname: x})[0].reshape(-1)
    E64 = E.astype(np.float64)
    E = (E64 / np.linalg.norm(E64, axis=1, keepdims=True)).astype(np.float32)
    if f is not None:
        np.save(f, E)
    return E


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0.0, c - h)), 4), round(float(min(1.0, c + h)), 4)]


def input_q_params(path):
    """(scale, zp) of the QuantizeLinear that consumes the graph input; output DQ (scale, zp)."""
    import onnx
    from onnx import numpy_helper as nh
    m = onnx.load(str(path), load_external_data=False)
    ini = {t.name: nh.to_array(t) for t in m.graph.initializer}
    q = next(n for n in m.graph.node if n.op_type == "QuantizeLinear" and n.input[0] == m.graph.input[0].name)
    dq = next(n for n in m.graph.node if m.graph.output[0].name in n.output)
    return (float(ini[q.input[1]]), int(ini[q.input[2]])), (float(ini[dq.input[1]]), int(ini[dq.input[2]])), dq.op_type
