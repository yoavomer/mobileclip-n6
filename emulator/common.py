"""shared helpers for the W1 validation scripts"""
import pathlib, numpy as np, onnxruntime as ort
R = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
N_IN = 3 * 256 * 256
FIXED_Q = (((37 * np.arange(N_IN) + 11) % 256) - 128).astype(np.int8)


def sess(path_or_bytes, opt=True, threads=3):
    """opt: True = ORT_ENABLE_ALL (default), False = ORT_DISABLE_ALL, "ext" = ORT_ENABLE_EXTENDED (no NCHWc)"""
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    if opt is False: so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    elif opt == "ext": so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    p = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes)
    return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])


def stats(a, b, z, mask=None):
    """a = emulator/reference, b = board; cosine on q - zp, max |diff| LSB, % exact."""
    a = np.asarray(a).astype(np.int64).ravel(); b = np.asarray(b).astype(np.int64).ravel()
    if mask is not None: a, b = a[mask], b[mask]
    d = np.abs(a - b); x, y = a - z, b - z
    return {"cos": float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))), "maxdiff": int(d.max()),
            "exact_pct": float((d == 0).mean() * 100), "within1_pct": float((d <= 1).mean() * 100), "n": int(a.size)}


def fmt(s): return f'{s["cos"]:.6f} / {s["maxdiff"]} / {s["exact_pct"]:.2f}%'


def q8(y, s, z): return np.clip(np.rint(np.asarray(y, np.float64) / s) + z, -128, 127).astype(np.int64)


def crc32(a):
    import zlib; return zlib.crc32(np.asarray(a, np.int8).tobytes()) & 0xFFFFFFFF
