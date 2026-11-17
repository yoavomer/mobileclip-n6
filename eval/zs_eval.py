"""W0 zero-shot ImageNet evaluator for the MobileCLIP-S0 image encoder ONNX models (ImageNetV2 matched-frequency).

Python:
    import sys; sys.path.insert(0, os.environ.get("MCLIP_ROOT", ".") + r"\\night\\W0_data"); import zs_eval
    r = zs_eval.evaluate("quant/layers/mci0_256_p9995.onnx", split="selection", mode="plain")   # -> dict
    d = zs_eval.paired_diff(r["npz"])                 # model vs FP32 (paired bootstrap)
    d = zs_eval.paired_diff(npz_a, npz_b)             # model A vs model B
CLI:
    python zs_eval.py --model M.onnx [--name NAME] [--split selection] [--mode plain|boardlike] [--text templates80|simple]
                      [--n N] [--threads 4] [--out results.json] [--no-cache]
    python zs_eval.py --paired A.npz [B.npz]

What it does: ORT CPU (default graph optimizations = ORT_ENABLE_ALL, intra_op_num_threads=threads, batch 1),
input float32 = u8/255 [1,3,256,256] (bit-exact equal to the MobileCLIP-S0 transform, see build_data_info.json),
embedding L2-normalized, logits = E @ text.T with text = text_imagenet.npy (80 OpenAI templates, default) or
text_imagenet_simple.npy ("a photo of a {name}.").
All accuracies in the returned dict are FRACTIONS (0..1). CIs are 95 %.
split="test" is refused unless the environment variable NIGHT_TEST_UNLOCK=orchestrator is set.
mode="boardlike": the model is first converted with night/W1_emulator/boardlike.py make_boardlike(src, dst) into a
temporary ONNX under night/W0_data/tmp/ (deleted afterwards) and that model is evaluated."""
import os, sys, json, math, time, hashlib, pathlib, argparse, glob
import numpy as np

R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
W = R / "night" / "W0_data"
RES = W / "results"
TMP = W / "tmp"
FP32 = R / "mci0_256_fp32.onnx"
SPLITS = ("selection", "calib_pool", "test")
TEXTS = {"templates80": W / "text_imagenet.npy", "simple": W / "text_imagenet_simple.npy"}
ATT = [f"/model/network.7/network.7.{b}/token_mixer/{s}" for b in "01" for s in ("Softmax_output_0", "Mul_output_0")]
Z = 1.959963984540054


# ---------------- data ----------------
def _check_split(split):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "test" and os.environ.get("NIGHT_TEST_UNLOCK") != "orchestrator":
        raise PermissionError("TEST split is LOCKED (see night/W0_data/TEST_LOCKED.txt). Only the orchestrator may "
                              "evaluate it, once, by setting NIGHT_TEST_UNLOCK=orchestrator.")


def load_split(split, n=None):
    """-> (u8 array memmap [N,3,256,256], labels int64 [N], file ids list[str]); first n entries if n is given."""
    _check_split(split)
    U = np.load(W / f"{split}_u8.npy", mmap_mode="r")
    y = np.load(W / f"{split}_labels.npy")
    ids = [l.split("\t")[0] for l in (W / f"{split}_files.txt").read_text().splitlines() if l.strip()]
    assert len(U) == len(y) == len(ids)
    if n is not None:
        n = int(n); assert 0 < n <= len(U), n
        U, y, ids = U[:n], y[:n], ids[:n]
    return U, y, ids


def load_text(text="templates80"):
    return np.load(TEXTS[text]).astype(np.float32)


# ---------------- helpers ----------------
def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _model_bytes(model):
    """path (str/Path, absolute or relative to the project root), raw bytes or onnx.ModelProto -> (bytes, label, path or None)."""
    if isinstance(model, (bytes, bytearray)):
        return bytes(model), "bytes", None
    if isinstance(model, (str, pathlib.Path)):
        p = pathlib.Path(model)
        if not p.is_absolute() and not p.exists():
            p = R / p
        return p.read_bytes(), p.stem, p
    if hasattr(model, "SerializeToString"):
        return model.SerializeToString(), "modelproto", None
    raise TypeError("model must be an ONNX path or onnx.ModelProto")


def wilson(k, n, z=Z):
    """Wilson score interval for k successes out of n -> [lo, hi] (fractions)."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def check_a2(model):
    """A2 rule: zero point of the 4 QuantizeLinear nodes whose input is one of ATT. -> dict"""
    import onnx
    from onnx import numpy_helper
    b, _, _ = _model_bytes(model)
    m = onnx.load_from_string(b)
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    for nd in m.graph.node:   # Constant-node zero points
        if nd.op_type == "Constant" and nd.output:
            for a in nd.attribute:
                if a.name == "value":
                    init[nd.output[0]] = numpy_helper.to_array(a.t)
    found = {}
    for nd in m.graph.node:
        if nd.op_type == "QuantizeLinear" and nd.input and nd.input[0] in ATT:
            zp = init.get(nd.input[2]) if len(nd.input) > 2 else np.array(0)
            sc = init.get(nd.input[1])
            found.setdefault(nd.input[0], []).append({"zp": None if zp is None else np.asarray(zp).ravel().tolist(),
                                                      "zp_dtype": None if zp is None else str(np.asarray(zp).dtype),
                                                      "scale": None if sc is None else np.asarray(sc).ravel().tolist()})
    if not found:
        return {"applicable": False, "note": "no QuantizeLinear on the attention operands (float model?)"}
    ok = all(t in found for t in ATT) and all(
        q["zp"] is not None and all(v == 0 for v in q["zp"]) for t in ATT for q in found.get(t, []))
    return {"applicable": True, "ok": bool(ok), "operands": found}


def _session(model_bytes, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads); so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # graph_optimization_level left at its default (ORT_ENABLE_ALL), like every earlier PC evaluation
    return ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])


def embed(model_bytes, U, threads=4, log_every=500, tag=""):
    """-> L2-normalized float32 embeddings [N,512]; input = u8/255 float32, batch 1."""
    s = _session(model_bytes, threads)
    iname = s.get_inputs()[0].name
    outs = [o.name for o in s.get_outputs()]
    oname = "embedding" if "embedding" in outs else outs[0]
    E = np.empty((len(U), 512), np.float32); t0 = time.time()
    for i in range(len(U)):
        x = U[i:i + 1].astype(np.float32) / np.float32(255)
        E[i] = s.run([oname], {iname: x})[0].reshape(-1)
        if log_every and (i + 1) % log_every == 0:
            print(f"  [{tag}] {i + 1}/{len(U)} {time.time() - t0:.0f}s", flush=True)
    E64 = E.astype(np.float64)
    return (E64 / np.linalg.norm(E64, axis=1, keepdims=True)).astype(np.float32), time.time() - t0


def _topk(E, T, k=5):
    L = E.astype(np.float64) @ T.astype(np.float64).T
    idx = np.argpartition(-L, k, axis=1)[:, :k]
    o = np.argsort(-np.take_along_axis(L, idx, 1), axis=1)
    return np.take_along_axis(idx, o, 1)


def fp32_reference(split, n=None, threads=4, cache=True):
    """FP32 embeddings of the whole split (cached in results/fp32ref_<split>.npz), sliced to n."""
    _check_split(split)
    U, y, ids = load_split(split)
    fb = FP32.read_bytes(); sha = _sha256_bytes(fb)
    f = RES / f"fp32ref_{split}.npz"
    if cache and f.exists():
        z = np.load(f, allow_pickle=False)
        if str(z["model_sha256"]) == sha and len(z["E"]) == len(U):
            E = z["E"]
            return (E[:n] if n else E), sha
    E, sec = embed(fb, U, threads, tag="fp32 ref")
    RES.mkdir(parents=True, exist_ok=True)
    np.savez(f, E=E, labels=y, ids=np.array(ids), model_sha256=np.array(sha), seconds=np.array(sec))
    return (E[:n] if n else E), sha


def _boardlike_fn():
    p = R / "night" / "W1_emulator" / "boardlike.py"
    if not p.exists():
        raise FileNotFoundError(f"mode='boardlike' needs {p} (workstream W1, make_boardlike(src_onnx_path, dst_onnx_path)); "
                                "it does not exist yet -> board-like numbers are pending.")
    # Execute exactly the bytes that are hashed (W1 may edit boardlike.py while we run; a cached `import` would then
    # run old code under a new hash). One fresh module per file version.
    src = p.read_bytes(); sha = _sha256_bytes(src)
    if _BL.get("sha") != sha:
        import types
        if str(p.parent) not in sys.path:
            sys.path.insert(0, str(p.parent))      # in case boardlike.py imports sibling W1 modules
        mod = types.ModuleType(f"w1_boardlike_{sha[:12]}"); mod.__file__ = str(p)
        sys.modules[mod.__name__] = mod
        exec(compile(src, str(p), "exec"), mod.__dict__)
        _BL.clear(); _BL.update(sha=sha, fn=mod.make_boardlike, version=getattr(mod, "VERSION", None))
    return _BL["fn"], _BL["sha"]


_BL = {}


def _find_cached(sha, mode, split, n):
    pat = str(RES / f"*_{sha[:12]}_{mode}_{split}_n{n}_*.npz")
    for f in sorted(glob.glob(pat)):
        try:
            z = np.load(f, allow_pickle=False)
            if str(z["model_sha256"]) == sha:
                return z["E"]
        except Exception:
            pass
    return None


# ---------------- main API ----------------
def evaluate(model, split="selection", mode="plain", n=None, threads=4, cache=True, text="templates80", name=None):
    """Zero-shot ImageNet evaluation of an image-encoder ONNX (path or ModelProto). Returns a dict (fractions)."""
    _check_split(split)
    if mode not in ("plain", "boardlike"):
        raise ValueError("mode must be 'plain' or 'boardlike'")
    if text not in TEXTS:
        raise ValueError(f"text must be one of {list(TEXTS)}")
    t_start = time.time()
    U, y, ids = load_split(split, n)
    N = len(y)
    src_bytes, label, src_path = _model_bytes(model)
    name = name or label
    src_sha = _sha256_bytes(src_bytes)
    a2 = check_a2(src_bytes)
    info = {"name": name, "model": str(src_path) if src_path else "<ModelProto>", "model_sha256": src_sha,
            "mode": mode, "split": split, "n": N, "text": text, "threads": threads, "a2_check_src": a2}
    run_bytes = src_bytes
    if mode == "boardlike":
        make_boardlike, bl_sha = _boardlike_fn()
        TMP.mkdir(parents=True, exist_ok=True)
        tag = f"bl_{src_sha[:12]}_{os.getpid()}"
        srcp = src_path
        if srcp is None:
            srcp = TMP / f"{tag}_src.onnx"; srcp.write_bytes(src_bytes)
        dst = TMP / f"{tag}.onnx"
        try:
            rep = make_boardlike(str(srcp), str(dst))
            run_bytes = dst.read_bytes()
            if isinstance(rep, dict):   # compact summary of W1's report
                s = {k: rep[k] for k in ("version", "rules", "a2_ok", "a2_zero_points", "removed_nodes") if k in rep}
                for k in ("gelu", "sigmoid", "softmax", "matmul_wzp"):
                    if isinstance(rep.get(k), list):
                        s[f"n_{k}"] = len(rep[k])
                        s[f"n_{k}_skipped"] = sum(1 for r in rep[k] if isinstance(r, dict) and "skipped" in r)
                info["boardlike_report"] = json.loads(json.dumps(s, default=str))
        finally:
            for f in glob.glob(str(TMP / f"{tag}*")):
                try:
                    os.remove(f)
                except OSError:
                    pass
        info["boardlike_py_sha256"] = bl_sha
        info["boardlike_model_sha256"] = _sha256_bytes(run_bytes)
    run_sha = _sha256_bytes(run_bytes)
    import onnxruntime as ort
    info["ort_version"] = ort.__version__

    # FP32 reference (same images)
    F, fsha = fp32_reference(split, n, threads, cache)
    T = load_text(text)
    E = None
    if cache:
        if run_sha == fsha:
            E = F
        else:
            E = _find_cached(run_sha, mode, split, N)
    sec = 0.0
    if E is None:
        E, sec = embed(run_bytes, U, threads, tag=name)
        info["ms_per_image"] = round(1000 * sec / N, 2)
    else:
        info["embeddings_from_cache"] = True

    P = _topk(E, T); PF = _topk(F, T)
    c1 = P[:, 0] == y; c5 = (P == y[:, None]).any(1)
    f1 = PF[:, 0] == y; f5 = (PF == y[:, None]).any(1)
    cos = (E.astype(np.float64) * F.astype(np.float64)).sum(1)
    k1, k5, kf1, kf5 = int(c1.sum()), int(c5.sum()), int(f1.sum()), int(f5.sum())
    info.update({
        "top1": k1 / N, "top1_ci95": wilson(k1, N), "top5": k5 / N, "top5_ci95": wilson(k5, N),
        "fp32_top1": kf1 / N, "fp32_top1_ci95": wilson(kf1, N), "fp32_top5": kf5 / N, "fp32_top5_ci95": wilson(kf5, N),
        "share_of_fp32_top1": (k1 / kf1) if kf1 else float("nan"),
        "share_of_fp32_top5": (k5 / kf5) if kf5 else float("nan"),
        "agreement_top1_with_fp32": float((P[:, 0] == PF[:, 0]).mean()),
        "cos_vs_fp32_mean": float(cos.mean()), "cos_vs_fp32_median": float(np.median(cos)), "cos_vs_fp32_min": float(cos.min()),
    })
    RES.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    npz = RES / f"{safe}_{run_sha[:12]}_{mode}_{split}_n{N}_{text}.npz"
    np.savez(npz, E=E, pred_top5=P.astype(np.int16), labels=y, ids=np.array(ids), correct1=c1, correct5=c5,
             fp32_pred_top5=PF.astype(np.int16), fp32_correct1=f1, fp32_correct5=f5, cos_vs_fp32=cos.astype(np.float32),
             model_sha256=np.array(run_sha), src_sha256=np.array(src_sha), mode=np.array(mode), split=np.array(split),
             text=np.array(text), name=np.array(name))
    info["npz"] = str(npz)
    if run_sha != fsha:
        d = paired_diff(npz)
        info["diff_top1_vs_fp32"] = d["diff_top1"]; info["diff_top1_vs_fp32_ci95"] = d["diff_top1_ci95"]
        info["share_of_fp32_top1_ci95"] = d["ratio_top1_ci95"]
        info["discordant_vs_fp32"] = {"model_only_correct": d["a_only"], "fp32_only_correct": d["b_only"],
                                      "sign_test_p": d["sign_test_p"]}
    info["seconds_total"] = round(time.time() - t_start, 1)
    return info


def _sign_test_p(a_only, b_only):
    """exact two-sided binomial (McNemar exact) p-value on the discordant pairs."""
    m = a_only + b_only
    if m == 0:
        return 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return min(1.0, 2 * p)


def paired_diff(a_npz, b_npz=None, B=10000, seed=0, metric="correct1", allow_text_mismatch=False):
    """Paired bootstrap 95 % CI of mean(a) - mean(b) for per-image correctness (default top-1).
    b_npz=None -> b = the FP32 reference stored inside a_npz (same images, same text). Images must be identical.
    allow_text_mismatch=True compares runs made with different text embeddings (e.g. the template effect)."""
    a = np.load(a_npz, allow_pickle=False)
    xa = a[metric].astype(np.float64)
    if b_npz is None:
        xb = a["fp32_" + metric].astype(np.float64); bname = "fp32"
    else:
        b = np.load(b_npz, allow_pickle=False)
        if not np.array_equal(a["ids"], b["ids"]):
            raise ValueError("the two npz files are not on the same images (ids differ)")
        if str(a["text"]) != str(b["text"]) and not allow_text_mismatch:
            raise ValueError("different text embeddings in the two npz files")
        xb = b[metric].astype(np.float64); bname = str(b["name"])
    n = len(xa); rng = np.random.default_rng(seed)
    diffs = np.empty(B); ratios = np.empty(B)
    for s in range(0, B, 500):
        idx = rng.integers(0, n, size=(min(500, B - s), n))
        ma = xa[idx].mean(1); mb = xb[idx].mean(1)
        diffs[s:s + len(idx)] = ma - mb
        ratios[s:s + len(idx)] = np.where(mb > 0, ma / np.maximum(mb, 1e-12), np.nan)
    a_only = int(((xa == 1) & (xb == 0)).sum()); b_only = int(((xa == 0) & (xb == 1)).sum())
    key = "top1" if metric == "correct1" else "top5"
    return {"a": str(a["name"]), "b": bname, "metric": metric, "n": n, "bootstrap_B": B, "seed": seed,
            f"a_{key}": float(xa.mean()), f"b_{key}": float(xb.mean()),
            f"diff_{key}": float(xa.mean() - xb.mean()),
            f"diff_{key}_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            f"ratio_{key}": float(xa.mean() / xb.mean()) if xb.mean() > 0 else float("nan"),
            f"ratio_{key}_ci95": [float(np.nanpercentile(ratios, 2.5)), float(np.nanpercentile(ratios, 97.5))],
            "a_only": a_only, "b_only": b_only, "sign_test_p": _sign_test_p(a_only, b_only)}


def _fmt(r):
    s = (f"{r['name']:<22s} {r['mode']:<9s} {r['split']} n={r['n']} text={r['text']}: top1 {100 * r['top1']:.2f}% "
         f"[{100 * r['top1_ci95'][0]:.2f},{100 * r['top1_ci95'][1]:.2f}]  top5 {100 * r['top5']:.2f}% "
         f"[{100 * r['top5_ci95'][0]:.2f},{100 * r['top5_ci95'][1]:.2f}]  share {100 * r['share_of_fp32_top1']:.1f}%  "
         f"agree {100 * r['agreement_top1_with_fp32']:.1f}%  cos {r['cos_vs_fp32_mean']:.4f}")
    if "diff_top1_vs_fp32" in r:
        s += (f"  d(top1) {100 * r['diff_top1_vs_fp32']:+.2f} pp [{100 * r['diff_top1_vs_fp32_ci95'][0]:+.2f},"
              f"{100 * r['diff_top1_vs_fp32_ci95'][1]:+.2f}]")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model"); ap.add_argument("--name")
    ap.add_argument("--split", default="selection"); ap.add_argument("--mode", default="plain")
    ap.add_argument("--text", default="templates80"); ap.add_argument("--n", type=int)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--out")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--paired", nargs="+", metavar="NPZ", help="A.npz [B.npz]: paired bootstrap A-B (B default FP32)")
    a = ap.parse_args(argv)
    if a.paired:
        d = paired_diff(a.paired[0], a.paired[1] if len(a.paired) > 1 else None)
        print(json.dumps(d, indent=1)); return d
    if not a.model:
        ap.error("--model is required (or --paired)")
    r = evaluate(a.model, split=a.split, mode=a.mode, n=a.n, threads=a.threads, cache=not a.no_cache, text=a.text,
                 name=a.name)
    print(_fmt(r), flush=True)
    if a.out:
        out = pathlib.Path(a.out)
        allr = json.loads(out.read_text()) if out.exists() else {}
        allr[f"{r['name']}|{r['split']}|{r['mode']}|{r['text']}|n{r['n']}"] = r
        out.write_text(json.dumps(allr, indent=1))
    return r


if __name__ == "__main__":
    main()
