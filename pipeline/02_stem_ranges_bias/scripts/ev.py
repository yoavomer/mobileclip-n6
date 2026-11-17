"""W2 wrapper around W0's zs_eval: same evaluator, but every output (npz, temporary board-like ONNX) goes into
night/W2_stem/results and night/W2_stem/tmp (W2 may only write inside its own folder). ORT threads = 3.
CLI: python -I ev.py <model.onnx> <name> <split> <mode> [n]   -> prints one JSON line, appends to results/all_results.jsonl
"""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, shutil, pathlib, time
R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
W2 = R / "night" / "W2_stem"
sys.path.insert(0, str(R / "night" / "W0_data"))
import zs_eval  # noqa: E402

zs_eval.RES = W2 / "results"
zs_eval.TMP = W2 / "tmp"
zs_eval.RES.mkdir(parents=True, exist_ok=True); zs_eval.TMP.mkdir(parents=True, exist_ok=True)
_src = R / "night" / "W0_data" / "results" / "fp32ref_selection.npz"
_dst = zs_eval.RES / "fp32ref_selection.npz"
if _src.exists() and not _dst.exists():
    shutil.copyfile(_src, _dst)   # read-only use of W0's cached FP32 embeddings (same FP32 SHA is checked by zs_eval)

THREADS = 3
P9995 = str(R / "quant" / "layers" / "mci0_256_p9995.onnx")
KEYS = ("name", "mode", "split", "n", "top1", "top1_ci95", "top5", "top5_ci95", "fp32_top1", "share_of_fp32_top1",
        "share_of_fp32_top1_ci95", "agreement_top1_with_fp32", "cos_vs_fp32_mean", "cos_vs_fp32_median",
        "cos_vs_fp32_min", "diff_top1_vs_fp32", "diff_top1_vs_fp32_ci95", "ms_per_image", "model_sha256", "npz",
        "boardlike_py_sha256", "seconds_total")


def run(model, name, split="selection", mode="plain", n=None, log=True):
    r = zs_eval.evaluate(str(model), split=split, mode=mode, n=n, threads=THREADS, name=name)
    a2 = r.get("a2_check_src", {})
    out = {k: r.get(k) for k in KEYS}
    out["a2_ok"] = a2.get("ok"); out["model"] = str(model)
    if mode == "boardlike":
        out["boardlike_version"] = r.get("boardlike_report", {}).get("version")
    out["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if log:
        with open(W2 / "results" / "all_results.jsonl", "a") as f:
            f.write(json.dumps(out) + "\n")
    return out


def paired(a_npz, b_npz=None, metric="correct1"):
    return zs_eval.paired_diff(a_npz, b_npz, metric=metric)


if __name__ == "__main__":
    m, name, split, mode = sys.argv[1:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else None
    print(json.dumps(run(m, name, split, mode, n)), flush=True)
