"""W3 wrapper around W0's zs_eval (same evaluator; npz + temporary board-like ONNX go into night/W3_beyond/results|tmp).
ORT threads = 3. Reads W0's cached FP32 SELECTION embeddings (copied, SHA-checked by zs_eval).
CLI: python -I ev.py <model.onnx> <name> <mode plain|boardlike> [n]  -> one JSON line, appended to results/selection_results.jsonl
Also prints paired diffs vs P99.95 (W0 npz of the same mode) and, if given via env W3_REF_<MODE>=npz, vs that reference."""
import sys, os, json, shutil, pathlib, time
R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
W3 = R / "night" / "W3_beyond"
sys.path.insert(0, str(R / "night" / "W0_data"))
import zs_eval  # noqa: E402

zs_eval.RES = W3 / "results"; zs_eval.TMP = W3 / "tmp"
zs_eval.RES.mkdir(parents=True, exist_ok=True); zs_eval.TMP.mkdir(parents=True, exist_ok=True)
_src = R / "night" / "W0_data" / "results" / "fp32ref_selection.npz"; _dst = zs_eval.RES / "fp32ref_selection.npz"
if _src.exists() and not _dst.exists(): shutil.copyfile(_src, _dst)
P9995_NPZ = {"plain": R / "night" / "W0_data" / "results" / "p9995_ba18908b45e8_plain_selection_n3000_templates80.npz",
             "boardlike": R / "night" / "W0_data" / "results" / "p9995_bl_d948cf85eeb1_boardlike_selection_n3000_templates80.npz"}
KEYS = ("name", "mode", "split", "n", "top1", "top1_ci95", "top5", "top5_ci95", "fp32_top1", "share_of_fp32_top1",
        "share_of_fp32_top1_ci95", "agreement_top1_with_fp32", "cos_vs_fp32_mean", "cos_vs_fp32_min", "diff_top1_vs_fp32",
        "diff_top1_vs_fp32_ci95", "ms_per_image", "model_sha256", "npz", "boardlike_py_sha256", "seconds_total")


def run(model, name, mode, n=None, refs=None):
    r = zs_eval.evaluate(str(model), split="selection", mode=mode, n=n, threads=3, name=name)
    out = {k: r.get(k) for k in KEYS}
    out["a2_ok"] = r.get("a2_check_src", {}).get("ok"); out["model"] = str(model)
    out["a2_zero_points"] = {k[-40:]: [q["zp"] for q in v] for k, v in r.get("a2_check_src", {}).get("operands", {}).items()}
    if mode == "boardlike": out["boardlike_version"] = r.get("boardlike_report", {}).get("version")
    if n is None or n == 3000:
        refs = dict(refs or {}); refs.setdefault("p9995", str(P9995_NPZ[mode]))
        for k, v in refs.items():
            if v and pathlib.Path(v).exists():
                for met in ("correct1", "correct5"):
                    d = zs_eval.paired_diff(r["npz"], v, metric=met)
                    key = "top1" if met == "correct1" else "top5"
                    out[f"paired_{key}_vs_{k}"] = [d[f"diff_{key}"], d[f"diff_{key}_ci95"], d["a_only"], d["b_only"], d["sign_test_p"]]
    out["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(W3 / "results" / "selection_results.jsonl", "a") as f: f.write(json.dumps(out) + "\n")
    return out


if __name__ == "__main__":
    m, name, mode = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else None
    refs = {k[7:].lower(): v for k, v in os.environ.items() if k.startswith("W3_REF_") and k.endswith("_" + mode.upper())}
    refs = {k.rsplit("_", 1)[0]: v for k, v in refs.items()}
    print(json.dumps(run(m, name, mode, n, refs)), flush=True)
