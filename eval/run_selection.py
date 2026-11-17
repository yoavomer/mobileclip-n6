"""W0 step 6: SELECTION (plain) for FP32 + the 4 quantized models, paired comparisons, optional board-like.
Writes results/results_selection.json (zs_eval CLI format) and results/paired_selection.json.
usage: python -I run_selection.py [plain|boardlike|paired|torchcheck] ..."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, pathlib
import numpy as np

W = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"\night\W0_data"); R = W.parents[1]
sys.path.insert(0, str(W))
import zs_eval  # noqa: E402

MODELS = {"fp32": "mci0_256_fp32.onnx", "a2": r"baseline\mci0_256_a2.onnx", "p9995": r"quant\layers\mci0_256_p9995.onnx",
          "B1": r"quant\int16\B1.onnx", "B2": r"quant\int16\B2.onnx"}
OUT = W / "results" / "results_selection.json"
PAIR = W / "results" / "paired_selection.json"


def save(r):
    allr = json.loads(OUT.read_text()) if OUT.exists() else {}
    allr[f"{r['name']}|{r['split']}|{r['mode']}|{r['text']}|n{r['n']}"] = r
    OUT.write_text(json.dumps(allr, indent=1))


def run(mode):
    for name, p in MODELS.items():
        if mode == "boardlike" and name == "fp32":
            continue
        r = zs_eval.evaluate(R / p, split="selection", mode=mode, name=name if mode == "plain" else f"{name}_bl")
        print(zs_eval._fmt(r), flush=True)
        if r["a2_check_src"].get("applicable"):
            print("   A2 ok:", r["a2_check_src"]["ok"], flush=True)
        save(r)


def paired():
    allr = json.loads(OUT.read_text())
    npz = {k.split("|")[0]: v["npz"] for k, v in allr.items() if k.endswith("|selection|plain|templates80|n3000")}
    npz.update({k.split("|")[0]: v["npz"] for k, v in allr.items() if k.endswith("|selection|boardlike|templates80|n3000")})
    out = {}
    pairs = [("p9995", "a2"), ("B1", "p9995"), ("B2", "p9995"), ("B2", "B1")]
    pairs += [(f"{m}_bl", m) for m in ("a2", "p9995", "B1", "B2")]
    pairs += [(a + "_bl", b + "_bl") for a, b in [("p9995", "a2"), ("B1", "p9995"), ("B2", "p9995"), ("B2", "B1")]]
    for a, b in [(m, None) for m in npz if m != "fp32"] + pairs:
        if a not in npz or (b is not None and b not in npz):
            continue
        for metric in ("correct1", "correct5"):
            d = zs_eval.paired_diff(npz[a], npz[b] if b else None, metric=metric)
            out[f"{a} - {b or 'fp32'} [{metric}]"] = d
            k = "top1" if metric == "correct1" else "top5"
            print(f"{a:>9s} - {b or 'fp32':<9s} {k}: {100 * d['diff_' + k]:+6.2f} pp  CI [{100 * d['diff_' + k + '_ci95'][0]:+.2f}, "
                  f"{100 * d['diff_' + k + '_ci95'][1]:+.2f}]  discordant {d['a_only']}/{d['b_only']}  p={d['sign_test_p']:.2g}")
    PAIR.write_text(json.dumps(out, indent=1))


def torchcheck(k=8):
    """PyTorch image encoder (the checkpoint, reparameterize=False) vs mci0_256_fp32.onnx on the first k SELECTION images."""
    import torch, mobileclip
    torch.set_num_threads(4)
    model, _, _ = mobileclip.create_model_and_transforms("mobileclip_s0", pretrained=str(R / "checkpoints" / "mobileclip_s0.pt"),
                                                         reparameterize=False)
    model.eval()
    U, y, ids = zs_eval.load_split("selection", k)
    X = U.astype(np.float32) / np.float32(255)
    with torch.no_grad():
        t = model.encode_image(torch.from_numpy(X)).float().numpy()
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    F = np.load(W / "results" / "fp32ref_selection.npz")["E"][:k]
    cos = (t * F).sum(1)
    res = {"n": k, "cos_torch_vs_onnx_fp32": cos.tolist(), "min": float(cos.min())}
    print(res); (W / "results" / "torch_vs_onnx_fp32.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    for step in sys.argv[1:] or ["plain", "paired"]:
        {"plain": lambda: run("plain"), "boardlike": lambda: run("boardlike"), "paired": paired, "torchcheck": torchcheck}[step]()
