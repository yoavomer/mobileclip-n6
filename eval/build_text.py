"""W0 step 4: FP32 MobileCLIP-S0 text embeddings of the 1000 ImageNet classes.

text_imagenet.npy (DEFAULT): open_clip IMAGENET_CLASSNAMES x OPENAI_IMAGENET_TEMPLATES (80 templates), exactly as
  clip_benchmark.metrics.zeroshot_classification.zero_shot_classifier (used by ml-mobileclip/eval/zeroshot_imagenet.py):
  per class: encode the 80 prompts -> L2-normalize each -> mean -> L2-normalize.
text_imagenet_simple.npy: single prompt "a photo of a {name}." -> L2-normalize.
Both 1000x512 float32. Model: mobileclip.create_model_and_transforms("mobileclip_s0", pretrained=ckpt,
reparameterize=False) + mobileclip.get_tokenizer("mobileclip_s0"), torch CPU FP32, 4 threads."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, time, hashlib, pathlib
import numpy as np, torch

R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
W = R / "night" / "W0_data"
torch.set_num_threads(4)
sys.path.insert(0, str(R / "ml-mobileclip"))
import mobileclip  # noqa: E402
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES  # noqa: E402

assert len(IMAGENET_CLASSNAMES) == 1000 and len(OPENAI_IMAGENET_TEMPLATES) == 80
model, _, _ = mobileclip.create_model_and_transforms(
    "mobileclip_s0", pretrained=str(R / "checkpoints" / "mobileclip_s0.pt"), reparameterize=False)
model.eval(); tok = mobileclip.get_tokenizer("mobileclip_s0")

(W / "imagenet_classnames.txt").write_text("\n".join(IMAGENET_CLASSNAMES) + "\n", encoding="utf-8")
(W / "imagenet_templates80.txt").write_text("\n".join(t("{}") for t in OPENAI_IMAGENET_TEMPLATES) + "\n", encoding="utf-8")


def enc(texts):
    with torch.no_grad():
        e = model.encode_text(tok(texts)).float()
    return e / e.norm(dim=-1, keepdim=True)


t0 = time.time()
simple = []
for i in range(0, 1000, 100):
    simple.append(enc([f"a photo of a {c}." for c in IMAGENET_CLASSNAMES[i:i + 100]]))
simple = torch.cat(simple).numpy().astype(np.float32)
np.save(W / "text_imagenet_simple.npy", simple)
print("simple done", simple.shape, f"{time.time() - t0:.0f}s", flush=True)

T = np.empty((1000, 512), np.float32)
for c, name in enumerate(IMAGENET_CLASSNAMES):
    e = enc([t(name) for t in OPENAI_IMAGENET_TEMPLATES])     # 80 x 512, each L2-normalized
    m = e.mean(dim=0); m = m / m.norm()
    T[c] = m.numpy()
    if c % 100 == 99:
        print(c + 1, f"{time.time() - t0:.0f}s", flush=True)
np.save(W / "text_imagenet.npy", T)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
info = {"default": "text_imagenet.npy (80 OpenAI templates, clip_benchmark ensembling)",
        "text_imagenet.npy": {"sha256": sha(W / "text_imagenet.npy"), "templates": 80},
        "text_imagenet_simple.npy": {"sha256": sha(W / "text_imagenet_simple.npy"), "prompt": "a photo of a {name}."},
        "norms_min_max": [float(np.linalg.norm(T, axis=1).min()), float(np.linalg.norm(T, axis=1).max())],
        "mean_cos_templates80_vs_simple": float((T * simple).sum(1).mean()),
        "seconds": round(time.time() - t0, 1)}
json.dump(info, open(W / "build_text_info.json", "w"), indent=1)
print(info)
