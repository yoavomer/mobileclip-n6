"""W5 task 1: FP32 MobileCLIP-S0 text embeddings for the demo vocabulary (pc/vocab.py).

Same recipe as night/W0_data/build_text.py (the one that built text_imagenet.npy):
  templates80: per entry the 80 open_clip OPENAI_IMAGENET_TEMPLATES with {name}=prompt_name are encoded,
               each L2-normalized, averaged, L2-normalized again (= clip_benchmark zero_shot_classifier).
  simple     : the single prompt "a photo of a {name}." -> L2-normalized.
Writes results/text_vocab_templates80.npy, results/text_vocab_simple.npy (N x 512 float32) + results/text_vocab_info.json.
Self-check: re-encodes 3 ImageNet class names with the same code and compares with W0's text_imagenet(_simple).npy rows.
torch CPU, 2 threads."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, time, hashlib, pathlib
import numpy as np, torch

R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
H = R / "night" / "W5_demo"
sys.path.insert(0, str(H / "pc")); sys.path.insert(0, str(R / "ml-mobileclip"))
torch.set_num_threads(2)
import vocab  # noqa: E402
import mobileclip  # noqa: E402
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES  # noqa: E402

model, _, _ = mobileclip.create_model_and_transforms(
    "mobileclip_s0", pretrained=str(R / "checkpoints" / "mobileclip_s0.pt"), reparameterize=False)
model.eval(); tok = mobileclip.get_tokenizer("mobileclip_s0")


def enc(texts):
    with torch.no_grad():
        e = model.encode_text(tok(texts)).float()
    return e / e.norm(dim=-1, keepdim=True)


def emb_t80(name):
    e = enc([t(name) for t in OPENAI_IMAGENET_TEMPLATES])
    m = e.mean(dim=0); return (m / m.norm()).numpy()


def emb_simple(names):
    return enc([f"a photo of a {n}." for n in names]).numpy()


t0 = time.time()
# self-check against W0 (same code path must reproduce W0's rows)
W0 = R / "night" / "W0_data"
T0, S0 = np.load(W0 / "text_imagenet.npy"), np.load(W0 / "text_imagenet_simple.npy")
chk = {}
for c in (504, 487, 761):
    a = emb_t80(IMAGENET_CLASSNAMES[c]); b = emb_simple([IMAGENET_CLASSNAMES[c]])[0]
    chk[IMAGENET_CLASSNAMES[c]] = {"t80_maxabs_vs_W0": float(np.abs(a - T0[c]).max()), "simple_maxabs_vs_W0": float(np.abs(b - S0[c]).max())}
print("self-check vs W0", chk, flush=True)

P = vocab.prompt_names()
simple = emb_simple(P).astype(np.float32)
t80 = np.stack([emb_t80(p) for p in P]).astype(np.float32)
out = H / "results"
np.save(out / "text_vocab_templates80.npy", t80); np.save(out / "text_vocab_simple.npy", simple)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
G = t80 @ t80.T; Gs = simple @ simple.T
np.fill_diagonal(G, -1); np.fill_diagonal(Gs, -1)
D = vocab.display_names()
info = {"n": len(P), "display": D, "prompt_names": P,
        "templates80_sha256": sha(out / "text_vocab_templates80.npy"), "simple_sha256": sha(out / "text_vocab_simple.npy"),
        "self_check_vs_W0": chk,
        "cos_t80_vs_simple_per_entry": dict(zip(D, [round(float(x), 4) for x in (t80 * simple).sum(1)])),
        "closest_other_entry_t80": {D[i]: [D[int(G[i].argmax())], round(float(G[i].max()), 4)] for i in range(len(D))},
        "closest_other_entry_simple": {D[i]: [D[int(Gs[i].argmax())], round(float(Gs[i].max()), 4)] for i in range(len(D))},
        "pairs_cos_t80": {f"{a}/{b}": round(float(t80[D.index(a)] @ t80[D.index(b)]), 4) for a, b in vocab.PAIRS},
        "pairs_cos_simple": {f"{a}/{b}": round(float(simple[D.index(a)] @ simple[D.index(b)]), 4) for a, b in vocab.PAIRS},
        "seconds": round(time.time() - t0, 1), "torch_threads": 2}
json.dump(info, open(out / "text_vocab_info.json", "w"), indent=1)
print(json.dumps({k: info[k] for k in ("n", "pairs_cos_t80", "pairs_cos_simple", "seconds")}, indent=1))
