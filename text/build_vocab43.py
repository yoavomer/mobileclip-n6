"""Demo run 9.11: extend W5's 41-word demo vocabulary with the objects that had no word (painting, mirror).

Same text-encoder method as W5 (night/W5_demo/pc/build_text.py, itself = W0 build_text.py):
FP32 MobileCLIP-S0 text encoder, the 80 OpenAI ImageNet templates with {name} = prompt name, each prompt L2-normalized,
averaged, L2-normalized again. torch CPU, 2 threads.
Self-check: the 41 old entries are re-encoded with this code and compared with W5's text_vocab_templates80.npy.
Board file: W5's own MCTX v1 packer (make_board_files.pack / parse), int8 rows + float32 row scales + CRC32.
Writes night/demo_run/pc/vocab43/: text_vocab43_templates80.npy, demo_text.bin, demo_text.txt, vocab43_info.json.
The model is not touched. usage: .venv/Scripts/python.exe -I build_vocab43.py"""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, time, hashlib, pathlib
import numpy as np, torch

R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
W5 = R / "night" / "W5_demo"
OUT = R / "night" / "demo_run" / "pc" / "vocab43"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(W5 / "pc")); sys.path.insert(0, str(R / "ml-mobileclip"))
torch.set_num_threads(2)
import vocab as V5  # noqa: E402  (W5 vocabulary, unchanged)
import make_board_files as MBF  # noqa: E402  (W5 packer, unchanged)
import mobileclip  # noqa: E402
from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES  # noqa: E402

NEW = [("painting", "painting", []),   # no ImageNet class: demo word only
       ("mirror", "mirror", [])]       # no ImageNet class: demo word only
VOCAB = list(V5.VOCAB) + NEW

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


t0 = time.time()
P = [p for _, p, _ in VOCAB]; D = [d for d, _, _ in VOCAB]
T = np.stack([emb_t80(p) for p in P]).astype(np.float32)
old = np.load(W5 / "results" / "text_vocab_templates80.npy")
dmax = float(np.abs(T[:len(old)] - old).max())
np.save(OUT / "text_vocab43_templates80.npy", T)
b, s, q = MBF.pack(T, "templates80")
(OUT / "demo_text.bin").write_bytes(b)
(OUT / "demo_text.txt").write_bytes(("\n".join(D) + "\n").encode("ascii"))
p = MBF.parse(b)
deq = p["q"].astype(np.float64) * p["scales"][:, None].astype(np.float64)
cos = (deq * T).sum(1) / np.linalg.norm(deq, axis=1) / np.linalg.norm(T, axis=1)
G = T @ T.T; np.fill_diagonal(G, -1)
sha = lambda f: hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
w5bin = (W5 / "board_files" / "demo_text.bin").read_bytes()
pw5 = MBF.parse(w5bin)
info = {"n": len(D), "display": D, "prompt_names": P, "new_entries": [d for d, _, _ in NEW],
        "selfcheck_old41_maxabs_vs_W5_fp32": dmax,
        "old41_int8_rows_equal_W5_bin": bool(np.array_equal(p["q"][:41], pw5["q"])),
        "old41_scales_equal_W5_bin": bool(np.array_equal(p["scales"][:41], pw5["scales"])),
        "int8_vs_fp32_row_cos_min": float(cos.min()), "int8_vs_fp32_row_cos_mean": float(cos.mean()),
        "parse_back_q_equal": bool(np.array_equal(p["q"], q)), "crc32": "0x%08x" % p["crc"], "bytes": len(b),
        "closest_other_entry": {D[i]: [D[int(G[i].argmax())], round(float(G[i].max()), 4)] for i in range(len(D))},
        "sha256": {f: sha(OUT / f) for f in ("text_vocab43_templates80.npy", "demo_text.bin", "demo_text.txt")},
        "seconds": round(time.time() - t0, 1)}
json.dump(info, open(OUT / "vocab43_info.json", "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k not in ("closest_other_entry", "display", "prompt_names")}, indent=1))
print("closest for new:", {k: info["closest_other_entry"][k] for k in ("painting", "mirror", "pen", "glass", "book", "mouse", "television")})
