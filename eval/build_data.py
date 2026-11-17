"""W0 step 1-3: ImageNetV2 matched-frequency -> fixed-seed stratified splits -> uint8 NCHW arrays.

Splits (per class, 10 images each): SELECTION 3, TEST 3 (locked), CALIB_POOL 1, UNUSED 3.
Per class: the 10 file names are sorted, then permuted with one numpy Generator(SEED) used in class order 0..999.
Order inside a split is "round-major": round r (r = 0..k-1) holds one image of every class, classes in a
per-round random permutation (same Generator). So the first 1000 entries of SELECTION contain each class once,
and any prefix n is a random, (almost) class-balanced subset.

Preprocess = EXACTLY the existing pipeline (quant/prep.py, export_mci0.py):
    _, _, preprocess = mobileclip.create_model_and_transforms("mobileclip_s0", pretrained=ckpt, reparameterize=False)
    x = preprocess(Image.open(p).convert("RGB"))      # Resize(256, bilinear, antialias) -> CenterCrop(256) -> ToTensor
We store u8 = rint(x*255) and assert, for EVERY image, that u8.astype(float32)/255 == x bit-exactly.
Independent cross-check on a sample: PIL Resize+CenterCrop -> np.asarray (no ToTensor) equals u8.

Usage: python -I build_data.py   (run with the project .venv)"""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, time, hashlib, pathlib
import numpy as np, torch
from PIL import Image

R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
W = R / "night" / "W0_data"
SRC = R / "data" / "imagenetv2_mf" / "imagenetv2-matched-frequency-format-val"
SEED = 20261108
K = {"selection": 3, "test": 3, "calib_pool": 1, "unused": 3}
torch.set_num_threads(4)
sys.path.insert(0, str(R / "ml-mobileclip"))
import mobileclip  # noqa: E402

_, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s0", pretrained=str(R / "checkpoints" / "mobileclip_s0.pt"), reparameterize=False)
print(preprocess)

# ---------- splits ----------
rng = np.random.default_rng(SEED)
per_class = {}
for c in range(1000):
    files = sorted(p.name for p in (SRC / str(c)).iterdir() if p.is_file())
    assert len(files) == 10, (c, len(files))
    perm = rng.permutation(10)
    files = [files[i] for i in perm]
    o = 0; per_class[c] = {}
    for s, k in K.items():
        per_class[c][s] = files[o:o + k]; o += k
splits = {}
for s, k in K.items():
    lst = []
    for r in range(k):
        for c in rng.permutation(1000):
            lst.append((f"{int(c)}/{per_class[int(c)][s][r]}", int(c)))
    splits[s] = lst
# no overlap
allf = [f for s in splits for f, _ in splits[s]]
assert len(allf) == len(set(allf)) == 10000
for s, lst in splits.items():
    (W / f"{s}_files.txt").write_text("".join(f"{f}\t{c}\n" for f, c in lst))
    cnt = np.bincount([c for _, c in lst], minlength=1000)
    assert (cnt == K[s]).all()
    print(s, len(lst), "per class", K[s])


# ---------- preprocess ----------
def load(rel):
    return preprocess(Image.open(SRC / rel).convert("RGB")).numpy()


info = {"seed": SEED, "source": str(SRC), "transform": str(preprocess), "splits": {}}
for s in ("selection", "test", "calib_pool"):
    lst = splits[s]; n = len(lst)
    U = np.empty((n, 3, 256, 256), np.uint8)
    t0 = time.time(); worst = 0.0
    for i, (rel, c) in enumerate(lst):
        x = load(rel)
        assert x.dtype == np.float32 and x.shape == (3, 256, 256), (rel, x.dtype, x.shape)
        u = np.rint(x * 255.0).astype(np.int32)
        assert u.min() >= 0 and u.max() <= 255
        u = u.astype(np.uint8)
        back = u.astype(np.float32) / np.float32(255)
        d = float(np.abs(back - x).max()); worst = max(worst, d)
        assert np.array_equal(back, x), (rel, d)
        U[i] = u
    y = np.array([c for _, c in lst], np.int64)
    np.save(W / f"{s}_u8.npy", U); np.save(W / f"{s}_labels.npy", y)
    info["splits"][s] = {"n": n, "max_abs_diff_u8div255_vs_transform_all_images": worst, "seconds": round(time.time() - t0, 1)}
    print(s, U.shape, "all", n, "images: max |u8/255 - transform| =", worst, f"{time.time() - t0:.0f}s")

# independent cross-check (no ToTensor, no rounding): PIL resize+crop -> asarray must equal the stored u8
tf_geo = preprocess.transforms[:2]
chk = {}
for s, m in (("selection", 40), ("test", 10), ("calib_pool", 10)):
    U = np.load(W / f"{s}_u8.npy", mmap_mode="r"); ok = 0; mx = 0.0
    for i in range(m):
        rel = splits[s][i][0]
        img = Image.open(SRC / rel).convert("RGB")
        for t in tf_geo:
            img = t(img)
        a = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
        ok += int(np.array_equal(a, U[i]))
        x = load(rel); mx = max(mx, float(np.abs(a.astype(np.float32) / 255 - x).max()))
    chk[s] = {"n": m, "identical": ok, "max_abs_diff_float": mx}
    print("independent check", s, chk[s])
info["independent_check_pil_resize_crop"] = chk
json.dump(info, open(W / "build_data_info.json", "w"), indent=1)
print("done")
