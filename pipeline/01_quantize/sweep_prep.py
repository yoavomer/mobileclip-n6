"""Preprocess test200 (mobileclip_s0 preprocess, as before) and build FP32 text embeddings for the 1000 ImageNet class
names (open_clip.zero_shot_metadata.IMAGENET_CLASSNAMES, prompt 'a photo of a {name}'). The COCO-80 set is quant/text80.npy."""
import pathlib, numpy as np, torch
from PIL import Image
import mobileclip
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
R = pathlib.Path(__file__).resolve().parents[2]; S = R / "quant" / "sweep"
model, _, preprocess = mobileclip.create_model_and_transforms("mobileclip_s0", pretrained=str(R / "checkpoints" / "mobileclip_s0.pt"), reparameterize=False)
model.eval(); tok = mobileclip.get_tokenizer("mobileclip_s0")
names = (S / "test200.txt").read_text().split()
X = np.stack([preprocess(Image.open(R / "data" / "coco_val2017_test" / n).convert("RGB")).numpy() for n in names]).astype(np.float32)
np.save(S / "test200.npy", X); print("test200", X.shape)
emb = []
with torch.no_grad():
    for i in range(0, 1000, 100):
        emb.append(model.encode_text(tok([f"a photo of a {c}" for c in IMAGENET_CLASSNAMES[i:i + 100]])).float())
t = torch.cat(emb); t = (t / t.norm(dim=-1, keepdim=True)).numpy()
np.save(S / "text_imagenet1000.npy", t); (S / "imagenet1000.txt").write_text("\n".join(IMAGENET_CLASSNAMES) + "\n")
print("imagenet text", t.shape)
