import os
"""Task 1: isolate MCi0 image encoder, reparameterize, verify, export FP32 ONNX, report stats."""
import pathlib, collections, numpy as np, torch, onnx, onnxruntime as ort
from PIL import Image
import mobileclip
from mobileclip.modules.common.mobileone import reparameterize_model

ROOT = pathlib.Path(__file__).parent
CKPT = ROOT / "checkpoints" / "mobileclip_s0.pt"
IMGS = sorted((ROOT / "calib_images").glob("*.jpg")) + (sorted(pathlib.Path(os.environ["EXTRA_CALIB_IMAGES"]).glob("*.jpg")) if os.environ.get("EXTRA_CALIB_IMAGES") else [])
ONNX_FP32 = ROOT / "mci0_256_fp32.onnx"

torch.manual_seed(0)
model, _, preprocess = mobileclip.create_model_and_transforms("mobileclip_s0", pretrained=str(CKPT), reparameterize=False)
enc = model.image_encoder.eval()
enc_rep = reparameterize_model(enc).eval()
print("preprocess:", preprocess)

x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in IMGS])
np.save(ROOT / "calib_inputs.npy", x[:24].numpy())  # COCO images only, NCHW float, used for PTQ
with torch.no_grad():
    e0, e1 = enc(x), enc_rep(x)
cos = torch.nn.functional.cosine_similarity(e0, e1, dim=1)
print(f"embedding shape {tuple(e0.shape)}; reparam cosine over {len(IMGS)} images: "
      f"min {cos.min():.7f} mean {cos.mean():.7f}; max |diff| {(e0 - e1).abs().max():.2e}")

n_orig = sum(p.numel() for p in enc.parameters())
n_rep = sum(p.numel() for p in enc_rep.parameters()) + sum(b.numel() for n, b in enc_rep.named_buffers() if "running" not in n and "num_batches" not in n)
print(f"params: original {n_orig:,}  reparameterized {sum(p.numel() for p in enc_rep.parameters()):,} (+buffers -> {n_rep:,})")

torch.onnx.export(enc_rep, torch.randn(1, 3, 256, 256), ONNX_FP32, input_names=["input"], output_names=["embedding"],
                  opset_version=17, dynamo=False, do_constant_folding=True)

# ---- verify ONNX vs torch ----
sess = ort.InferenceSession(str(ONNX_FP32), providers=["CPUExecutionProvider"])
o = np.concatenate([sess.run(None, {"input": x[i:i + 1].numpy()})[0] for i in range(len(x))])
cos_onnx = torch.nn.functional.cosine_similarity(torch.from_numpy(o), e1, dim=1)
print(f"ONNX vs torch(reparam) cosine min {cos_onnx.min():.7f}")

# ---- stats from the ONNX graph ----
m = onnx.shape_inference.infer_shapes(onnx.load(str(ONNX_FP32)))
g = m.graph
shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in list(g.value_info) + list(g.input) + list(g.output)}
inits = {t.name: t for t in g.initializer}
consts = {}
for n in g.node:
    if n.op_type == "Constant":
        consts[n.output[0]] = onnx.numpy_helper.to_array(n.attribute[0].t)
params = sum(int(np.prod(t.dims)) for t in inits.values()) + sum(c.size for c in consts.values() if c.size > 1)

macs = collections.Counter()
for n in g.node:
    if n.op_type == "Conv":
        w = inits[n.input[1]]; out = shapes[n.output[0]]
        macs["Conv"] += int(np.prod(out)) * int(np.prod(w.dims[1:]))
    elif n.op_type in ("MatMul", "Gemm"):
        a = shapes.get(n.input[0]); out = shapes[n.output[0]]
        k = a[-1] if n.op_type == "MatMul" or not any(at.name == "transA" and at.i for at in n.attribute) else a[-2]
        kind = "MatMul(const)" if (n.input[1] in inits or n.input[1] in consts) else "MatMul(act x act)"
        macs[kind] += int(np.prod(out)) * k
print(f"ONNX params {params:,} ({params * 4 / 2**20:.2f} MiB fp32, ~{params / 2**20:.2f} MiB int8)")
print(f"MACs total {sum(macs.values()) / 1e9:.4f} G ; by kind: " + ", ".join(f"{k} {v / 1e6:.1f}M" for k, v in macs.items()))
ops = collections.Counter(n.op_type for n in g.node)
print("operator types:", ", ".join(f"{k}:{v}" for k, v in ops.most_common()))
