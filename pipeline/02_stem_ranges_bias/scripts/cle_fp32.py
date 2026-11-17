"""M2 step 1: float error of cross-layer equalization (CLE) across the stem GELUs, FP32 model only (no quantization).
Pair A: conv0 out-channel c scaled by 1/s_c (weight + bias), conv1 (depthwise) channel c weight x s_c.
Pair B: conv1 out-channel c scaled by 1/t_c (weight + bias), conv2 in-channel c weight x t_c.
Exact for ReLU; for GELU the network computes s*GELU(x/s) instead of GELU(x), so the FP32 embedding changes.
Scale choices: 'dfq' = Nagel et al. weight-range rule s = sqrt(r1/r2) (r = per-channel max|W|);
               'act'  = activation-range rule s_c = r_act_c / max_c r_act (all conv-output channels reach the max range,
                        from the FP32 calib-pool per-channel min/max); also each pair alone and a mild version (s clipped
                        to [0.5, 2]).
Metric: cosine of the modified FP32 embedding vs the original FP32 embedding, and top-1 agreement (ImageNet text),
on the first N calib-pool images.  -> results/cle_fp32_n<N>.json"""
import sys, json, copy, pathlib
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
base = onnx.load(str(w2lib.FP32_PRE))
ini = {t.name: t for t in base.graph.initializer}
W = {k: nh.to_array(ini[f"model.patch_embed.{k}.reparam_conv.weight"]).astype(np.float64) for k in range(3)}
B = {k: nh.to_array(ini[f"model.patch_embed.{k}.reparam_conv.bias"]).astype(np.float64) for k in range(3)}
cs = np.load(w2lib.W2 / "results" / "fp32_gelu_chanstats_n200.npz")
names = [str(x) for x in cs["names"]]
ract = {k: np.maximum(np.abs(cs[f"cmin{names.index(w2lib.STEM[k])}"]), np.abs(cs[f"cmax{names.index(w2lib.STEM[k])}"])) for k in range(3)}
U = np.load(w2lib.R / "night" / "W0_data" / "calib_pool_u8.npy", mmap_mode="r")[:N]
T = np.load(w2lib.R / "night" / "W0_data" / "text_imagenet.npy").astype(np.float64)


def embed(m):
    so = ort.SessionOptions(); so.intra_op_num_threads = 3; so.inter_op_num_threads = 1
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    E = np.stack([s.run(["embedding"], {"input": U[i:i + 1].astype(np.float32) / np.float32(255)})[0].ravel() for i in range(N)]).astype(np.float64)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def variant(sA=None, sB=None):
    m = copy.deepcopy(base); I = {t.name: t for t in m.graph.initializer}
    w0, b0, w1, b1, w2 = W[0].copy(), B[0].copy(), W[1].copy(), B[1].copy(), W[2].copy()
    if sA is not None:
        w0 /= sA[:, None, None, None]; b0 /= sA; w1 *= sA[:, None, None, None]
    if sB is not None:
        w1 /= sB[:, None, None, None]; b1 /= sB; w2 *= sB[None, :, None, None]
    for nm, a in ((f"model.patch_embed.0.reparam_conv.weight", w0), ("model.patch_embed.0.reparam_conv.bias", b0),
                  ("model.patch_embed.1.reparam_conv.weight", w1), ("model.patch_embed.1.reparam_conv.bias", b1),
                  ("model.patch_embed.2.reparam_conv.weight", w2)):
        I[nm].CopyFrom(nh.from_array(a.astype(np.float32), nm))
    return m


r0 = np.abs(W[0]).reshape(64, -1).max(1); r1 = np.abs(W[1]).reshape(64, -1).max(1)
r1o = r1; r2i = np.abs(W[2]).reshape(64, 64).max(0)
sA_dfq = np.sqrt(r0 / r1); sB_dfq = np.sqrt(r1o / r2i)
sA_act = ract[0] / ract[0].max(); sB_act = ract[1] / ract[1].max()
clip = lambda s: np.clip(s, 0.5, 2.0)
F = embed(base)
PF = (F @ T.T).argmax(1)
out = {"N": N, "scales": {}}
cases = {"dfq_A": (sA_dfq, None), "dfq_B": (None, sB_dfq), "dfq_AB": (sA_dfq, sB_dfq),
         "act_A": (sA_act, None), "act_B": (None, sB_act), "act_AB": (sA_act, sB_act),
         "dfq_AB_clip0.5-2": (clip(sA_dfq), clip(sB_dfq)), "act_AB_clip0.5-2": (clip(sA_act), clip(sB_act)),
         "identity_check": (np.ones(64), np.ones(64))}
for k, (a, b) in cases.items():
    E = embed(variant(a, b))
    cos = (E * F).sum(1); P = (E @ T.T).argmax(1)
    out[k] = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "top1_agree_with_fp32": float((P == PF).mean()),
              "sA_range": None if a is None else [float(a.min()), float(a.max())],
              "sB_range": None if b is None else [float(b.min()), float(b.max())]}
    print(k, json.dumps(out[k]), flush=True)
json.dump(out, open(w2lib.W2 / "results" / f"cle_fp32_n{N}.json", "w"), indent=1)
