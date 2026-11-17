"""Recreate the A2 baseline ONNX from mci0_256_qkvconv.onnx (same procedure as optionA/make_variants.py).

A2 = only quantization parameters change, the graph is untouched:
  * softmax output feeding attn·V (blocks network.7.0 and 7.1): scale 1/127, zero point 0 (int8)
  * Q·Kᵀ weights operand (token_mixer/Mul_output_0, blocks 7.0 and 7.1): symmetric, zero point 0,
    scale = max(|min|, |max|) / 127 of the ORIGINAL calibration range.
The calibration range is recomputed exactly as the original PTQ saw it: mci0_256_fp32_pre.onnx on the
24 images in calib_inputs.npy (ORT CPU). The values obtained when A2 was made are kept below as a check.

usage: python recreate_a2.py [out.onnx]      (default: baseline/mci0_256_a2_recreated.onnx)"""
import sys, copy, hashlib, pathlib, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh

R = pathlib.Path(__file__).resolve().parent.parent
SRC = R / "mci0_256_qkvconv.onnx"
SRC_SHA256 = "a02a4bf0802fb8c0ca0dd949787f53f6d0d577b1038957833226c109c0e12869"
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else R / "baseline" / "mci0_256_a2_recreated.onnx"
B = lambda b: f"/model/network.7/network.7.{b}/token_mixer/"
SOFT = [B(b) + "Softmax_output_0" for b in "01"]
QK = [B(b) + "Mul_output_0" for b in "01"]
EXPECTED_RANGE = {QK[0]: (-1.2311087846755981, 1.3456376791000366), QK[1]: (-1.7568382024765015, 1.869292974472046)}

assert hashlib.sha256(open(SRC, "rb").read()).hexdigest() == SRC_SHA256, "source ONNX is not the expected mci0_256_qkvconv.onnx"

# original calibration range of the Q·Kᵀ weights tensors
pre = onnx.load(str(R / "mci0_256_fp32_pre.onnx"))
vi = {v.name: v for v in onnx.shape_inference.infer_shapes(pre).graph.value_info}
for t in QK + SOFT: pre.graph.output.append(vi[t])
sess = ort.InferenceSession(pre.SerializeToString(), providers=["CPUExecutionProvider"])
calib = np.load(R / "calib_inputs.npy")
rng = {t: [np.inf, -np.inf] for t in QK + SOFT}
for i in range(len(calib)):
    for t, v in zip(QK + SOFT, sess.run(QK + SOFT, {"input": calib[i:i + 1]})):
        rng[t] = [min(rng[t][0], float(v.min())), max(rng[t][1], float(v.max()))]
for t in QK:
    print(f"calibration range {t}: {rng[t]}  (expected {EXPECTED_RANGE[t]})")
    assert tuple(rng[t]) == EXPECTED_RANGE[t], "calibration range differs from the one used for A2"

changes = {t: (1.0 / 127.0, 0) for t in SOFT}
for t in QK: changes[t] = (max(abs(rng[t][0]), abs(rng[t][1])) / 127.0, 0)

m = copy.deepcopy(onnx.load(str(SRC)))
for t, (sc, zp) in changes.items():
    for suf, val in (("_scale", np.array(sc, np.float32)), ("_zero_point", np.array(zp, np.int8))):
        ini = next(x for x in m.graph.initializer if x.name == t + suf)
        old = nh.to_array(ini).item(); ini.CopyFrom(nh.from_array(val, ini.name))
        print(f"  {ini.name}: {old} -> {val.item()}")
onnx.checker.check_model(m)
onnx.save(m, str(OUT))
print("wrote", OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest())
