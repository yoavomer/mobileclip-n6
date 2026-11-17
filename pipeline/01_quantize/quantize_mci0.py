"""Task 1.5: static PTQ, ONNX QDQ, INT8 activations+weights, per-channel weights (ST-recommended recipe)."""
import pathlib, collections, numpy as np, onnx, onnxruntime as ort
from onnxruntime.quantization import (quantize_static, CalibrationDataReader, QuantFormat,
                                      QuantType, CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = pathlib.Path(__file__).parent
FP32, PRE, INT8 = ROOT / "mci0_256_fp32.onnx", ROOT / "mci0_256_fp32_pre.onnx", ROOT / "mci0_256_int8.onnx"
calib = np.load(ROOT / "calib_inputs.npy")  # (24,3,256,256) float in [0,1]

quant_pre_process(str(FP32), str(PRE), skip_symbolic_shape=False)


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = iter([{"input": calib[i:i + 1]} for i in range(len(calib))])

    def get_next(self):
        return next(self.it, None)


quantize_static(str(PRE), str(INT8), Reader(), quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=True, calibrate_method=CalibrationMethod.MinMax)

for p in (PRE, INT8):
    g = onnx.load(str(p)).graph
    print(p.name, "ops:", ", ".join(f"{k}:{v}" for k, v in collections.Counter(n.op_type for n in g.node).most_common()))

# quick sanity: INT8 (fake-quant on CPU) vs FP32 embedding cosine on the calibration images
s32 = ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
s8 = ort.InferenceSession(str(INT8), providers=["CPUExecutionProvider"])
a = np.concatenate([s32.run(None, {"input": calib[i:i + 1]})[0] for i in range(len(calib))])
b = np.concatenate([s8.run(None, {"input": calib[i:i + 1]})[0] for i in range(len(calib))])
cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
print(f"INT8 vs FP32 cosine on calibration set (optimistic, same images): min {cos.min():.4f} mean {cos.mean():.4f}")
