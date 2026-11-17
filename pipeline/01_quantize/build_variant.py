"""Build one calibration variant. Only activation calibration differs from the original pipeline:
  1. quantize_static on mci0_256_fp32_pre.onnx with EXACTLY the original settings of quantize_mci0.py
     (QDQ, QInt8/QInt8, per_channel=True, no other options), only calibrate_method / images / percentile change.
     calibration_cache_path only writes the computed tensor ranges to JSON (no effect on quantization).
  2. qkv MatMul -> Conv 1x1 rewrite, unchanged code from make_full_qkvconv.py (gives the "qkvconv" model).
  3. A2 rule, same code as baseline/recreate_a2.py: softmax output feeding attn·V -> scale 1/127, zp 0;
     Q·Kᵀ weights operand (token_mixer/Mul_output_0) -> zp 0, scale = max(|lo|,|hi|)/127 of THIS variant's
     calibration range (from the cache JSON).
usage: build_variant.py <name> <MinMax|Percentile|Entropy> <inputs.npy> [percentile|-] [stride]
  stride: optional ORT CalibStridedMinMax (chunked calibration, memory only; changes histogram results, see stride_test.json)"""
import sys, json, copy, hashlib, pathlib, time, numpy as np, onnx
from onnx import numpy_helper as nh
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod
R = pathlib.Path(__file__).resolve().parent.parent; import os; Q = pathlib.Path(os.environ.get("QUANT_OUT", R / "quant"))   # QUANT_OUT: optional output folder
sys.path.insert(0, str(R))
from make_full_qkvconv import rewrite_int8, clean_and_save          # unchanged rewrite used for mci0_256_qkvconv

name, method, npy = sys.argv[1], sys.argv[2], sys.argv[3]
pct = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
stride = int(sys.argv[5]) if len(sys.argv) > 5 else None
PRE = R / "mci0_256_fp32_pre.onnx"
X = np.load(npy) if npy.endswith(".npy") else np.load(R / npy)


class Reader(CalibrationDataReader):                                 # same as quantize_mci0.py
    def __init__(self):
        self.it = iter([{"input": X[i:i + 1]} for i in range(len(X))])

    def get_next(self):
        return next(self.it, None)

    def __len__(self):                                               # only used by CalibStridedMinMax
        return len(X)

    def set_range(self, start_index, end_index):                     # only used by CalibStridedMinMax
        self.it = iter([{"input": X[i:i + 1]} for i in range(start_index, min(end_index, len(X)))])


out_int8, out_qkvc, out_a2 = Q / f"{name}_int8.onnx", Q / f"{name}_qkvconv.onnx", Q / f"{name}_a2.onnx"
cache = Q / f"{name}_ranges.json"
extra = {"CalibPercentile": pct} if pct is not None else {}
if stride: extra["CalibStridedMinMax"] = stride
t0 = time.time()
REUSE = len(sys.argv) > 6 and sys.argv[6] == "reuse"                # redo only steps 2-3 from an existing int8 + cache
if not REUSE:
  if cache.exists(): cache.unlink()
  quantize_static(str(PRE), str(out_int8), Reader(), quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=True, calibrate_method=getattr(CalibrationMethod, method),
                extra_options=extra, calibration_cache_path=str(cache))
t_quant = time.time() - t0
clean_and_save(rewrite_int8(onnx.load(str(out_int8))), out_qkvc)

# ---- A2 rule (identical code to baseline/recreate_a2.py, range taken from this variant's calibration) ----
rng_all = json.load(open(cache))
def rng_of(t):                     # cache format: {"data": {tensor: {"lowest": {"data": [x], ...}, "highest": {...}}}}
    v = rng_all["data"][t]
    g = lambda x: float(np.asarray(x["data"]).ravel()[0]) if isinstance(x, dict) else float(x)   # {"data":[x]} / {"data":x} / x
    return g(v["lowest"]), g(v["highest"])
B = lambda b: f"/model/network.7/network.7.{b}/token_mixer/"
SOFT = [B(b) + "Softmax_output_0" for b in "01"]
QK = [B(b) + "Mul_output_0" for b in "01"]
rng = {t: rng_of(t) for t in QK}
changes = {t: (1.0 / 127.0, 0) for t in SOFT}
for t in QK: changes[t] = (max(abs(rng[t][0]), abs(rng[t][1])) / 127.0, 0)
m = copy.deepcopy(onnx.load(str(out_qkvc)))
log = []
for t, (sc, zp) in changes.items():
    for suf, val in (("_scale", np.array(sc, np.float32)), ("_zero_point", np.array(zp, np.int8))):
        ini = next(x for x in m.graph.initializer if x.name == t + suf)
        old = nh.to_array(ini).item(); ini.CopyFrom(nh.from_array(val, ini.name)); log.append((ini.name, old, val.item()))
onnx.checker.check_model(m)
onnx.save(m, str(out_a2))
# confirm the A2 rule in the saved model: zp 0 on the weights operand of every attention MatMul
mm = onnx.load(str(out_a2)); ini = {t.name: nh.to_array(t) for t in mm.graph.initializer}
prod = {o: n for n in mm.graph.node for o in n.output}
zps = {}
for n in mm.graph.node:
    if n.op_type == "MatMul" and "token_mixer" in n.name and n.name.split("/")[-1] in ("MatMul", "MatMul_1"):
        dq = prod[n.input[0]]; zps[n.name] = int(ini[dq.input[2]])
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
info = {"name": name, "method": method, "inputs": str(npy), "n_images": int(len(X)), "percentile": pct, "stride": stride, "quant_seconds": round(t_quant, 1),
        "int8": [str(out_int8), sha(out_int8)], "qkvconv": [str(out_qkvc), sha(out_qkvc)], "a2": [str(out_a2), sha(out_a2)],
        "qk_ranges": rng, "a2_changes": log, "attention_weights_operand_zp": zps}
json.dump(info, open(Q / f"{name}_info.json", "w"), indent=1)
print(json.dumps(info, indent=1))
