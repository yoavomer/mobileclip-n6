"""M4: bias correction of the weight-quantization error (Nagel et al. 2019, empirical form of E[(W_q - W) x]).
For every chosen Conv of the quantized model (int8 per-channel weights, int32 bias):
   delta[c] = mean over calib images, positions of conv(x_fp32, W_q - W_fp32)[c]      (no bias, same stride/pad/group)
   x_fp32 = the FP32 input of that conv (mci0_256_fp32_pre.onnx on calib-pool images)
   new int32 bias[c] = old[c] - round(delta[c] / bias_scale[c])        (bias_scale = s_in * s_w[c], unchanged)
usage: python -I bias_corr.py <out.onnx> <src.onnx> <stem|all> [N=200] [offset=0]  -> also <out>.json with per-conv stats"""
import sys, json, pathlib, time
import numpy as np, onnx, onnxruntime as ort, torch
from onnx import numpy_helper as nh
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib

out, src, scope = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
OFF = int(sys.argv[5]) if len(sys.argv) > 5 else 0   # first calib-pool image used
m = w2lib.load(src)
init, prod, cons = w2lib.maps(m)
fp = onnx.load(str(w2lib.FP32_PRE)); fpi = {t.name: nh.to_array(t) for t in fp.graph.initializer}
fpnodes = {n.name: n for n in fp.graph.node if n.op_type == "Conv"}
fpout = {o for n in fp.graph.node for o in n.output} | {"input"}
jobs = []
for n in m.graph.node:
    if n.op_type != "Conv": continue
    if scope == "stem" and "/patch_embed/" not in n.name: continue
    xdq, wdq = prod.get(n.input[0]), prod.get(n.input[1])
    bdq = prod.get(n.input[2]) if len(n.input) > 2 else None
    if xdq is None or wdq is None or bdq is None or xdq.op_type != "DequantizeLinear" or wdq.op_type != "DequantizeLinear" \
            or bdq.op_type != "DequantizeLinear":
        continue
    if wdq.input[0] not in init or bdq.input[0] not in init: continue
    xq = prod.get(xdq.input[0])
    xf = xq.input[0] if xq is not None and xq.op_type == "QuantizeLinear" else None
    if xf is None or xf not in fpout or n.name not in fpnodes: continue
    fn = fpnodes[n.name]
    wfp = fpi.get(fn.input[1])
    if wfp is None: continue
    wq = nh.to_array(init[wdq.input[0]]).astype(np.float64)
    sw = nh.to_array(init[wdq.input[1]]).astype(np.float64); zw = nh.to_array(init[wdq.input[2]]).astype(np.float64)
    shp = [-1] + [1] * (wq.ndim - 1)
    Wd = (wq - zw.reshape(shp)) * sw.reshape(shp)
    if Wd.shape != wfp.shape: continue
    att = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
    jobs.append(dict(node=n.name, x=xf, dW=torch.from_numpy(Wd - wfp.astype(np.float64)), stride=tuple(att.get("strides", [1, 1])),
                     pads=att.get("pads", [0, 0, 0, 0]), group=int(att.get("group", 1)), dil=tuple(att.get("dilations", [1, 1])),
                     bq=bdq.input[0], bs=nh.to_array(init[bdq.input[1]]).astype(np.float64), sum=np.zeros(Wd.shape[0]), cnt=0))
print(f"{len(jobs)} convs", flush=True)
xs = sorted({j["x"] for j in jobs})
for t in xs:
    if t != "input": fp.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
so = ort.SessionOptions(); so.intra_op_num_threads = 3; so.inter_op_num_threads = 1
s = ort.InferenceSession(fp.SerializeToString(), so, providers=["CPUExecutionProvider"])
outs_names = [t for t in xs if t != "input"]
U = np.load(w2lib.R / "night" / "W0_data" / "calib_pool_u8.npy", mmap_mode="r")[OFF:OFF + N]
torch.set_num_threads(3)
t0 = time.time()
for i in range(N):
    x = U[i:i + 1].astype(np.float32) / np.float32(255)
    vals = dict(zip(outs_names, s.run(outs_names, {"input": x}))); vals["input"] = x
    for j in jobs:
        v = torch.from_numpy(vals[j["x"]].astype(np.float64))
        p = j["pads"]
        v = torch.nn.functional.pad(v, (p[1], p[3], p[0], p[2]))
        y = torch.nn.functional.conv2d(v, j["dW"], stride=j["stride"], dilation=j["dil"], groups=j["group"])
        j["sum"] += y.sum(dim=(0, 2, 3)).numpy(); j["cnt"] += y.shape[0] * y.shape[2] * y.shape[3]
    if (i + 1) % 50 == 0: print(f"{i + 1}/{N} {time.time() - t0:.0f}s", flush=True)
rep = []
for j in jobs:
    delta = j["sum"] / j["cnt"]
    t = init[j["bq"]]; b = nh.to_array(t).astype(np.int64)
    corr = np.rint(delta / j["bs"]).astype(np.int64)
    nb = b - corr
    assert np.all(np.abs(nb) < 2 ** 31 - 1)
    t.CopyFrom(nh.from_array(nb.astype(np.int32), j["bq"]))
    rep.append({"node": j["node"], "delta_absmax": float(np.abs(delta).max()), "delta_absmean": float(np.abs(delta).mean()),
                "corr_int_absmax": int(np.abs(corr).max()), "n_changed": int((corr != 0).sum()), "C": int(len(b))})
ok, zps = w2lib.a2_check(m); assert ok, zps
sha = w2lib.save(m, out)
json.dump({"out": out, "src": src, "scope": scope, "N": N, "offset": OFF, "sha256": sha, "a2_zps": zps, "convs": rep}, open(out + ".json", "w"), indent=1)
for r in rep[:10]: print(r)
print("done", len(rep), sha, time.time() - t0)
