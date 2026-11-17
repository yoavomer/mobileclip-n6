"""FP32 statistics of every activation tensor of the P99.95 QDQ graph (same tensor names), on CALIB-A
(= calib_pool rows 0..N-1, default N=500; rows 500..999 are the screening set and are NOT used here).
FP32 graph = sens.edit(all activation QDQ bypassed, all weights FP32) = the 'allfloat' graph of quant/layers/sens.py
(reproduces FP32 to cosine 1.0000 [from report]). Pass 1: exact min/max per tensor. Pass 2: 8192-bin histogram on
[min, max]. Output: results/fp32_stats_calibA_n<N>.npz  (names, mins, maxs, hist[318, 8192], count, sum, sumsq)
usage: collect_stats.py [N=500] [threads=3]"""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, time, pathlib
sys.path.insert(0, os.environ.get("MCLIP_ROOT", ".") + r"\quant\layers")
import numpy as np, onnx
import sens  # read-only import of the earlier tool (edit / fp32_init)
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w3lib as L

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TH = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NB = 8192
m, actQ, wDQ = sens.load()
names = list(actQ)
mf = sens.edit(m, names, list(wDQ))
g = mf.graph
have = {o.name for o in g.output}
for t in names:
    if t not in have: g.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
s = L.session(mf.SerializeToString(), TH)
outs = [o.name for o in s.get_outputs()]
idx = [outs.index(t) for t in names]
U = np.load(L.W0 / "calib_pool_u8.npy", mmap_mode="r")[:N]
# sanity: embedding vs FP32 model
F = L.embed(L.FP32.read_bytes(), U[:4], TH)
mn = np.full(len(names), np.inf); mx = np.full(len(names), -np.inf)
t0 = time.time()
for i in range(N):
    r = s.run(None, {"input": U[i:i + 1].astype(np.float32) / np.float32(255)})
    if i < 4:
        e = r[outs.index("embedding" if "embedding" in outs else "embedding_QuantizeLinear_Input")].reshape(-1).astype(np.float64); e /= np.linalg.norm(e)
        print("check cos vs FP32", float(e @ F[i]), flush=True)
    for k, j in enumerate(idx):
        a = r[j]; mn[k] = min(mn[k], float(a.min())); mx[k] = max(mx[k], float(a.max()))
    if (i + 1) % 100 == 0: print(f"pass1 {i + 1}/{N} {time.time() - t0:.0f}s", flush=True)
H = np.zeros((len(names), NB), np.int64); S1 = np.zeros(len(names)); S2 = np.zeros(len(names)); CNT = np.zeros(len(names), np.int64)
for i in range(N):
    r = s.run(None, {"input": U[i:i + 1].astype(np.float32) / np.float32(255)})
    for k, j in enumerate(idx):
        a = r[j].ravel().astype(np.float64)
        w = (mx[k] - mn[k]) / NB if mx[k] > mn[k] else 1.0
        b = np.clip(((a - mn[k]) / w).astype(np.int64), 0, NB - 1)
        H[k] += np.bincount(b, minlength=NB); S1[k] += a.sum(); S2[k] += (a * a).sum(); CNT[k] += a.size
    if (i + 1) % 100 == 0: print(f"pass2 {i + 1}/{N} {time.time() - t0:.0f}s", flush=True)
np.savez(L.W3 / "results" / f"fp32_stats_calibA_n{N}.npz", names=np.array(names), mins=mn, maxs=mx, hist=H, count=CNT,
         sum=S1, sumsq=S2, nbins=np.array(NB))
print("done", time.time() - t0)
