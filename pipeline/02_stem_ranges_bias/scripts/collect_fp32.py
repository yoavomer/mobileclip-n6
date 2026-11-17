"""FP32 statistics of the 30 GELU chain inputs on calib-pool images (FP32 = mci0_256_fp32_pre.onnx, ORT, 3 threads).
-> results/fp32_gelu_hist_n<N>.npz : per tensor fine histogram (65,536 bins on [-R, R], R = 3 x the P99.95 |range|),
   exact min / max, count, per-channel min / max / mean;  plus results/fp32_stem_sample_n<N>.npz: a random 1/16 sample
   of the 3 stem conv outputs (float32) for the local error scans.
usage: python -I collect_fp32.py [N=200]"""
import sys, json, time, pathlib
import numpy as np, onnx, onnxruntime as ort
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
chains = w2lib.gelu_chains(w2lib.load())
names = [c["x"] for c in chains]
m = onnx.load(str(w2lib.FP32_PRE))
have = {o for n in m.graph.node for o in n.output}
missing = [t for t in names if t not in have]
assert not missing, missing
for t in names:
    m.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
so = ort.SessionOptions(); so.intra_op_num_threads = 3; so.inter_op_num_threads = 1
s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
U = np.load(w2lib.R / "night" / "W0_data" / "calib_pool_u8.npy", mmap_mode="r")[:N]
NB = 65536
Rng = {c["x"]: 3.0 * max(abs(w2lib.qp_to_range(c["s_in"], c["z_in"])[0]), abs(w2lib.qp_to_range(c["s_in"], c["z_in"])[1])) for c in chains}
H = {t: np.zeros(NB, np.int64) for t in names}
mn = {t: np.inf for t in names}; mx = {t: -np.inf for t in names}
cmin = {}; cmax = {}; csum = {}; ccnt = {}
rng = np.random.default_rng(0)
samp = {t: [] for t in w2lib.STEM}
t0 = time.time()
for i in range(N):
    x = U[i:i + 1].astype(np.float32) / np.float32(255)
    outs = s.run(names, {"input": x})
    for t, v in zip(names, outs):
        v = v[0]  # C,H,W
        r = Rng[t]
        idx = np.clip(((v.ravel() + r) / (2 * r) * NB).astype(np.int64), 0, NB - 1)
        H[t] += np.bincount(idx, minlength=NB)
        mn[t] = min(mn[t], float(v.min())); mx[t] = max(mx[t], float(v.max()))
        vc = v.reshape(v.shape[0], -1)
        cmin[t] = np.minimum(cmin.get(t, np.full(vc.shape[0], np.inf)), vc.min(1))
        cmax[t] = np.maximum(cmax.get(t, np.full(vc.shape[0], -np.inf)), vc.max(1))
        csum[t] = csum.get(t, 0) + vc.astype(np.float64).sum(1); ccnt[t] = ccnt.get(t, 0) + vc.shape[1]
        if t in samp:
            f = v.ravel(); samp[t].append(f[rng.integers(0, f.size, f.size // 16)])
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{N} {time.time() - t0:.0f}s", flush=True)
out = w2lib.W2 / "results"
np.savez(out / f"fp32_gelu_hist_n{N}.npz", names=np.array(names), R=np.array([Rng[t] for t in names]),
         H=np.stack([H[t] for t in names]), min=np.array([mn[t] for t in names]), max=np.array([mx[t] for t in names]))
np.savez(out / f"fp32_gelu_chanstats_n{N}.npz", names=np.array(names),
         **{f"cmin{k}": cmin[t] for k, t in enumerate(names)}, **{f"cmax{k}": cmax[t] for k, t in enumerate(names)},
         **{f"cmean{k}": csum[t] / ccnt[t] for k, t in enumerate(names)})
np.savez(out / f"fp32_stem_sample_n{N}.npz", **{f"pe{k}": np.concatenate(samp[t]) for k, t in enumerate(w2lib.STEM)})
print("done", time.time() - t0)
