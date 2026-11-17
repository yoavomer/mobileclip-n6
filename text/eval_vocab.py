"""W5 task 5: demo-vocabulary accuracy on the PC (closed set = the 41 demo entries, pc/vocab.py).

Images: ImageNetV2 (matched-frequency) photos whose ImageNet label is mapped to a demo entry (vocab.VOCAB):
  set "CU"  = CALIB_POOL + unused (4 per class)  <- main set (as the task asks)
  set "SEL" = SELECTION (3 per class)            <- extra, reported separately and pooled ("ALL")
  TEST is never read.
Pipelines: ref (MobileCLIP transform) and cam_QVGA (camera model + exact OpenMV board preprocessing, pc/omv_sim.py).
Models: fp32, a2, a2_bl, p9995, p9995_bl (+ extra ONNX paths given on the command line; board-like added as <path>_bl).
Text: templates80 / simple, FP32 and the board's int8 rows (board_files format; quantized here the same way).
Metrics: top-1 / top-3 (micro, Wilson 95 % CI), macro top-1 over entries, close-pair confusions, paired
templates80-vs-simple difference (bootstrap CI, discordant counts).
usage: eval_vocab.py [extra_model.onnx ...]"""
import sys, json, time, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common, vocab, make_board_files  # noqa: E402

extra = sys.argv[1:]
m2v = vocab.imagenet_to_vocab(); D = vocab.display_names(); NV = len(D)


def subset(name):
    out = [(rel, l) for rel, l in common.read_list(name) if l in m2v]
    return out


sets = {"CU": subset("calib_pool") + subset("unused"), "SEL": subset("selection")}
TXT = {}
for mode in ("templates80", "simple"):
    T = np.load(common.RES / f"text_vocab_{mode}.npy").astype(np.float64)
    TXT[mode] = T
    s, q = make_board_files.quantize_rows(T)
    Tq = q.astype(np.float64); TXT[mode + "_int8"] = Tq / np.linalg.norm(Tq, axis=1, keepdims=True)
MODELS = ["fp32", "a2", "a2_bl", "p9995", "p9995_bl"] + [x for p in extra for x in (p, p + "_bl")]
PIPES = ["ref", "cam_QVGA"]
B = 10000; rng = np.random.default_rng(0)


def metrics(E, y, T):
    L = E.astype(np.float64) @ T.T
    order = np.argsort(-L, 1)
    c1 = order[:, 0] == y; c3 = (order[:, :3] == y[:, None]).any(1)
    n = len(y); k1, k3 = int(c1.sum()), int(c3.sum())
    per = {D[v]: [int(c1[y == v].sum()), int((y == v).sum())] for v in range(NV) if (y == v).any()}
    macro = float(np.mean([a / b for a, b in per.values()]))
    pairs = {}
    for a, b in vocab.PAIRS:
        ia, ib = D.index(a), D.index(b)
        pairs[f"{a}->{b}"] = [int(((y == ia) & (order[:, 0] == ib)).sum()), int((y == ia).sum())]
        pairs[f"{b}->{a}"] = [int(((y == ib) & (order[:, 0] == ia)).sum()), int((y == ib).sum())]
    return {"n": n, "top1": round(k1 / n, 4), "top1_ci95": common.wilson(k1, n), "top3": round(k3 / n, 4),
            "top3_ci95": common.wilson(k3, n), "macro_top1": round(macro, 4), "pair_confusions": pairs,
            "per_entry_top1": per}, c1, order[:, 0]


def paired(a, b):
    d = a.astype(float) - b.astype(float); n = len(d)
    bs = d[rng.integers(0, n, (B, n))].mean(1)
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "a_only": int(((a == 1) & (b == 0)).sum()), "b_only": int(((a == 0) & (b == 1)).sum())}


t0 = time.time()
res = {"sets": {k: len(v) for k, v in sets.items()}, "vocab_entries": NV, "imagenet_classes_mapped": len(m2v),
       "models": {}, "a2_rule": {}, "notes": "closed-set top-k over the 41 demo entries; label = vocab entry of the ImageNet class"}
emb = {}
for sname, items in sets.items():
    y = np.array([m2v[l] for _, l in items])
    for p in PIPES:
        U = common.planes_cached(p, items, f"vocab{sname}")
        for m in MODELS:
            emb[(sname, p, m)] = common.embed_u8(m, U, cache_tag=f"{p}_vocab{sname}")
            print("emb", sname, p, m, f"{time.time() - t0:.0f}s", flush=True)
for m in MODELS:
    res["a2_rule"][m] = common.a2_check(m)
    res["models"][m] = {"info": common.session(m)[1]}
yall = np.concatenate([np.array([m2v[l] for _, l in sets[s]]) for s in ("CU", "SEL")])
for m in MODELS:
    for p in PIPES:
        for sname in ("CU", "SEL", "ALL"):
            if sname == "ALL":
                E = np.concatenate([emb[("CU", p, m)], emb[("SEL", p, m)]]); y = yall
            else:
                E = emb[(sname, p, m)]; y = np.array([m2v[l] for _, l in sets[sname]])
            row = {}; c = {}; pr = {}
            for t in TXT:
                row[t], c[t], pr[t] = metrics(E, y, TXT[t])
            row["paired_templates80_minus_simple_top1"] = paired(c["templates80"], c["simple"])
            row["paired_int8_minus_fp32_text_templates80"] = paired(c["templates80_int8"], c["templates80"])
            row["top1_same_pred_int8_vs_fp32_text_templates80"] = round(float((pr["templates80_int8"] == pr["templates80"]).mean()), 4)
            if m != "fp32":
                Ef = (np.concatenate([emb[("CU", p, "fp32")], emb[("SEL", p, "fp32")]]) if sname == "ALL" else emb[(sname, p, "fp32")])
                row["cos_vs_fp32_mean"] = round(float((E * Ef).sum(1).mean()), 4)
                _, cf, prf = metrics(Ef, y, TXT["templates80"])
                row["top1_agree_with_fp32_templates80"] = round(float((pr["templates80"] == prf).mean()), 4)
            res["models"][m][f"{p}/{sname}"] = row
            print(m, p, sname, {t: (row[t]["top1"], row[t]["top3"]) for t in TXT}, row["paired_templates80_minus_simple_top1"], flush=True)
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(common.RES / "vocab_eval.json", "w"), indent=1, default=str)
print("done", res["seconds"])
