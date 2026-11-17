"""Pick, for each of the 27 non-stem GELU inputs, the probed range with an exact compiled table and the lowest local error
(if no probed range is exact: the lowest compiled-table error). Stem = S1t. -> results/spec_S2t.json + printed table."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib

ev_ = []
for p in ("a", "b"):
    ev_ += [r for r in json.load(open(w2lib.W2 / "results" / f"probe2{p}_read_eval.json")) if not r["label"].startswith("val")]
s1t = json.load(open(w2lib.W2 / "results" / "spec_S1t.json"))["ranges"]
by = {}
for r in ev_: by.setdefault(r["tensor"], []).append(r)
spec = dict(s1t); rows = []
for t, lst in by.items():
    ex = [r for r in lst if r["n_diff"] == 0]
    pick = min(ex, key=lambda r: r["err_ideal"]) if ex else min(lst, key=lambda r: r["err_compiled"])
    lab = pick["label"]; lo = float(lab.split("_lo")[1].split("_hi")[0]); hi = float(lab.split("_hi")[1])
    # rebuild exactly the probed (scale, zp): range_to_qp(lo, hi) with the 3-decimal label values must equal the probe params
    s, z = w2lib.range_to_qp(lo, hi)
    if abs(s - pick["params"][0]) > 1e-12 * max(1, s) or z != pick["params"][1]:
        lo, hi = pick["range"]                                       # fall back to the exact representable range
        s, z = w2lib.range_to_qp(lo, hi)
    spec[t] = [lo, hi]
    rows.append((t, len(ex), len(lst), pick["n_diff"], pick["err_ideal"], pick["err_compiled"], (s, z), tuple(pick["params"][:2])))
json.dump({"name": "S2t_all30_tableaware", "ranges": spec}, open(w2lib.W2 / "results" / "spec_S2t.json", "w"), indent=1)
for r in rows:
    print(f"{r[0][7:60]:53s} exact {r[1]}/{r[2]}  pick diff {r[3]}  err {r[4]:.3e}  qp {r[6][0]:.6g}/{r[6][1]} probe {r[7][0]:.6g}/{r[7][1]}")
print(len(spec), "tensors")
