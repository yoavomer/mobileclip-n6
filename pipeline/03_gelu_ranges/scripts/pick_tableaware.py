"""Pick, per chain, the probed option with an exact compiled GELU table (0 codes different from ideal) and the lowest
chain-local error; fall back to the fewest differing codes. Checks the 2 validation branches of each probe (expected
1 and 10 differing codes, as in the P99.95 full compile). Writes results/spec_cl2t_27.txt and results/pick_tableaware.json."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import json, pathlib
R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"\night\W3_beyond\results")
opts = json.load(open(R / "gprobe_options.json"))
rd = []
for x in "ab":
    r = json.load(open(R / f"gprobe_{x}_read.json"))
    assert [v["n_diff"] for v in r[:2]] == [1, 10], [v["n_diff"] for v in r[:2]]
    rd += r[2:]
flat = [(o, k) for o in opts for k in range(len(o["options"]))]
assert len(flat) == len(rd)
for (o, k), r in zip(flat, rd):
    assert all(abs(a - b) <= 1e-12 * max(1, abs(a)) for a, b in zip(r["params"], o["options"][k]["params"]))
    o["options"][k]["n_diff"] = r["n_diff"]; o["options"][k]["rows"] = r["rows"]
parts, pick = [], []
for o in opts:
    ex = [p for p in o["options"] if p["n_diff"] == 0]
    best = min(ex, key=lambda p: p["local_err"]) if ex else min(o["options"], key=lambda p: (p["n_diff"], p["local_err"]))
    s_i, z_i, s_o, z_o = best["params"]
    parts += [f"{o['output']}@{s_o!r}@{z_o}", f"{o['input']}@{s_i!r}@{z_i}"]
    cl2 = o["options"][0]
    pick.append({"chain": o["base"], "cl2_n_diff": cl2["n_diff"], "cl2_err": cl2["local_err"], "picked": {k: best[k] for k in ("fh", "dlo", "params", "local_err", "n_diff", "rows")},
                 "n_exact_options": len(ex), "cur_err": o["cur_err"]})
(R / "spec_cl2t_27.txt").write_text("q=" + "|".join(parts))
json.dump(pick, open(R / "pick_tableaware.json", "w"), indent=1)
for p in pick:
    print(f"{p['chain'].split('/model/')[-1][:42]:42s} cl2 diff {p['cl2_n_diff']:2d} | picked fh {p['picked']['fh']} dlo {p['picked']['dlo']:+.1f} diff {p['picked']['n_diff']} err {p['picked']['local_err']:.3g} (cl2 {p['cl2_err']:.3g}, P99.95-top {p['cur_err']:.3g}) exact options {p['n_exact_options']}")
