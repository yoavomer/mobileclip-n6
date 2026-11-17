import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent)); import w2lib
sc = {r["tensor"]: r for r in json.load(open(w2lib.W2 / "results" / "scan_local_n200.json"))}
P0, P1, P2 = w2lib.STEM
fr = {t: sc[t]["free"] for t in w2lib.STEM}; be = {t: sc[t]["best"] for t in w2lib.STEM}
S = [{"name": "base", "ranges": {}},
     {"name": "free3", "ranges": {t: fr[t] for t in w2lib.STEM}},
     {"name": "best3", "ranges": {t: be[t] for t in w2lib.STEM}}]
# single tensor at free, others P99.95
for k, t in enumerate(w2lib.STEM):
    S.append({"name": f"free_pe{k}_only", "ranges": {t: fr[t]}})
# pe2 lo/hi grid (others free)
for lo in (-4.0, -2.0, -1.5):
    S.append({"name": f"pe2lo{lo}", "ranges": {**{t: fr[t] for t in w2lib.STEM}, P2: [lo, fr[P2][1]]}})
for hi in (3.5, 5.0):
    S.append({"name": f"pe2hi{hi}", "ranges": {**{t: fr[t] for t in w2lib.STEM}, P2: [fr[P2][0], hi]}})
for lo in (-4.0, -2.5, -1.2, -0.9):
    S.append({"name": f"pe0lo{lo}", "ranges": {**{t: fr[t] for t in w2lib.STEM}, P0: [lo, fr[P0][1]]}})
for lo in (-4.0, -1.5):
    S.append({"name": f"pe1lo{lo}", "ranges": {**{t: fr[t] for t in w2lib.STEM}, P1: [lo, fr[P1][1]]}})
json.dump(S, open(w2lib.W2 / "results" / "specs_r1.json", "w"), indent=1)
print(len(S)); print(json.dumps(S[1], indent=0))
