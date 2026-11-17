import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent)); import w2lib
sc = {r["tensor"]: r for r in json.load(open(w2lib.W2 / "results" / "scan_local_n200.json"))}
stem_best = {t: sc[t]["best"] for t in w2lib.STEM}
others = [t for t in sc if t not in w2lib.STEM]
best27 = {t: sc[t]["best"] for t in others}
free27 = {t: (sc[t]["free"] if sc[t]["err_free"] <= sc[t]["err_cur"] else sc[t]["best"]) for t in others}
S = [{"name": "all30_best", "ranges": {**stem_best, **best27}},
     {"name": "all30_free", "ranges": {**stem_best, **free27}},
     {"name": "others27_best_only", "ranges": best27}]
json.dump(S, open(w2lib.W2 / "results" / "specs_r2.json", "w"), indent=1)
json.dump({"name": "best3", "ranges": stem_best}, open(w2lib.W2 / "results" / "spec_best3.json", "w"), indent=1)
print(len(S), len(best27))
