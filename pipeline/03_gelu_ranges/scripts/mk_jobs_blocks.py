import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent)); import w3lib as L
m = L.load(); B = L.blocks(m)
jobs = [["base", "base"]]
for b, ts in B.items():
    jobs.append([f"blk:{b}", f"bypblock={b}"])
    print(b, len(ts), len([t for t in ts if t not in L.ATT]))
allbut = "|".join(t for t in L.Info(m).actQ if t not in L.ATT)
jobs.append(["allfloat_act", "byp=" + allbut])
json.dump(jobs, open(L.W3 / "results" / "jobs_blocks.json", "w"), indent=0)
print(len(jobs))
