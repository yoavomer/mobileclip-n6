"""per-tensor fix-one jobs (bypass one board tensor) inside the given blocks. Board tensors = activation Q tensors except
the GELU-inner Erf/Add/Mul outputs (they do not exist on the board) and the 4 A2 operands.
usage: mk_jobs_tensors.py <out.json> <block1> [block2 ...] [--base path]"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent)); import w3lib as L
args = sys.argv[2:]; base = L.BASE
if "--base" in args:
    i = args.index("--base"); base = args[i + 1]; args = args[:i] + args[i + 2:]
m = L.load(base); B = L.blocks(m)
inner = {t for c in L.gelu_chains(m) for t in c["inner"]}
F = L.followers(m); fol = {x for v in F.values() for x in v}
jobs = []
for b in args:
    for t in B[b]:
        if t in inner or t in L.ATT or t in fol: continue      # followers are bypassed together with their leader
        jobs.append([f"t:{t}", "byp=" + "|".join([t] + F.get(t, []))])
json.dump(jobs, open(sys.argv[1], "w"), indent=0)
print(len(jobs)); [print(" ", j[0]) for j in jobs]
