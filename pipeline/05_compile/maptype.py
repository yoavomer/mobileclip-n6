"""Type-level mapping comparison of two compiles: count of (op type, mapping) over leaf epochs (c_info.json).
usage: maptype.py <cand c_info> <ref c_info>"""
import json, sys, collections
def load(p):
    d = json.load(open(p)); c = collections.Counter()
    for ep in d["graphs"][0]["nodes"]:
        sub = ep.get("subgraph_nodes", []) if ep.get("mapping") == "NODE_EC" else [ep]
        for e in sub:
            for s in e.get("subgraph_nodes", []):
                if s.get("type") not in ("Param", ""): c[(s["type"], e.get("mapping"))] += 1
    return c
a, b = load(sys.argv[1]), load(sys.argv[2])
for k in sorted(set(a) | set(b)):
    if a[k] != b[k]: print(f"  {k[0]:20s} {k[1]:12s} cand {a[k]:4d} ref {b[k]:4d}")
sw = lambda c: sum(v for (t, m), v in c.items() if m in ("NODE_SW", "NODE_SW_HW"))
print("SW+hybrid op count cand", sw(a), "ref", sw(b))
