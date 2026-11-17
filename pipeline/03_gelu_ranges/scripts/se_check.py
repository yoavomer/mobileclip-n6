"""Byte-wise hyperRAM access check of a compile (W4's se_traffic.py parse/pattern imported read-only; output stays here).
usage: se_check.py <gws_neural_art_dir> <tag>  -> results/<tag>_se_summary.json + stdout"""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, collections, pathlib
sys.path.insert(0, os.environ.get("MCLIP_ROOT", ".") + r"\night\W4_speed")
import se_traffic as SE  # W4, read-only import (its main() is not called, so nothing is written into W4's folder)
d, tag = sys.argv[1], sys.argv[2]
recs = SE.parse(str(pathlib.Path(d) / "network_blob_trace.c"))
for r in recs: r.update(SE.pattern(r))
agg = collections.defaultdict(collections.Counter)
for r in recs:
    k = f"{r['pool']}_{'R' if r['dir'] == 0 else 'W'}"
    agg[k]["bytes"] += r["bytes"]; agg[k]["runs"] += r["runs"]; agg[k]["n"] += 1
    if r["run"] < SE.LINE: agg[k]["small_runs"] += r["runs"]; agg[k]["small_bytes"] += r["bytes"]
res = {"tag": tag, "n_configs": len(recs), "pools": {k: dict(v) for k, v in sorted(agg.items())}}
json.dump(res, open(pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"\night\W3_beyond\results") / f"{tag}_se_summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))
