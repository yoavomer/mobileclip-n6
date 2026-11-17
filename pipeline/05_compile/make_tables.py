"""Write the compiled GELU/Sigmoid tables of one compile as JSON (W2's bl_ct.tables_for, read-only use).
usage: python -I make_tables.py <cand.onnx> <gws dir> <out.tables.json>"""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, pathlib
sys.path.insert(0, os.environ.get("MCLIP_ROOT", ".") + r"\night\W2_stem\scripts")
import bl_ct
T, rep = bl_ct.tables_for(sys.argv[1], sys.argv[2])
json.dump(T, open(sys.argv[3], "w"))
print(json.dumps({"out": sys.argv[3], "tables": len(T["gelu"]) + len(T["sigmoid"]),
                  "total_codes_differing_from_ideal": sum(x["codes_differing_from_ideal"] for x in rep)}))
