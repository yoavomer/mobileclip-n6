"""Build a candidate from P99.95 (or another source) by giving chosen activation tensors a new asymmetric int8 range.
python -I build_ranges.py <out.onnx> <spec.json> [src.onnx]   spec = {"tensor": [lo, hi], ...}
Only the scale / zero-point initializers of that tensor's QuantizeLinear and its DequantizeLinear(s) change
(new initializer names). Checks: A2 rule, onnx.checker, every other initializer byte-identical."""
import sys, json, pathlib
import numpy as np
from onnx import numpy_helper as nh
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import w2lib


def build(spec, out, src=w2lib.P9995, m=None):
    m = m if m is not None else w2lib.load(src)
    before = {t.name: t.SerializeToString() for t in m.graph.initializer}
    log = []
    for t, (lo, hi) in spec.items():
        s, z = w2lib.range_to_qp(lo, hi)
        old = w2lib.set_qparams(m, t, s, z)
        log.append({"tensor": t, "old_scale": old[0], "old_zp": old[1], "old_range": list(w2lib.qp_to_range(*old)),
                    "new_scale": s, "new_zp": z, "new_range": list(w2lib.qp_to_range(s, z)), "asked": [lo, hi]})
    after = {t.name: t.SerializeToString() for t in m.graph.initializer}
    changed = [k for k in after if k in before and after[k] != before[k]]
    assert not changed, changed
    ok, zps = w2lib.a2_check(m)
    assert ok, zps
    sha = w2lib.save(m, out)
    return {"out": str(out), "sha256": sha, "a2_zps": zps, "edits": log,
            "n_new_init": len([k for k in after if k not in before]), "n_dropped_init": len([k for k in before if k not in after])}


if __name__ == "__main__":
    out, spec = sys.argv[1], json.load(open(sys.argv[2]))
    spec = spec.get("ranges", spec) if "ranges" in spec else spec
    src = sys.argv[3] if len(sys.argv) > 3 else w2lib.P9995
    print(json.dumps(build(spec, out, src), indent=1))
