"""Summary of one compile made by night/tools/compile.sh (copy of W2's compile_report.py method + the type-level mapping
check night/final/maptype.py): exit code, leaf epochs HW / SW / hybrid / no-op, SW MACs (= 'epoch OPS'/2 of the SW epochs
in gws network.csv), hyperRAM / memory lines of the report, rel size + SHA256, maptype vs P99.95 (no op type may move to
NODE_SW / NODE_SW_HW, SW+hybrid count must stay 82), bisect/compare_mapping.py vs P99.95 (op names may not align when
the graph changes).
usage: python -I compile_report.py <cand_out_base>  -> <cand_out_base>/summary.json (+ printed)"""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, json, re, csv, subprocess, hashlib, pathlib, collections
R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
PY = str(R / ".venv" / "Scripts" / "python.exe")
REF = R / "quant/layers/gws_p9995/neural_art__network/c_info.json"


def leaves(ci):
    d = json.load(open(ci)); out = []
    for ep in d["graphs"][0]["nodes"]:
        sub = ep.get("subgraph_nodes", []) if ep.get("mapping") == "NODE_EC" else [ep]
        for e in sub:
            names = [s["name"] for s in e.get("subgraph_nodes", []) if s.get("type") != "Param"]
            if names: out.append((e["name"], e.get("mapping"), names))
    return out


def summary(base):
    base = pathlib.Path(base); s = {"dir": str(base)}
    ex = base / "compile_exit.txt"
    s["exit"] = int(ex.read_text().strip()) if ex.exists() else None
    ci = base / "gout" / "c_info.json"
    if not ci.exists():
        log = (base / "compile.log").read_text(errors="replace") if (base / "compile.log").exists() else ""
        s["error"] = [l.strip()[-200:] for l in log.splitlines() if "Assertion" in l or "E10" in l or "Error" in l][-4:]
        return s
    Lv = leaves(ci); c = collections.Counter(m for _, m, _ in Lv)
    s["epochs"] = {"HW": c.get("NODE_HW", 0), "SW": c.get("NODE_SW", 0), "hybrid": c.get("NODE_SW_HW", 0), "noop": c.get("NODE_NO_X", 0)}
    csvp = base / "gws" / "neural_art__network" / "network.csv"
    if not csvp.exists(): csvp = base / "network.csv"
    if csvp.exists():
        rows = {int(r["epochs"].strip()): r for r in csv.DictReader(open(csvp)) if r["epochs"].strip().isdigit()}
        sw = [int(re.search(r"\d+", n).group()) for n, m, _ in Lv if m == "NODE_SW" and re.search(r"\d+", n)]
        s["sw_macs"] = int(round(sum(float(rows[e]["epoch OPS"]) / 2 for e in sw if e in rows)))
    rep = base / "gout" / "network_generate_report.txt"
    if rep.exists():
        t = rep.read_text(errors="replace")
        m = re.search(r"hyperRAM\s+\[[^\]]+\]:\s+([\d.]+ \w?B)", t); s["hyperRAM"] = m.group(1) if m else None
        m = re.search(r"Total:\s+([\d.]+ MB).*?weights:\s+([\d.]+ \w?B).*?activations:\s+([\d.]+ \w?B)", t)
        if m: s["total"], s["weights"], s["activations"] = m.groups()
    rel = base / "gout" / "network_rel.bin"
    if rel.exists():
        b = rel.read_bytes(); s["rel_bytes"] = len(b); s["rel_sha256"] = hashlib.sha256(b).hexdigest()
    o = subprocess.run([PY, "-I", str(R / "night" / "final" / "maptype.py"), str(ci), str(REF)], capture_output=True, text=True)
    s["maptype_vs_p9995"] = o.stdout.strip().splitlines()
    sw = [l for l in s["maptype_vs_p9995"] if l.startswith("SW+hybrid")]
    moved = [l for l in s["maptype_vs_p9995"] if not l.startswith("SW+hybrid") and ("NODE_SW" in l)]
    s["maptype_ok"] = bool(sw) and sw[0].split()[-3] == sw[0].split()[-1] == "82" and not moved
    o = subprocess.run([PY, "-I", str(R / "bisect" / "compare_mapping.py"), str(ci), str(REF)], capture_output=True, text=True)
    txt = o.stdout.strip().splitlines()
    s["compare_mapping_vs_p9995"] = [l for l in txt if "same mapping" in l or "different" in l][-3:] or txt[-3:]
    return s


if __name__ == "__main__":
    B = pathlib.Path(sys.argv[1]); s = summary(B)
    json.dump(s, open(B / "summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))
