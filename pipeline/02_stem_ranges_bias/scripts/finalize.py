"""Copy the chosen best candidate into night/W2_stem/best/ (ONNX + compiled network_rel.bin + c_info.json) and write
best/SHA256SUMS.txt.  usage: python -I finalize.py <candidate_name>"""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys, shutil, pathlib, hashlib, json
W2 = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"\night\W2_stem")
name = sys.argv[1]
B = W2 / "best"; B.mkdir(exist_ok=True)
src = {f"{name}.onnx": W2 / "models" / f"{name}.onnx",
       f"{name}_network_rel.bin": W2 / "compile" / name / "gout" / "network_rel.bin",
       f"{name}_c_info.json": W2 / "compile" / name / "gout" / "c_info.json",
       f"{name}_network_generate_report.txt": W2 / "compile" / name / "gout" / "network_generate_report.txt"}
lines = []
for dst, s in src.items():
    shutil.copyfile(s, B / dst)
    h = hashlib.sha256((B / dst).read_bytes()).hexdigest()
    lines.append(f"{h}  {dst}  ({(B / dst).stat().st_size:,} B; copy of {s})")
(B / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
