"""PC-side checks around one ROMFS model swap in OpenMV IDE (written overnight 8.11, NOT run yet).
The IDE's ROMFS editor reads the board into %TEMP%\\openmv-romfs-staging\\romfs0.img and writes the committed image there.
Run from the working folder (MCLIP_ROOT) with python -I.

  swap_check.py pre  <name in ROMFS to delete> <disk copy>
      before deleting: the file's bytes inside the staged image == the disk copy (SHA256): the rule "delete a ROMFS file
      only if a byte-identical copy exists on disk". Prints FAILED -> STOP (do not delete) otherwise.
  swap_check.py post <added name> <disk copy> <expected address hex> [expected image SHA256]
      after Commit: the added file's bytes inside the staged image == the disk copy, its address == the planned one
      (night/final/board/plan_preview.json), 8-aligned; optional: the whole staged image SHA256 (baseline restore:
      eeeb3cabf8c9fd30ac38fd2d89763edaa4563bf2af2dfd0ab199a61a53d1e34f).
Every call appends to night/final/board/swap_log.json."""
import sys, os, json, hashlib, importlib.util, pathlib, time
R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"")
STAGE = pathlib.Path(os.environ["TEMP"]) / "openmv-romfs-staging" / "romfs0.img"
LOG = R / "night" / "final" / "board" / "swap_log.json"
spec = importlib.util.spec_from_file_location("romfs_align_ro", R / "tools" / "romfs_align.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
sha = lambda b: hashlib.sha256(b).hexdigest()


def log(step, ok, info):
    d = json.load(open(LOG)) if LOG.exists() else []
    d.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step, "ok": ok, "info": info})
    json.dump(d, open(LOG, "w"), indent=1)
    print(step, "OK" if ok else "FAILED -> STOP, do not delete / do not commit / no predict", json.dumps(info, indent=1))
    if not ok: sys.exit(1)


def files():
    img = STAGE.read_bytes()
    return img, {n: (o, s) for n, o, s in ra.parse(str(STAGE))}


def pre(name, copy):
    img, F = files(); info = {"staged": str(STAGE), "files": sorted(F)}
    want = sha(pathlib.Path(copy).read_bytes()); info["disk_copy"] = copy; info["disk_sha"] = want
    if name not in F: log("pre", False, {**info, "error": f"{name} not in the staged image"})
    o, s = F[name]; info["in_romfs_sha"] = sha(img[o:o + s]); info["addr"] = hex(ra.BASE + o)
    log("pre " + name, info["in_romfs_sha"] == want, info)


def post(name, copy, addr, img_sha=None):
    img, F = files(); info = {"staged": str(STAGE), "files": sorted(F), "staged_sha": sha(img), "staged_len": len(img)}
    want = sha(pathlib.Path(copy).read_bytes()); info["disk_sha"] = want
    if name not in F: log("post", False, {**info, "error": f"{name} not in the staged image"})
    o, s = F[name]; a = ra.BASE + o
    info.update(in_romfs_sha=sha(img[o:o + s]), addr=hex(a), expected_addr=addr, mod8=a % 8)
    misaligned = [n for n, (o2, _) in F.items() if ra.is_model(n) and (ra.BASE + o2) % 8]
    info["misaligned_model_files"] = misaligned
    ok = info["in_romfs_sha"] == want and a == int(addr, 16) and a % 8 == 0 and not misaligned
    if img_sha: info["expected_image_sha"] = img_sha; ok = ok and info["staged_sha"] == img_sha
    log("post " + name, ok, info)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["pre"] and len(a) == 3: pre(a[1], a[2])
    elif a[:1] == ["post"] and len(a) in (4, 5): post(*a[1:])
    else: print(__doc__)
