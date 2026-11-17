"""Plan a ROMFS edit so that every model file is 8-byte aligned, and generate a board-side check script.

Model file = any file that is not *.txt / *.cascade / a pad file (pad files are named mci0_pad*).
Layout rule (verified on 2026-11-07 against every mci0_* file in 4 IDE images): the IDE keeps the existing order,
appends new files at the end, and each file record is  05 + 7-byte varuint + namelen + name + 02 + minimal varuint(size),
with the data offset padded to 4 bytes.  Files before the first deleted one keep their offsets.
The tool checks the rule on the existing image for the region it re-lays, and refuses if it does not hold.

usage:
  romfs_align.py plan  <base.img> --del name ... --add path ...      -> prints the plan (add order incl. pads) + writes
                                                                        board\\rom_check.py and evidence\\romfs_plan.json
  romfs_align.py check <new.img>                                     -> compares a committed image with romfs_plan.json
"""
import json, os, sys

BASE = 0x70800000
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLAN = os.path.join(ROOT, "evidence", "romfs_plan.json")
CHECK = os.path.join(ROOT, "board", "rom_check.py")
PAD_DIR = os.path.join(ROOT, "upload", "pads")


def parse(img):
    d = open(img, "rb").read()

    def vu(p):
        v = 0
        while True:
            b = d[p]; p += 1
            v = (v << 7) | (b & 0x7F)
            if not b & 0x80: return v, p

    assert d[:3] == b"\xd2\xcd\x31"
    ln, p = vu(3)
    out, end = [], p + ln
    while p < end:
        kind, p = vu(p); l, p = vu(p); q = p + l
        if kind == 5:
            nl, r = vu(p); name = d[r:r + nl].decode(); r += nl
            while r < q:
                k2, r = vu(r); l2, r = vu(r)
                if k2 == 2: out.append([name, r, l2])
                r += l2
        p = q
    return out


def vlen(n):
    k = 1
    while n >= 1 << (7 * k): k += 1
    return k


def place(cur, name, size):
    off = cur + 1 + 7 + 1 + len(name.encode()) + 1 + vlen(size)
    return off + (-off) % 4


def is_model(n):
    return not (n.endswith(".txt") or n.endswith(".cascade") or n.startswith("mci0_pad"))


def plan(img, dels, adds):
    files = parse(img)
    names = [f[0] for f in files]
    for n in dels:
        if n not in names: sys.exit(f"STOP: {n} is not in the image")
    k = min([names.index(n) for n in dels] + [len(files)])
    # rule check on the region that will be re-laid (files after the first deleted one, in the old image)
    for i in range(max(k, 1), len(files)):
        prev_end = files[i - 1][1] + files[i - 1][2]
        if place(prev_end, files[i][0], files[i][2]) != files[i][1]:
            sys.exit(f"STOP: layout rule does not reproduce {files[i][0]} in {img}; not safe to plan")
    keep = [f for f in files if f[0] not in dels]
    prefix = keep[:k]
    cur = prefix[-1][1] + prefix[-1][2]
    layout, problems = [(n, o, s, "kept") for n, o, s in prefix], []
    for n, _, s in keep[k:]:
        o = place(cur, n, s); layout.append((n, o, s, "kept, moves")); cur = o + s
        if is_model(n) and (BASE + o) % 8:
            problems.append(f"{n} (existing file) would move to 0x{BASE + o:08x} (mod8=4): needs delete + re-add")
    add_order, pad_i = [], 0
    for path in adds:
        n, s = os.path.basename(path), os.path.getsize(path)
        o = place(cur, n, s)
        if is_model(n) and (BASE + o) % 8:
            pad_i += 1
            pn = "mci0_pad" if pad_i == 1 else f"mci0_pad{pad_i}"
            for ps in (4, 8):
                po = place(cur, pn, ps)
                if (BASE + place(po + ps, n, s)) % 8 == 0: break
            os.makedirs(os.path.join(PAD_DIR, f"{pn}_{ps}"), exist_ok=True)
            pp = os.path.join(PAD_DIR, f"{pn}_{ps}", pn)
            open(pp, "wb").write(b"\0" * ps)
            layout.append((pn, po, ps, "added pad")); add_order.append(pp); cur = po + ps
            o = place(cur, n, s)
        layout.append((n, o, s, "added")); add_order.append(path); cur = o + s
    exp = {n: BASE + o for n, o, s, _ in layout if is_model(n)}
    for n, a in exp.items():
        if a % 8: problems.append(f"{n} at 0x{a:08x} is not 8-aligned")
    json.dump({"base_image": img, "delete": dels, "add_order": add_order,
               "expected": {n: hex(a) for n, a in exp.items()}, "total_bytes": cur}, open(PLAN, "w"), indent=1)
    write_check(exp)
    print("DELETE:", ", ".join(dels) or "-")
    print("ADD (in this order):"); [print("  ", p, os.path.getsize(p), "B") for p in add_order]
    print("LAYOUT (model files and pads, from the first changed record):")
    for n, o, s, why in layout[max(k - 1, 0):]:
        print(f"   {n:32s} 0x{BASE + o:08x} mod8={(BASE + o) % 8}  {s:9d} B  {why}")
    print(f"ROMFS data end: {cur} B ({cur / 1e6:.2f} MB)")
    print("PROBLEMS:" if problems else "ALL MODEL FILES 8-ALIGNED", *problems, sep="\n   ")
    return not problems


def write_check(exp):
    lines = ["# Generated by tools/romfs_align.py: actual vs expected flash address of every model file in /rom.",
             "import os, uctypes, time",
             "EXP = {" + ", ".join(f'"{n}": 0x{a:08x}' for n, a in sorted(exp.items())) + "}",
             "bad = []",
             "time.sleep_ms(1000)",
             "for n in sorted(os.listdir('/rom')):",
             "    if n.endswith('.txt') or n.endswith('.cascade') or n.startswith('mci0_pad'):",
             "        continue",
             "    f = open('/rom/' + n, 'rb'); a = uctypes.addressof(f) & 0xFFFFFFFF; f.close()",
             "    e = EXP.get(n, -1)",
             "    ok = (a == e) and (a % 8 == 0)",
             "    if not ok: bad.append(n)",
             "    print('R %-30s 0x%08x mod8=%d %s' % (n, a, a % 8, 'OK' if ok else 'BAD exp=0x%08x' % (e & 0xFFFFFFFF)))",
             "    time.sleep_ms(120)",
             "for n in EXP:",
             "    if n not in os.listdir('/rom'): bad.append(n + '(missing)')",
             "print('ROM CHECK: ' + ('ALL OK' if not bad else 'STOP ' + ' '.join(bad)))"]
    open(CHECK, "w", newline="\n").write("\n".join(lines) + "\n")


def check(img):
    p = json.load(open(PLAN))
    got = {n: BASE + o for n, o, s in parse(img) if is_model(n)}
    bad = [n for n, a in p["expected"].items() if got.get(n) != int(a, 16)] + [n for n, a in got.items() if a % 8]
    for n, a in sorted(got.items()):
        print(f"   {n:32s} 0x{a:08x} mod8={a % 8}  expected {p['expected'].get(n, '-')}")
    print("IMAGE CHECK:", "ALL OK" if not bad else "STOP " + " ".join(sorted(set(bad))))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "plan":
        img = a[1]; dels, adds, mode = [], [], None
        for x in a[2:]:
            if x == "--del": mode = dels
            elif x == "--add": mode = adds
            else: mode.append(x)
        sys.exit(0 if plan(img, dels, adds) else 1)
    elif a[0] == "check":
        check(a[1])
