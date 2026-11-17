"""List files in a MicroPython ROMFS image with the offset of their data (big-endian base-128 varuints, 0x80 = continue).
Record kinds (micropython vfs_rom.c): 1 padding, 2 verbatim data, 3 indirect data, 4 directory, 5 file.
usage: romfs_ls.py romfs.img [base_address]"""
import sys
d = open(sys.argv[1], "rb").read()
base = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0x70800000

def vu(p):
    v = 0
    while True:
        b = d[p]; p += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80: return v, p

def walk(p, end, path):
    while p < end:
        kind, p = vu(p); ln, p = vu(p); q = p + ln
        if kind == 4 or kind == 5:
            nl, r = vu(p); name = d[r:r + nl].decode(); r += nl
            if kind == 4:
                walk(r, q, path + name + "/")
            else:
                while r < q:
                    k2, r = vu(r); l2, r = vu(r)
                    if k2 == 2:
                        a = base + r
                        print(f"{path+name:34s} off=0x{r:08x} addr=0x{a:08x} size={l2:9d} mod8={a%8} mod64={a%64} mod4096={a%4096}")
                    r += l2
        p = q

assert d[:3] == b"\xd2\xcd\x31", d[:3]
ln, p = vu(3)
walk(p, p + ln, "")
