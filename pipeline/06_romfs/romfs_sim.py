"""Simulate the IDE ROMFS layout after a known file: record = 05 + 7-byte padded varuint + namelen + name + 02 + data-length
varuint (minimal), with extra 0x80 pad bytes so that the data offset is 4-aligned. Verified against today's romfs0.img."""
import sys
def vlen(n):
    k = 1
    while n >= 1 << (7 * k): k += 1
    return k
def layout(start, files):
    cur, out = start, []
    for name, size in files:
        off = cur + 1 + 7 + 1 + len(name) + 1 + vlen(size)
        off += (-off) % 4
        out.append((name, off, size)); cur = off + size
    return out
if __name__ == "__main__":
    partA_end = 0x009396e0 + 6342776
    today = [("mci0_K512c",270832),("mci0_K512s",281992),("mci0_K768c",410352),("mci0_K768s",421448),("mci0_K1024c",545520),("mci0_K1024s",556640),
             ("mci0_K1280c",680688),("mci0_K1280s",691808),("mci0_K1536c",815856),("mci0_K1536s",826976),("mci0_K1536c_16",794376),("mci0_K1536s_16",805568)]
    for n, o, s in layout(partA_end, today): print("today-sim", n, hex(o))
