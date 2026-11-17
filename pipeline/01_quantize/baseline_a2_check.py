# Baseline sanity check: mci0_256_a2 (A2) on the fixed input vs ORT running A2. WDT 30 s.
import time, machine, gc, uctypes, binascii, ml
def step(s):
    print(s); time.sleep_ms(120)
NAME = "mci0_256_a2"; EXPECT = 0x70f2b1b8; ZO = 6
f = open("/rom/" + NAME, "rb"); a = uctypes.addressof(f) & 0xFFFFFFFF; f.close()
step("A0 %s at 0x%08x mod8=%d %s" % (NAME, a, a % 8, "MATCH" if a == EXPECT else "MISMATCH"))
if a != EXPECT or a % 8:
    step("STOP: address check failed, no predict"); raise SystemExit
REF = {"default": binascii.a2b_base64("BPwDBAT+DgYDAf8HEAHyChkL/P8IAvj9BA0VCvwDBQcMCuIM/g4EBgUQAwoLCQwA/AcQDAcHCPcD+Q0TMe0A/gSaBgMSARYRCAERCRES/PUQCQb7DAoLBwsD/goABgQBCRMPDAIDAv8H/QP+/9f7/g8Q/Qr9DAQAAQkcAAX8AwQAMPcO9/4a/wMWBQwDFgcF/RECBwYDBBcRAwUN/QMXBwYMDwoJFgYOC/n9AgD7ARQcBgYGDAgbEf4EAxLtGv8BA/sIDxX+cA8LDwIT/hAHCQAD+v/6DgYCAv4FAA8DAgToByoSHf4NBAofEQoECwcCCAj/9QcNBQwKAAkABA7+AxIFCgP8CggLCgX9DAsJEfATFAUGAqkFC/sS/AAOCPkTBxMI/gcKCREBCAED8/YEARQfBgEGDw0B/AEFEgbwDf38+AYPGA8W9wUUC/wP/AL9//0DCQ0O/g8C8hALCf36Buv8ABAK9/P8AxIQAhQFBwsGAwMBEhMDAgMJBAQWGvURB/8JCPYDCwcHBP8G/QkKAgf1DQcDAvb1DwoMBwMH+P4J+w4ACgT9BQUACQD7/AIEAyDtBP4NBgIBCwsLFwUMDQIQCQf9Ef8D/AMRA+sECgYI8v0CAQMHABAA+gT0BgQDBhIQCP0RGwIDCQEPCwgA6/oGABgRCwMJBA4CCwsEDgc="), "opt_off": binascii.a2b_base64("CPz9CQQFDAsEAAEFEgPwChwH/v4LAvb/BwsbDPwEBgQPCuISAAsGBwQIAAcKDAr9+gsMEQgHEfUJ+hASMu4F+/+bCQQVBRkSCP8SBwwQ+foNCAX6DQwKBwgCAA0CBQMECQ8RCQYBBP0E+Qj/ANn6/QwP/Af6DQD9AQch/QIFAgX7MfgS9/4U/gERBhECGAYC/RECCwQFBxUM/QQP/wIYCQIREAwFEwcLDPf7/AH+BRQeBgIIDAYeDwECAg/tGf4BAvwKExX9aA0MDwMTAAsFDv0B+fz6DAUIDAUCAg0I/gDkBy8LIQEJBgsdEw4GDwcDBQz98ggLAw4JAAoDBA8A/xMDBwT2EQYICAb8DQoPFPQRFAUDAqYACPsTAQEMCgIOAhUG/AsJChH/CgMD8/YFAhInCAEEDwwGAf4DEAfwC/35/AUPExMW8wQXCfsL+wL8/PwDDQoK+w4C7wwHCgX3COr8BA4H/O/9ARYLABADDA0KCf4AEhIDAfsJCAEUGfMRAgAFCPcBDAwHBAMG/gYHBgfxDwgDBPb5DA0LC/8G8v0I/RP+DQD7CQICCQT4/AAIBiDrBQMOBwP/CQwKFQIJDAcLCQX9Df3+9gQM/OwFCggE8P4D//8GAhb++wfxBQMKChQOCvwQHwYFBwELDQj+6/sD/xkTBwQICQ0FCAYADAc=")}
gc.collect(); m = ml.Model("/rom/" + NAME); step("A1 %s" % (m,))
PAT = bytes(((((37 * k + 11) % 256) - 128) & 0xFF) for k in range(256))
def s8(b): return [v - 256 if v > 127 else v for v in b]
def std(v):
    mu = sum(v) / len(v); return (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5
def cosz(p, q):
    n = a2 = b2 = 0
    for i in range(len(p)):
        x = p[i] - ZO; y = q[i] - ZO; n += x * y; a2 += x * x; b2 += y * y
    return n / ((a2 * b2) ** 0.5)
def fill(buf, shape, dtype): buf[:] = PAT * (len(buf) // 256)
got = []
def grab(model, inputs, outputs): got.append(bytes(outputs[0].tobytes())); return None
def nop_fill(buf, shape, dtype): pass
def nop_cb(model, inputs, outputs): return None
wdt = machine.WDT(timeout=30000); step("B0 WDT on (30 s)")
t0 = time.ticks_us(); m.predict([fill], callback=grab); dt = time.ticks_diff(time.ticks_us(), t0); wdt.feed()
g = s8(got[0])
step("C fixed t=%d us crc32=0x%08x first16=%s" % (dt, binascii.crc32(got[0]) & 0xFFFFFFFF, g[:16]))
for md in ("default", "opt_off"):
    r = s8(REF[md])
    step("D fixed vs ORT-A2-%s: cos=%.6f maxdiff=%d std board=%.3f ORT=%.3f ratio=%.3f sat board=%d/%d" % (md, cosz(g, r), max(abs(g[i] - r[i]) for i in range(512)), std(g), std(r), std(g) / std(r), g.count(-128), g.count(127)))
for i in range(3): m.predict([nop_fill], callback=nop_cb); wdt.feed()
T = []
for i in range(20):
    gc.collect(); wdt.feed(); t0 = time.ticks_us(); m.predict([nop_fill], callback=nop_cb); T.append(time.ticks_diff(time.ticks_us(), t0))
wdt.feed(); T.sort()
step("T clean predict 20 runs: median %d us (min %d max %d)" % ((T[9] + T[10]) // 2, T[0], T[19]))
step("ALL DONE (WDT resets the board in ~30 s, expected)")
