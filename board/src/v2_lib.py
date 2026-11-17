# ---------------------------------------------------------------- v2 library (overnight 2, 10.11)
# What is new compared with the known-good demo (the model, the camera path and the input bytes are NOT changed:
# set_source / draw_planes / fill / predict / grab above are the known-good core, byte for byte):
#   1. scoring with ulab (C, vectorised) instead of the pure-Python loop: same cosine, ~1 ms instead of ~22 ms
#   2. smoothing: exponential moving average (EMA) of the unit image embedding, m = (1 - ALPHA) * m + ALPHA * e
#   3. "?" when unsure: the #1 word is shown only if (cos #1 - cos #2) >= THR_ON; once shown it stays while it is
#      still #1 and its margin is >= THR_OFF (hysteresis, so the label does not flicker at the threshold)
#   4. the per-frame line is printed without the 30 ms pause (one line per ~180 ms frame is already paced)
# ulab calls used here were all proven on this board in run A (release/logs/p2_cap_emb_dark.txt, U lines).
from ulab import numpy as np

ALPHA = 0.2                      # EMA weight of the newest frame (1.0 = no smoothing)
THR_ON = 0.012                   # show a word when its cosine margin over #2 is at least this
THR_OFF = 0.006                  # keep the shown word while it is #1 and its margin is at least this
MAX_FRAMES = 0                   # 0 = run until stopped
PROFILE = 0                      # 1 = also time the resize inside fill() (adds a wrapper call)
V2 = {"Tf": None, "ema": None, "label": -1}
CNT = [0, 0, 0, 0, 0]            # frames, raw #1 changes, smoothed #1 changes, shown-label changes, unsure frames
PREV = [None, None, None]        # previous raw #1, smoothed #1, shown label
T_SC = [0, 0]                    # us: raw scoring (unit + cosine + top-2), smoothing + top3 + decision
T_RS = [0]                       # us: resize part of fill() (only with PROFILE)


def crc(b):
    return binascii.crc32(b) & 0xFFFFFFFF


def nm(names, i):
    return "?" if i < 0 else names[i].replace(" ", "_")


def setup_text_ulab(norms):
    """Unit-length float rows of the words, from the same embedded bytes that load_text() parsed."""
    if TEXT_SOURCE != "embedded":
        raise ValueError("v2 needs TEXT_SOURCE = embedded")
    b = binascii.a2b_base64(TEXT_B64)
    n = len(norms)
    Ti = np.frombuffer(b[16 + 4 * n:], dtype=np.int8).reshape((n, 512))
    Tf = np.array(Ti, dtype=np.float)
    for i in range(n):
        Tf[i] = Tf[i] / norms[i]
    wdt.feed()
    V2["Tf"] = Tf
    reset_state()


def reset_state():
    V2["ema"] = None; V2["label"] = -1
    PREV[0] = None; PREV[1] = None; PREV[2] = None
    for i in range(len(CNT)):
        CNT[i] = 0


def unit(ob):
    """int8 output bytes -> unit-length float vector (the output scale cancels; zero point is 0 for this model)."""
    v = np.array(np.frombuffer(ob, dtype=np.int8), dtype=np.float)
    if IO[3]:
        v = v - IO[3]
    nv = np.linalg.norm(v)
    return v / nv if nv > 0 else v


def top2(l):
    a = 0 if l[0] >= l[1] else 1
    b = 1 - a
    for i in range(2, len(l)):
        x = l[i]
        if x > l[a]:
            b = a; a = i
        elif x > l[b]:
            b = i
    return a, b


def smooth(e):
    """EMA of the unit embedding; returns the cosine of the smoothed vector with every word (ulab array)."""
    m = V2["ema"]
    m = e if m is None else m * (1.0 - ALPHA) + e * ALPHA
    V2["ema"] = m
    return np.dot(V2["Tf"], m) / np.linalg.norm(m)


def decide(top):
    """top = top3() of the smoothed scores. Returns (shown word index or -1 for '?', margin #1 - #2)."""
    i0 = top[0][0]
    margin = top[0][1] - top[1][1]
    cur = V2["label"]
    if not (cur == i0 and margin >= THR_OFF):
        cur = i0 if margin >= THR_ON else -1
    V2["label"] = cur
    return cur, margin


def process(m):
    """One frame, after SRC[0] / ROI[0] are set. Returns (output bytes, raw #1, raw cos #1, raw margin,
    smoothed top-3 [(i, cos, p)], shown label, smoothed margin)."""
    ob = predict(m)                                  # known-good core: fill() -> NPU -> copy of the 512 int8
    t0 = time.ticks_us()
    e = unit(ob)
    lr = np.dot(V2["Tf"], e).tolist()                # raw cosines = what the known-good demo scores
    r1, r2 = top2(lr)
    t1 = time.ticks_us()
    top = top3(smooth(e).tolist())
    label, margin = decide(top)
    t2 = time.ticks_us()
    T_SC[0] = time.ticks_diff(t1, t0); T_SC[1] = time.ticks_diff(t2, t1)
    CNT[0] += 1
    if PREV[0] is not None:
        CNT[1] += 1 if r1 != PREV[0] else 0
        CNT[2] += 1 if top[0][0] != PREV[1] else 0
        CNT[3] += 1 if label != PREV[2] else 0
    CNT[4] += 1 if label < 0 else 0
    PREV[0] = r1; PREV[1] = top[0][0]; PREV[2] = label
    return ob, r1, lr[r1], lr[r1] - lr[r2], top, label, margin


def timed_src(mv):
    t0 = time.ticks_us()
    draw_planes(mv)
    T_RS[0] = time.ticks_diff(time.ticks_us(), t0)


def overlay_v2(img, names, top, label, fps):
    """Same layout as the known-good overlay. New: the first line is yellow with a leading '?' when unsure."""
    x0, y0, w, h = ROI[0]
    img.draw_rectangle((x0, y0, w, h), color=(255, 255, 0))          # the area that the model sees
    img.draw_rectangle((x0, y0, w, 62), color=(0, 0, 0), fill=True)
    for r in range(len(top)):
        i, c, p = top[r]
        t = "%s %d%%" % (names[i], int(p * 100 + 0.5))
        col = (255, 255, 255)
        if r == 0:
            if label < 0:
                t = "? " + t; col = (255, 255, 0)
            else:
                col = (0, 255, 0)
        img.draw_string((x0 + 4, y0 + 2 + 20 * r), t, color=col, scale=2)
    img.draw_string((x0 + 4, y0 + h - 12), "cos %.2f %.2f %.2f  %.2f fps" % (top[0][1], top[1][1], top[2][1], fps),
                    color=(255, 255, 0), scale=1)
