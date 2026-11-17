# ---------------------------------------------------------------- core (shared by demo.py and validate_input.py)
# APIs used and where they were checked (OpenMV firmware 5.0.1 source, ext/openmv-5.0.1):
#   csi.CSI().reset/pixformat/framesize/snapshot(time=)  modules/py_csi_ng.c; same calls as board/check_yolo_cut.py
#   image.Image(w, h, image.GRAYSCALE, buffer=...)        modules/py_image.c py_image_make_new (buffer kw)
#   img.draw_image(src, 0, 0, roi=, rgb_channel=, hint=)  modules/py_image.c py_image_line_op, lib/imlib/draw.c
#   img.to_ndarray("b", buffer=...)                       modules/py_image.c py_image_to_ndarray (GRAYSCALE: u8 ^ 0x80)
#   img.draw_string((x, y), s, color=, scale=)            modules/py_image.c py_image_draw_string (xy is ONE tuple)
#   img.draw_rectangle((x, y, w, h), color=, fill=)       modules/py_image.c py_image_draw_rectangle
#   img.get_pixel((x, y), rgbtuple=)                      modules/py_image.c py_image_get_pixel
#   ml.Model(path); m.predict([fill], callback=cb)        modules/py_ml.c (fill(buf, shape, dtype) gets the input
#     tensor as a bytearray BY REFERENCE; cb(model, inputs, outputs) gets outputs[0] = ulab ndarray VIEW of the output
#     tensor; input_scale / input_zero_point / ... are tuples, one item per tensor)
#   ml/model.py wrapper (scripts/libraries/ml/ml-core): ml.Model -> uml.Model, labels from <path>.txt if present
#   math.exp: used by scripts/libraries/ml/ml-core/ml/utils.py
#   uctypes.addressof(open(rom file)), binascii.crc32 / a2b_base64 / b2a_base64, struct.unpack("<512f"),
#   bytes(outputs[0].tobytes()) inside the callback: all used by board scripts that ran on this board (board/i16_*.py,
#   board/stem4_*.py).
# The input tensor and the output tensor share ONE buffer (REPORT "Board vs ORT - fact collection" 0d/1c), so the
# input is rewritten by fill() before EVERY predict and the output is copied inside the callback.

def step(s):
    print(s)
    time.sleep_ms(30)                # the IDE terminal drops unpaced lines (REPORT "Stem: where does the board round?")


def start_wait():
    for _ in range(6):               # 3 s: the IDE terminal drops the first ~0.5 s of output
        time.sleep_ms(500)
        wdt.feed()


# ---- text embeddings (format MCTX v1, see night/W5_demo/README.md) ----
def parse_text(b, names):
    if bytes(b[0:4]) != b"MCTX":
        raise ValueError("demo_text: bad magic")
    ver, n, d, mode, crc = struct.unpack("<HHHHI", b[4:16])
    body = b[16:]
    if ver != 1 or len(body) != 4 * n + n * d:
        raise ValueError("demo_text: bad version/size")
    if (binascii.crc32(body) & 0xFFFFFFFF) != crc:
        raise ValueError("demo_text: CRC32 mismatch")
    if len(names) != n:
        raise ValueError("demo_text: %d names for %d rows" % (len(names), n))
    rows = []
    norms = []
    o = 16 + 4 * n
    for i in range(n):
        r = struct.unpack("<%db" % d, b[o + i * d:o + (i + 1) * d])
        rows.append(r)
        norms.append(sum([x * x for x in r]) ** 0.5)
        if i % 8 == 7:
            wdt.feed()
    return rows, norms, mode, crc, d


def load_text():
    if TEXT_SOURCE == "embedded":
        b = binascii.a2b_base64(TEXT_B64)
        names = TEXT_NAMES.split("\n")
    elif TEXT_SOURCE == "rom":
        f = open("/rom/demo_text.bin", "rb"); b = f.read(); f.close()
        f = open("/rom/demo_text.txt", "r"); names = [l.strip() for l in f.read().split("\n") if l.strip()]; f.close()
    elif TEXT_SOURCE == "module":
        import demo_text
        b = binascii.a2b_base64(demo_text.BIN_B64)
        names = list(demo_text.NAMES)
    else:
        raise ValueError("TEXT_SOURCE must be embedded | rom | module")
    wdt.feed()
    rows, norms, mode, crc, d = parse_text(b, names)
    step("T text source=%s entries=%d dim=%d prompt_mode=%s crc32=0x%08x" % (
        TEXT_SOURCE, len(names), d, ("templates80", "simple")[mode] if mode < 2 else str(mode), crc))
    return names, rows, norms


# ---- model ----
def open_model():
    path = "/rom/" + MODEL_NAME
    f = open(path, "rb"); a = uctypes.addressof(f) & 0xFFFFFFFF; f.close()
    e = EXPECT.get(MODEL_NAME)
    step("A0 %s at 0x%08x mod8=%d%s" % (MODEL_NAME, a, a % 8, "" if e is None else (
        " expected 0x%08x %s" % (e[0], "MATCH" if a == e[0] else "MISMATCH (info only)"))))
    if a % 8:
        step("STOP: model address is not 8-byte aligned, no predict")
        raise SystemExit
    wdt.feed(); gc.collect()
    m = ml.Model(path)
    wdt.feed()
    step("A1 %s" % (m,))
    ok = (len(m.input_shape) == 1 and tuple(m.input_shape[0]) == (1, 3, 256, 256) and m.input_dtype[0] == "b"
          and len(m.output_shape) == 1 and tuple(m.output_shape[0]) == (1, 512) and m.output_dtype[0] == "b"
          and m.output_scale[0] > 0)
    if not ok:
        step("STOP: unexpected model I/O (need int8 input (1,3,256,256) NCHW and int8 output (1,512))")
        raise SystemExit
    if e is not None and m.len != e[1]:
        step("WARN model_size %d, expected %d" % (m.len, e[1]))
    IO[0] = m.input_scale[0]; IO[1] = m.input_zero_point[0]; IO[2] = m.output_scale[0]; IO[3] = m.output_zero_point[0]
    make_lut(IO[0], IO[1])
    step("A2 in_scale=%r in_zp=%d out_scale=%r out_zp=%d lut=%s" % (IO[0], IO[1], IO[2], IO[3],
                                                                     "xor80" if LUT_XOR[0] else "table"))
    return m


def make_lut(s, z):
    """q = clamp(round_half_even(p / 255 / s) + z, -128, 127) for every pixel value p, s/z READ from ml.Model."""
    lut = bytearray(256)
    xor = True
    for p in range(256):
        v = (p / 255) / s
        fl = int(v // 1)
        d = v - fl
        q = fl + (1 if (d > 0.5 or (d == 0.5 and fl % 2 == 1)) else 0) + z
        q = -128 if q < -128 else (127 if q > 127 else q)
        lut[p] = q & 0xFF
        if lut[p] != ((p - 128) & 0xFF):
            xor = False
    LUT[0] = bytes(lut)
    LUT_XOR[0] = xor


# ---- preprocessing: centre square crop -> 256x256 bilinear -> R, G, B planes (NCHW) -> int8 ----
def set_source(img):
    w = img.width(); h = img.height(); s = w if w < h else h
    ROI[0] = ((w - s) // 2, (h - s) // 2, s, s)
    SRC[0] = img


def draw_planes(mv):
    """Camera source: centre square (ROI[0]) of SRC[0] -> 256x256 bilinear -> R, G, B uint8 planes (NCHW order)."""
    src = SRC[0]; roi = ROI[0]
    for c in range(3):                               # plane c = channel c of NCHW: 0 = R, 1 = G, 2 = B
        image.Image(256, 256, image.GRAYSCALE, buffer=mv[c * 65536:(c + 1) * 65536]).draw_image(
            src, 0, 0, roi=roi, rgb_channel=c, hint=image.BILINEAR | image.SCALE_ASPECT_IGNORE)


SOURCE = [draw_planes]                               # demo_run 9.11: the known-answer test swaps in a file source


def fill(buf, shape, dtype):
    """predict() input callback: writes the WHOLE int8 input tensor (196,608 bytes) from SRC[0] / ROI[0]."""
    t0 = time.ticks_us()
    if dtype != 98 or len(buf) != 196608:            # 98 = ord("b")
        raise ValueError("unexpected input tensor %r %r %r" % (len(buf), shape, dtype))
    SOURCE[0](memoryview(buf))                       # writes the three uint8 planes (camera: draw_planes)
    if LUT_XOR[0]:                                   # LUT[p] == p - 128: in place u8 -> u8 ^ 0x80 (C, vectorized)
        image.Image(256, 768, image.GRAYSCALE, buffer=buf).to_ndarray("b", buffer=buf)
    else:                                            # other input scale/zp: generic table (slow, Python)
        lut = LUT[0]
        for i in range(196608):
            buf[i] = lut[buf[i]]
            if (i & 16383) == 0:
                wdt.feed()
    if CAPTURE[0] is not None:
        CAPTURE[0].append(bytes(buf))                # exact input bytes (validation only), before the NPU runs
    T_FILL[0] = time.ticks_diff(time.ticks_us(), t0)


def grab(model, inputs, outputs):
    """predict() output callback: copy the int8 output NOW (the buffer is shared with the next input)."""
    OUT[0] = bytes(outputs[0].tobytes())
    return None


def predict(m):
    wdt.feed()
    t0 = time.ticks_us()
    m.predict([fill], callback=grab)
    T_PRED[0] = time.ticks_diff(time.ticks_us(), t0)
    wdt.feed()
    return OUT[0]


# ---- scoring ----
def score(ob, rows, norms):
    """dequantize (q - zp) * out_scale -> L2-normalize -> cosine with every text row.
    The positive factor out_scale and the int8 row scales cancel in the cosine, so it is computed exactly in integers:
    cos_i = (q_i . v) / (|q_i| |v|) with v = q_out - zp_out."""
    zp = IO[3]
    v = [(x - 256 if x > 127 else x) - zp for x in ob]
    nv = sum([x * x for x in v]) ** 0.5
    sc = []
    for i in range(len(rows)):
        t = rows[i]
        s = 0
        for k in range(512):
            s += t[k] * v[k]
        sc.append(s / (norms[i] * nv) if nv > 0 else 0.0)
    return sc


def top3(sc):
    idx = sorted(range(len(sc)), key=lambda i: -sc[i])[:3]
    mx = sc[idx[0]]
    z = sum([math.exp(LOGIT_SCALE * (x - mx)) for x in sc])
    return [(i, sc[i], math.exp(LOGIT_SCALE * (sc[i] - mx)) / z) for i in idx]


def check_frame(img):
    if img.format() != image.RGB565:
        step("STOP: camera frame is not RGB565 (format %r)" % (img.format(),))
        raise SystemExit


def open_camera():
    cam = csi.CSI()
    wdt.feed()
    cam.reset()
    wdt.feed()
    cam.pixformat(csi.RGB565)
    cam.framesize(FRAMESIZE)
    wdt.feed()
    cam.snapshot(time=2000)                          # let exposure / white balance settle (2 s < WDT 10 s)
    wdt.feed()
    return cam
