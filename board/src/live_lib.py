# ---------------------------------------------------------------- live word library (overnight 2, Part 5)
# The typed word arrives in ONE fixed-size file on the board's own drive: /flash/liveword.bin (the PC sees the same
# file as E:\liveword.bin and overwrites it in place). The board re-opens and reads the file once per frame and
# NEVER writes to the drive.
# Layout (1024 bytes, little endian): 0 "MCLW" | 4 u16 version = 1 | 6 u16 seq | 8 u32 crc32 of bytes 12..1023 |
#   12 f32 row scale | 16 u32 PC ms | 20 u8 word length | 21 word, utf-8, 64 bytes | 512 the 512 int8 values.
# A word is taken only when seq is new AND the CRC is right. Otherwise the previous word stays, and a half-written
# file is simply read again on the next frame.
LW_PATH = "/flash/liveword.bin"
LW = {"seq": -1, "word": "", "M": None, "have": 0, "taken": 0, "bad": 0, "err": 0, "pc_ms": 0, "last_err": ""}
LW_BUF = bytearray(1024)
T_LW = [0, 0]                    # us: last poll (open + read + close + header check), last take (crc + parse)
LW_PAL = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]   # marker = seq % 6


def lw_setup():
    LW["M"] = np.array(np.frombuffer(bytes(512), dtype=np.int8).reshape((1, 512)), dtype=np.float)


def lw_poll():
    """Read the file. Returns True when a new word was taken on this call."""
    t0 = time.ticks_us()
    n = -1
    try:
        f = open(LW_PATH, "rb")
        try:
            n = f.readinto(LW_BUF)
        finally:
            f.close()
    except OSError as e:
        LW["err"] += 1; LW["last_err"] = repr(e)
        T_LW[0] = time.ticks_diff(time.ticks_us(), t0)
        return False
    b = LW_BUF
    took = False
    if n == 1024 and b[0] == 77 and b[1] == 67 and b[2] == 76 and b[3] == 87 and b[4] == 1 and b[5] == 0:   # "MCLW", 1
        seq = b[6] | (b[7] << 8)
        T_LW[0] = time.ticks_diff(time.ticks_us(), t0)
        if seq != LW["seq"]:
            t1 = time.ticks_us()
            wl = b[20]
            if crc(memoryview(b)[12:]) == struct.unpack_from("<I", b, 8)[0] and 0 < wl <= 64:
                v = np.array(np.frombuffer(b[512:1024], dtype=np.int8), dtype=np.float)
                nv = np.linalg.norm(v)
                if nv > 0:
                    try:
                        w = bytes(b[21:21 + wl]).decode()
                    except Exception:
                        w = "?"
                    LW["M"][0] = v / nv
                    LW["seq"] = seq; LW["word"] = w; LW["pc_ms"] = struct.unpack_from("<I", b, 16)[0]
                    LW["have"] = 1; LW["taken"] += 1
                    took = True
            if not took:
                LW["bad"] += 1
            T_LW[1] = time.ticks_diff(time.ticks_us(), t1)
    else:
        LW["bad"] += 1
        T_LW[0] = time.ticks_diff(time.ticks_us(), t0)
    return took


def lw_cos(e, en):
    """Cosine of the typed word with the embedding e (ulab vector of 512 with length en)."""
    return np.dot(LW["M"], e).tolist()[0] / en


def lw_rank(cw, cosines):
    """1 = the typed word scores higher than every one of the built-in words."""
    r = 1
    for x in cosines:
        if x > cw:
            r += 1
    return r


def overlay_live(img, cw, rank, nwords):
    """Adds the typed word (bottom of the model's area) and the seq colour marker (left margin) to the v2 overlay."""
    x0, y0, w, h = ROI[0]
    img.draw_rectangle((x0, y0 + h - 48, w, 34), color=(0, 0, 0), fill=True)
    if LW["have"]:
        col = (0, 255, 0) if rank == 1 else ((255, 255, 0) if rank <= 3 else (255, 255, 255))
        img.draw_string((x0 + 4, y0 + h - 46), "> " + LW["word"], color=col, scale=2)
        img.draw_string((x0 + 4, y0 + h - 24), "cos %.3f rank %d/%d" % (cw, rank, nwords + 1),
                        color=col, scale=1)
        if x0 >= 58:
            img.draw_rectangle((4, 4, 50, 50), color=LW_PAL[LW["seq"] % 6], fill=True)
    else:
        img.draw_string((x0 + 4, y0 + h - 46), "> (no word)", color=(255, 255, 255), scale=2)
