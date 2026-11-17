"""Overnight 2, Part 5 (PC side): type a word -> MobileCLIP-S0 text encoder -> 512 int8 values -> the board.

Channel: ONE fixed-size file on the board's own USB drive, E:/liveword.bin (the board sees it as /flash/liveword.bin).
It is created once (with the board idle) and afterwards overwritten IN PLACE (same size, same place on the drive), so
only its data sectors change. The board script re-opens and reads it every frame and never writes. Nothing else on the
drive is touched (main.py, ROMFS, firmware: not touched).

Text method = the demo words (night/demo_run/pc/build_vocab43.py): the 80 OpenAI ImageNet templates with the typed
word, each prompt L2-normalised, averaged, L2-normalised again; then W5's row quantiser (scale = max|x| / 127, int8).

File layout (1024 bytes, little endian):
    0  4s   magic "MCLW"
    4  H    version = 1
    6  H    seq: changes with every word; the board takes a word when seq differs from the one it has
    8  I    crc32 of bytes 12..1023
   12  f    row scale (not needed for the cosine; kept for completeness)
   16  I    PC time of the write, ms modulo 2^32
   20  B    length of the word in bytes
   21  64s  the word, utf-8, zero padded
  512  512  the int8 row

usage (always: python -I live_word.py ...):
  --selfcheck                           re-encode 3 demo prompts, compare the int8 rows with the demo's own rows
  --create WORD                         create the file (refuses if it exists). Do this with the board idle.
  (no option)                           interactive: type a word + Enter; an empty line quits
  --words glass,pen,painting --gap 10   scripted, for unattended measurements (gap = seconds between words)
  --watch                               also time "word visible in the IDE frame buffer" from the colour marker that
                                        live_demo draws (colour = seq modulo 6); needs the IDE visible on the screen
  --path E:/liveword.bin   --log NAME   (log: release/live/logs/NAME.jsonl, one JSON line per word)
  --threads N                           CPU threads for the text encoder (default 4: fastest on this PC, see
                                        release/live/results/encode_time.json)
  --quick-first                         optional: first send a quick vector made from ONE prompt (shown on the board
                                        as "word ..."), then replace it by the exact 80-prompt vector (shown as
                                        "word"). Without this option only the exact vector is sent (the demo method).
"""
import sys, os, time, json, struct, zlib, argparse, pathlib

R = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + "")
LIVE = R / "release" / "live"
SIZE = 1024
PAL = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]   # = LW_PAL on the board


def load_encoder(threads=2):
    """Returns emb(word) -> unit float32 vector of 512 (80 templates, the demo recipe)."""
    import torch
    sys.path.insert(0, str(R / "ml-mobileclip")); torch.set_num_threads(threads)
    import mobileclip
    from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
    model, _, _ = mobileclip.create_model_and_transforms(
        "mobileclip_s0", pretrained=str(R / "checkpoints" / "mobileclip_s0.pt"), reparameterize=False)
    model.eval(); tok = mobileclip.get_tokenizer("mobileclip_s0")

    def emb(word):
        with torch.no_grad():
            e = model.encode_text(tok([t(word) for t in OPENAI_IMAGENET_TEMPLATES])).float()
        e = e / e.norm(dim=-1, keepdim=True); m = e.mean(dim=0)
        return (m / m.norm()).numpy()
    def one(word):
        """ONE prompt only ("a photo of a X."): about 50x faster, but NOT the demo method (see --quick-first)."""
        with torch.no_grad():
            e = model.encode_text(tok(["a photo of a %s." % word])).float()[0]
        return (e / e.norm()).numpy()
    emb.one = one
    return emb


def quant(e):
    """W5's quantize_rows for one row: scale = max|x| / 127, round to nearest, clip to +-127."""
    import numpy as np
    e = e.astype(np.float64)
    s = np.abs(e).max() / 127.0
    q = np.clip(np.rint(e / s), -127, 127).astype(np.int8)
    return float(np.float32(s)), q


def pack(seq, word, scale, q):
    w = word.encode("utf-8")
    assert 0 < len(w) <= 64, "word must be 1..64 bytes"
    assert len(q) == 512
    body = struct.pack("<fIB", scale, int(time.time() * 1000) & 0xFFFFFFFF, len(w)) + w.ljust(64, b"\0")
    body = body.ljust(512 - 12, b"\0") + bytes(memoryview(q).cast("B"))
    blob = b"MCLW" + struct.pack("<HHI", 1, seq & 0xFFFF, zlib.crc32(body) & 0xFFFFFFFF) + body
    assert len(blob) == SIZE
    return blob


def parse(blob):
    if len(blob) != SIZE or blob[:4] != b"MCLW":
        return None
    ver, seq, crc = struct.unpack_from("<HHI", blob, 4)
    if ver != 1 or zlib.crc32(blob[12:]) & 0xFFFFFFFF != crc:
        return None
    scale, pc_ms, wl = struct.unpack_from("<fIB", blob, 12)
    return {"seq": seq, "word": blob[21:21 + wl].decode("utf-8"), "scale": scale, "pc_ms": pc_ms, "row": blob[512:]}


def write_in_place(path, blob):
    """Overwrite the existing file, same size, and push it to the drive now."""
    assert os.path.getsize(path) == SIZE, "the file must already exist with %d bytes (use --create once)" % SIZE
    with open(path, "r+b") as f:
        f.seek(0); f.write(blob); f.flush(); os.fsync(f.fileno())


def current_seq(path):
    try:
        p = parse(open(path, "rb").read())
        return p["seq"] if p else 0
    except OSError:
        return 0


# ------------------------------------------------------------------ screen marker (only with --watch)
def grab():
    from PIL import ImageGrab
    import numpy as np
    return np.asarray(ImageGrab.grab(all_screens=True).convert("RGB")).astype(np.int16)


def colour_mask(a, col, tol=70):
    import numpy as np
    return (np.abs(a - np.array(col, np.int16)).max(axis=2) <= tol)


def find_marker(col, win=20):
    """Bounding box (x0, y0, x1, y1) of the area where a win x win square is completely of this colour, or None."""
    import numpy as np
    m = colour_mask(grab(), col).astype(np.int32)
    c = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    full = (c[win:, win:] - c[:-win, win:] - c[win:, :-win] + c[:-win, :-win]) == win * win
    ys, xs = np.nonzero(full)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + win, int(ys.max()) + win


def marker_colour(box):
    """Index of the palette colour that fills the centre of the box right now, or -1."""
    from PIL import ImageGrab
    import numpy as np
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    im = ImageGrab.grab(all_screens=True).crop((cx - 4, cy - 4, cx + 5, cy + 5)).convert("RGB")
    p = np.asarray(im).astype(np.int16).reshape(-1, 3).mean(0)
    for i, col in enumerate(PAL):
        if max(abs(p[j] - col[j]) for j in range(3)) <= 70:
            return i
    return -1


def wait_marker(box, want, timeout=6.0):
    """Returns (time the wanted colour was first seen or None, number of screen grabs, mean grab period in ms)."""
    t0 = time.time(); n = 0
    while time.time() - t0 < timeout:
        n += 1
        if marker_colour(box) == want:
            t = time.time()
            return t, n, (t - t0) * 1000.0 / n
        time.sleep(0.005)
    return None, n, (time.time() - t0) * 1000.0 / max(1, n)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="E:/liveword.bin")
    ap.add_argument("--create", default=None); ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--words", default=None); ap.add_argument("--gap", type=float, default=10.0)
    ap.add_argument("--first-wait", type=float, default=0.0)
    ap.add_argument("--watch", action="store_true"); ap.add_argument("--log", default="live_word")
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--quick-first", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    emb = load_encoder(a.threads)
    t_load = time.time() - t0
    print("text encoder loaded in %.1f s" % t_load, flush=True)

    if a.selfcheck:
        import numpy as np
        sys.path.insert(0, str(R / "night" / "W5_demo" / "pc"))
        import make_board_files as MBF
        info = json.load(open(R / "night/demo_run/pc/vocab43/vocab43_info.json"))
        ref = MBF.parse((R / "night/demo_run/pc/vocab43/demo_text.bin").read_bytes())
        s_ref, q_ref = ref["scales"], ref["q"]
        out = {}
        for d in ("glass", "pen", "painting"):
            i = info["display"].index(d)
            s, q = quant(emb(info["prompt_names"][i]))
            out[d] = {"prompt": info["prompt_names"][i], "int8_values_different": int((q != q_ref[i]).sum()),
                      "scale_rel_diff": abs(s - float(s_ref[i])) / float(s_ref[i])}
        print(json.dumps(out)); return

    if a.create is not None:
        assert not os.path.exists(a.path), "%s exists already: not overwritten by --create" % a.path
        s, q = quant(emb(a.create)); blob = pack(1, a.create, s, q)
        with open(a.path, "xb") as f:
            f.write(blob); f.flush(); os.fsync(f.fileno())
        print("created", a.path, os.path.getsize(a.path), "bytes, seq 1, word", repr(a.create)); return

    seq = current_seq(a.path)
    (LIVE / "logs").mkdir(exist_ok=True)
    logf = open(LIVE / "logs" / (a.log + ".jsonl"), "a", encoding="utf-8")
    box = None
    if a.watch:
        for _ in range(40):                                  # the board script may still be starting
            box = find_marker(PAL[seq % 6])
            if box:
                break
            time.sleep(0.5)
        print("marker for seq %d: %s" % (seq, box), flush=True)
    words = [w.strip() for w in a.words.split(",")] if a.words else None
    if words and a.first_wait:
        time.sleep(a.first_wait)
    k = 0
    while True:
        if words is not None:
            if k >= len(words):
                break
            word = words[k]
        else:
            try:
                word = input("word> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not word:
                break
        t_enter = time.time()                                # the moment Enter was pressed (scripted: "typed" now)
        quick = None
        if a.quick_first:
            s, q = quant(emb.one(word))
            tq_enc = time.time()
            seq = (seq + 1) & 0xFFFF
            write_in_place(a.path, pack(seq, (word + " ...")[:64], s, q))
            tq_wr = time.time()
            quick = {"seq": seq, "encode_ms": round((tq_enc - t_enter) * 1000, 1), "write_ms": round((tq_wr - tq_enc) * 1000, 1)}
            if box:                                          # watch the screen while the exact vector is computed
                import threading
                seen = []
                th = threading.Thread(target=lambda sq=seq: seen.append(wait_marker(box, sq % 6, 3.0)))
                th.start()
        s, q = quant(emb(word))
        t_enc = time.time()
        seq = (seq + 1) & 0xFFFF
        write_in_place(a.path, pack(seq, word, s, q))
        t_wr = time.time()
        rec = {"seq": seq, "word": word, "t_enter": round(t_enter, 4), "encode_ms": round((t_enc - t_enter) * 1000, 1),
               "write_ms": round((t_wr - t_enc) * 1000, 1)}
        if quick is not None:
            if box:
                th.join()
                rec_t = seen[0][0]
                quick["visible_ms_after_enter"] = None if rec_t is None else round((rec_t - t_enter) * 1000, 1)
                quick["visible_ms_after_write"] = None if rec_t is None else round((rec_t - tq_wr) * 1000, 1)
            rec["quick"] = quick
        if box:
            t_seen, n_grab, grab_ms = wait_marker(box, seq % 6)
            rec["grabs"] = n_grab; rec["grab_period_ms"] = round(grab_ms, 1)
            rec["visible_ms_after_enter"] = None if t_seen is None else round((t_seen - t_enter) * 1000, 1)
            rec["visible_ms_after_write"] = None if t_seen is None else round((t_seen - t_wr) * 1000, 1)
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        k += 1
        if words is not None and k < len(words):
            time.sleep(max(0.0, a.gap - (time.time() - t_enter)))
    logf.close()


if __name__ == "__main__":
    main()
