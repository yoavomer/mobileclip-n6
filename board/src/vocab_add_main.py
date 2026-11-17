# ---------------------------------------------------------------- "add a word" demo, camera loop (11.11)
# = the demo v2 loop (smoothing, "?" when unsure, top-3), but a word typed on the PC is ADDED to the word list:
# from the next frame on it is scored, ranked, smoothed and shown exactly like the 43 built-in words (top-3 lines,
# percentages, "?" rule). Nothing is drawn at the bottom except a small word counter ("43 words" -> "44 words").
# The word that is in /flash/liveword.bin when the script starts is NOT added (the demo starts with the 43 words);
# only words the PC writes after the start are added. The same word again (e.g. the exact vector after the quick
# "word ..." one) replaces its own row. The board only READS the file.
# Output lines:
#   VA ADD <word> row <i> words <n> seq <s> t <ms> k <frame>     a word was added / replaced
#   V <frame> words <n> | show <label> | top3 <w> <cos> <w> <cos> <w> <cos> | <fps> fps t <ms>
VA_FRAMES = 0                    # 0 = until stopped


def va_add(names, n0):
    """Adds the word just taken by lw_poll() as a row of V2["Tf"] (or replaces its row). Returns its row index."""
    w = LW["word"]
    if w.endswith(" ..."):
        w = w[:-4]
    w = w.strip()
    Tf = V2["Tf"]
    n = len(names)
    for i in range(n0, n):
        if names[i] == w:
            Tf[i] = LW["M"][0]
            return i
    T2 = np.array(np.frombuffer(bytes(512 * (n + 1)), dtype=np.int8).reshape((n + 1, 512)), dtype=np.float)
    for i in range(n):
        T2[i] = Tf[i]
    T2[n] = LW["M"][0]
    V2["Tf"] = T2
    names.append(w)
    gc.collect()
    return n


def overlay_count(img, nwords, n0):
    x0, y0, w, h = ROI[0]
    t = "%d words" % nwords
    if nwords > n0:
        t += " (+%d)" % (nwords - n0)
    img.draw_string((x0 + w - 8 * len(t) - 4, y0 + h - 24), t, color=(255, 255, 0), scale=1)


def main():
    start_wait()
    step("VA demo: model=%s text=%s file=%s ALPHA=%.2f THR_ON=%.4f THR_OFF=%.4f VA_FRAMES=%d (typed words join the word list; the board only reads the file; Stop or the end: the WDT resets the board ~10 s later, expected)" % (
        MODEL_NAME, TEXT_SOURCE, LW_PATH, ALPHA, THR_ON, THR_OFF, VA_FRAMES))
    names, rows, norms = load_text()
    setup_text_ulab(norms)
    del rows
    gc.collect()
    lw_setup()
    lw_poll()                                        # baseline: the word already in the file is NOT added
    step("VA baseline seq=%d word=%s (not added)" % (LW["seq"], LW["word"]))
    LW["have"] = 0
    n0 = len(names)
    cam = open_camera()
    m = open_model()                                 # alignment check BEFORE the first predict
    k = 0
    t_prev = None
    fps = 0.0
    while VA_FRAMES == 0 or k < VA_FRAMES:
        wdt.feed()
        t_now = time.ticks_us()
        if t_prev is not None:
            fps = 1000000.0 / max(1, time.ticks_diff(t_now, t_prev))
        t_prev = t_now
        img = cam.snapshot()
        set_source(img)
        if k == 0:
            check_frame(img)
            step("C frame %dx%d roi=%r" % (img.width(), img.height(), ROI[0]))
        if lw_poll():
            r = va_add(names, n0)
            print("VA ADD %s row %d words %d seq %d t %d k %d" % (names[r], r, len(names), LW["seq"],
                                                                   time.ticks_ms(), k))
        ob, r1, rc, rm, top, label, margin = process(m)   # wdt.feed() before and after the predict inside
        overlay_v2(img, names, top, label, fps)
        overlay_count(img, len(names), n0)
        print("V %d words %d | show %s | top3 %s %.4f %s %.4f %s %.4f | %.2f fps t %d" % (
            k, len(names), nm(names, label), nm(names, top[0][0]), top[0][1], nm(names, top[1][0]), top[1][1],
            nm(names, top[2][0]), top[2][1], fps, time.ticks_ms()))
        k += 1
        wdt.feed()
    step("VA DONE frames=%d words=%d taken=%d bad=%d err=%d" % (k, len(names), LW["taken"], LW["bad"], LW["err"]))
