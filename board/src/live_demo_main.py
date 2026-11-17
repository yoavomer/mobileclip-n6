# ---------------------------------------------------------------- live word demo, camera loop (overnight 2, Part 5)
# = the demo v2 loop (smoothing, "?" when unsure, top-3 of the 43 built-in words) + the word typed on the PC.
# Once per frame, right after the model has run, the board reads /flash/liveword.bin. The typed word's score is the
# cosine of its text row with the SMOOTHED image embedding (the same vector the top-3 uses), so it is as steady as
# the top-3. "rank" = position of the typed word among the 43 built-in words + itself (1 = it beats all of them).
# Output lines:
#   LW NEW seq word ...   a new word was taken (t = board ms)
#   L <frame> seq <n> word <w> cos <smoothed> rank <r> raw <this frame alone> | show <v2 label> | top3 ... |
#     <fps> fps loop <us> poll <us> lw <us: typed-word score + its drawing> t <board ms>
#   S ...                 every 100 frames: counters kept on the board
LIVE_FRAMES = 0                  # 0 = until stopped


def main():
    start_wait()
    step("LIVE demo: model=%s text=%s file=%s ALPHA=%.2f THR_ON=%.4f THR_OFF=%.4f LIVE_FRAMES=%d (the board only reads the file; Stop or the end: the WDT resets the board ~10 s later, expected)" % (
        MODEL_NAME, TEXT_SOURCE, LW_PATH, ALPHA, THR_ON, THR_OFF, LIVE_FRAMES))
    names, rows, norms = load_text()
    setup_text_ulab(norms)
    del rows
    gc.collect()
    lw_setup()
    cam = open_camera()
    m = open_model()                                 # alignment check BEFORE the first predict
    nw = len(names)
    k = 0
    t_prev = None
    fps = 0.0
    loop_us = 0
    acc = [0, 0, 0]; nacc = 0                        # loop, poll, typed-word work
    while LIVE_FRAMES == 0 or k < LIVE_FRAMES:
        wdt.feed()
        t_now = time.ticks_us()
        if t_prev is not None:
            loop_us = time.ticks_diff(t_now, t_prev)
            fps = 1000000.0 / max(1, loop_us)
        else:
            loop_us = 0
        t_prev = t_now
        img = cam.snapshot()
        set_source(img)
        if k == 0:
            check_frame(img)
            step("C frame %dx%d roi=%r" % (img.width(), img.height(), ROI[0]))
        ob, r1, rc, rm, top, label, margin = process(m)   # wdt.feed() before and after the predict inside
        if lw_poll():
            print("LW NEW seq=%d word=%s pc_ms=%d t=%d k=%d poll %d take %d" % (
                LW["seq"], LW["word"], LW["pc_ms"], time.ticks_ms(), k, T_LW[0], T_LW[1]))
        t0 = time.ticks_us()
        overlay_v2(img, names, top, label, fps)
        cw = 0.0; cr = 0.0; rank = 0
        if LW["have"]:
            ema = V2["ema"]
            en = np.linalg.norm(ema)
            cw = lw_cos(ema, en)
            rank = lw_rank(cw, (np.dot(V2["Tf"], ema) / en).tolist())
            cr = lw_cos(unit(ob), 1.0)
        overlay_live(img, cw, rank, nw)
        t_lw = time.ticks_diff(time.ticks_us(), t0)
        if loop_us and k > 1:
            acc[0] += loop_us; acc[1] += T_LW[0]; acc[2] += t_lw; nacc += 1
        print("L %d seq %d word %s cos %.4f rank %d raw %.4f | show %s | top3 %s %.4f %s %.4f %s %.4f | %.2f fps loop %d poll %d lw %d t %d" % (
            k, LW["seq"], LW["word"].replace(" ", "_"), cw, rank, cr, nm(names, label),
            nm(names, top[0][0]), top[0][1], nm(names, top[1][0]), top[1][1], nm(names, top[2][0]), top[2][1],
            fps, loop_us, T_LW[0], t_lw, time.ticks_ms()))
        k += 1
        if k % 100 == 0:
            step("S k=%d taken=%d bad=%d err=%d last_err=%s | last %d loops mean us: loop %d poll %d lw+overlay %d" % (
                k, LW["taken"], LW["bad"], LW["err"], LW["last_err"], nacc, acc[0] // max(1, nacc),
                acc[1] // max(1, nacc), acc[2] // max(1, nacc)))
            acc[0] = 0; acc[1] = 0; acc[2] = 0; nacc = 0
            t_prev = None                            # the summary line (with its pause) is not counted as loop time
        wdt.feed()
    step("LIVE DONE frames=%d taken=%d bad=%d err=%d last_err=%s" % (k, LW["taken"], LW["bad"], LW["err"], LW["last_err"]))
