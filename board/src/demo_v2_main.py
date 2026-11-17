# ---------------------------------------------------------------- demo v2 main loop (overnight 2)
# Output lines:
#   G <frame> raw <word> <cos> m <margin> | ema <word> <cos> m <margin> | show <word or ?> | <fps> fps loop <us>
#     cap <us> pred <us> fill <us> rs <us> sc <us>+<us> ov <us> pr <us> crc 0x<output crc>
#     raw  = this frame alone (what the known-good demo shows), ema = after smoothing, show = what v2 displays
#     loop = time of the previous whole loop, pr = time of the previous print
#   S ... every 100 frames: counters kept on the board (exact even if the terminal loses a line)
def summary(k, acc, nacc):
    step("S k=%d raw_changes=%d ema_changes=%d show_changes=%d unsure=%d | last %d loops mean us: loop %d cap %d pred %d fill %d sc %d ov %d pr %d" % (
        k, CNT[1], CNT[2], CNT[3], CNT[4], nacc, acc[0] // max(1, nacc), acc[1] // max(1, nacc), acc[2] // max(1, nacc),
        acc[3] // max(1, nacc), acc[4] // max(1, nacc), acc[5] // max(1, nacc), acc[6] // max(1, nacc)))


def main():
    start_wait()
    step("V2 demo: model=%s text=%s ALPHA=%.2f THR_ON=%.4f THR_OFF=%.4f MAX_FRAMES=%d PROFILE=%d (Stop: the WDT resets the board ~10 s later, expected)" % (
        MODEL_NAME, TEXT_SOURCE, ALPHA, THR_ON, THR_OFF, MAX_FRAMES, PROFILE))
    names, rows, norms = load_text()
    setup_text_ulab(norms)
    del rows
    gc.collect()
    cam = open_camera()
    m = open_model()                                 # alignment check BEFORE the first predict
    if PROFILE:
        SOURCE[0] = timed_src
    k = 0
    t_prev = None
    fps = 0.0                                        # full-loop rate of the previous iteration (incl. drawing/print)
    loop_us = 0; t_pr = 0
    acc = [0, 0, 0, 0, 0, 0, 0]; nacc = 0
    while MAX_FRAMES == 0 or k < MAX_FRAMES:
        wdt.feed()
        t_now = time.ticks_us()
        if t_prev is not None:
            loop_us = time.ticks_diff(t_now, t_prev)
            fps = 1000000.0 / max(1, loop_us)
        else:
            loop_us = 0
        t_prev = t_now
        img = cam.snapshot()
        t_cap = time.ticks_diff(time.ticks_us(), t_now)
        set_source(img)
        if k == 0:
            check_frame(img)
            step("C frame %dx%d roi=%r" % (img.width(), img.height(), ROI[0]))
        ob, r1, rc, rm, top, label, margin = process(m)   # wdt.feed() before and after the predict inside
        t0 = time.ticks_us()
        overlay_v2(img, names, top, label, fps)
        t_ov = time.ticks_diff(time.ticks_us(), t0)
        if loop_us and k > 1:
            acc[0] += loop_us; acc[1] += t_cap; acc[2] += T_PRED[0]; acc[3] += T_FILL[0]
            acc[4] += T_SC[0] + T_SC[1]; acc[5] += t_ov; acc[6] += t_pr; nacc += 1
        t0 = time.ticks_us()
        if k % PRINT_EVERY == 0:
            print("G %d raw %s %.4f m %.4f | ema %s %.4f m %.4f | show %s | %.2f fps loop %d cap %d pred %d fill %d rs %d sc %d+%d ov %d pr %d crc 0x%08x" % (
                k, nm(names, r1), rc, rm, nm(names, top[0][0]), top[0][1], margin, nm(names, label), fps, loop_us, t_cap,
                T_PRED[0], T_FILL[0], T_RS[0], T_SC[0], T_SC[1], t_ov, t_pr, crc(ob)))
        t_pr = time.ticks_diff(time.ticks_us(), t0)
        k += 1
        if k % 100 == 0:
            summary(k, acc, nacc)
            for i in range(len(acc)):
                acc[i] = 0
            nacc = 0
            t_prev = None                            # the summary line (with its pause) is not counted as loop time
        wdt.feed()
    step("V2 DONE frames=%d" % k)
