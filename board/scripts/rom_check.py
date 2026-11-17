import machine
wdt = machine.WDT(timeout=10000)
# Final rom_check for ROMFS state 'w3u_splitdw4' (night/board_val, 8.11.2026). WDT 10 s first. Read-only.
# Same model-file checks as night/final/board/rc_w3u_splitdw4.py (EXP copied from it), plus a full listing of
# EVERY file in /rom (including .txt, .cascade and the pad) with its flash address, mod8 and size.
import os, uctypes, time
EXP = {"blazeface_front_128.tflite": 0x708a4fa0, "face_landmarks_192.tflite": 0x708dac20, "flashbw_1mb": 0x70e27e10, "fomo_face_detection.tflite": 0x70800040, "force_int_quant.tflite": 0x7080fae0, "mci0_256_w3u_splitdw4": 0x70f2b1c0, "micro_speech.tflite": 0x70b12d40, "palm_detection_full_192.tflite": 0x70998600, "person_detect.tflite": 0x708145c0, "yolo_lc_192.tflite": 0x70857580, "yolov8n_192_cut.tflite": 0x70b1b600}
bad = []
for _ in range(6):
    time.sleep_ms(500); wdt.feed()
print('RCF state=w3u_splitdw4')
names = sorted(os.listdir('/rom'))
for n in names:
    f = open('/rom/' + n, 'rb'); a = uctypes.addressof(f) & 0xFFFFFFFF; f.close()
    sz = os.stat('/rom/' + n)[6]
    model = not (n.endswith('.txt') or n.endswith('.cascade') or n.startswith('mci0_pad'))
    st = '-'
    if model:
        e = EXP.get(n, -1); ok = (a == e) and (a % 8 == 0); st = 'OK' if ok else 'BAD exp=0x%08x' % (e & 0xFFFFFFFF)
        if not ok: bad.append(n)
    print('F %-32s 0x%08x mod8=%d %9d %s' % (n, a, a % 8, sz, st))
    time.sleep_ms(120); wdt.feed()
for n in EXP:
    if n not in names: bad.append(n + '(missing)')
print('FILES: %d' % len(names))
print('ROM CHECK: ' + ('ALL OK' if not bad else 'STOP ' + ' '.join(bad)))
