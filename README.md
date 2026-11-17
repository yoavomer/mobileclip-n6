# MobileCLIP-S0 on the OpenMV N6: open-vocabulary recognition on a microcontroller NPU

Point a camera at an object and it shows the word that matches it. To add a new object, you type a word; there is no
retraining. This repo ports the image encoder of Apple's **MobileCLIP-S0** to the **OpenMV N6** camera board
(STM32N657, Arm Cortex-M55 + ST Neural-ART NPU), running int8 on the NPU at about **5.5 frames per second**.

**Videos:**
- [`media/add_word_ball.mp4`](media/add_word_ball.mp4): adding a word live.
  - The board knows 43 words, and for a ball it shows "? vase".
  - "ball" is typed on the PC. It joins the word list and is recognized: "ball 100%", "44 words (+1)".
- [`media/demo_43_words.mp4`](media/demo_43_words.mp4): the 43-word demo, with the live top-3 words on the frame
  buffer.

Numbers are tagged: **[measured]** = measured on the board or the PC, **[estimate]** = computed from a model of the
hardware, not measured.

---

## The starting point: the model did not run

The FP32 model was exported from PyTorch to ONNX, quantized to int8 with ONNX Runtime's default recipe (MinMax, 24
photos), and compiled with ST's `stedgeai` for the NPU. On the board:

1. `predict()` **hung** and never returned.
2. Once it ran, the output was **garbage**: cosine **0.113** with the PC's output for the same model and the same
   input bytes. A correct port gives about 0.95. 124 of the 512 output values were stuck at −128 / 127 [measured].
3. Even after the board matched the PC, the int8 model was **useless**: 3.6 % zero-shot top-1, against 59.4 % in FP32
   [measured].

## The 8 problems

| # | Problem | Where the cause was | Fix | Effect |
|---|---|---|---|---|
| 1 | `predict()` hangs | The attention's qkv projection (a MatMul/Gemm) as compiled by `stedgeai` for the N6. The exact mechanism inside the runtime was not isolated. | Rewrite the qkv MatMul as the equivalent 1×1 convolution ([`make_full_qkvconv.py`](pipeline/01_quantize/make_full_qkvconv.py)) | The full model runs and `predict()` returns [measured] |
| 2 | Board output unrelated to the PC's (cosine 0.113) | ST's **software convolution**, used for the attention matrix products, does not subtract the zero point of the operand passed as "weights". Proven by a formula that reproduces **100 % of 32,768** board values, 0 difference. | **A2 rule:** give those operands zero point 0 (symmetric). 8 numbers in the model change. ([`recreate_a2.py`](pipeline/01_quantize/recreate_a2.py)) | Board vs PC: cosine **0.113 → 0.928** [measured] |
| 3 | Accuracy 3.6 % top-1 | Quantization: MinMax calibration lets rare outliers set the range, so int8 steps are about 6× too coarse | Percentile 99.95 calibration, chosen by a sweep (99.5 halves the score; 99 collapses) | Top-1 **3.6 % → 42.6 %** [measured, locked TEST set, board-like emulation] |
| 4 | Most of the remaining loss is in the stem (the first 3 conv blocks) | The NPU runs conv → round → **one exact 8-bit GELU table** → round. The ONNX graph rounds 4 times (Div, Erf, Add, Mul). Ranges were tuned for the wrong graph. INT16 was not an option: the compiler refuses it there. | GELU-aware stem ranges: pick each GELU **input** range to minimize the error **after** the GELU. Needed a **board-like PC emulator** that does what the NPU does. | **+8.6 pp → 49.8 %** top-1 [measured, SELECTION set, board-like] |
| 5 | Systematic per-channel error | Rounding int8 weights shifts each output channel's mean | Bias correction: put the average error back into the int32 bias (400 images) | **+3.0 to +4.5 pp** [measured, SELECTION] |
| 6 | Board and emulator still differ (cosine 0.932) | The compiler's GELU tables are **±1 step off** on 73 input codes (49 of them in the stem). The error is systematic, not noise. | Table-aware ranges for the 27 GELU chains after the stem, chosen with probe compiles; the emulator uses the real compiled tables | Top-1 **+1.8 pp → 54.8 %** = **92 % of FP32** [measured, locked TEST set]. Board vs emulator: cosine **0.932 → 0.982** on 50 photos [measured] |
| 7 | 318 ms per image | The first stem block writes a 1 MB tensor to external hyperRAM. The next depthwise conv reads it back **one byte per 64-byte line**: 1,048,576 single-byte reads, about 0.16 µs each (about 169 ms) [estimate, fitted to 16 board timings] | **splitdw4:** split the depthwise conv into 4 groups of 16 channels. It is the same math, and the compiler then keeps the data in internal SRAM. ([`stem_splitdw.py`](pipeline/04_split/stem_splitdw.py)) | **318.4 ms → 145.7 ms (2.19×)**, bit-identical output (same CRC on a fixed input and on 50/50 photos) [measured] |
| 8 | The same model file hangs or runs depending on where it lands in flash | OpenMV's ROMFS editor aligns files to 4 bytes. Neural-ART relocatable models need 8. The loader's check (`& 0x3`) lets a 4-mod-8 address through. | Place the model at an 8-byte-aligned address and check it on the board before running ([`romfs_align.py`](pipeline/06_romfs/romfs_align.py), [`rom_check.py`](board/scripts/rom_check.py)) | The byte-identical file runs at an address ≡ 0 (mod 8) and hangs at ≡ 4 (mod 8) [measured] |

How each claim was checked:
- **Board vs PC:** compared byte for byte with CRC32 on fixed inputs.
- **The choice:** the final model was chosen on a 3,000-image SELECTION set (ImageNet).
- **The final number:** the 3,000-image TEST set (ImageNetV2, 3 per class) was **locked** and opened once, after the
  choice was frozen. An independent check recomputed every number from per-image files.
- **No retraining:** all weights are Apple's. Only quantization ranges, int8 biases and graph form changed.

## Architecture: what runs where

```
                PC (Python, PyTorch)                        OpenMV N6 board
  ┌───────────────────────────────────┐        ┌───────────────────────────────────────────────┐
  │ MobileCLIP-S0 TEXT encoder (FP32) │        │ Cortex-M55 CPU (MicroPython)                  │
  │  word -> 80 prompt templates      │ USB    │  camera frame 320x200 -> centre crop          │
  │  -> 512-d vector (normalized)     ├───────►│  -> 256x256 -> int8                    ~26 ms │
  │  written to E:/liveword.bin       │ drive  │  cosine vs word list (ulab), smoothing,       │
  │  (pc/live_word.py)                │        │  "?" when unsure, top-3, overlay        ~6 ms │
  └───────────────────────────────────┘        │  + 64 SW and 18 hybrid epochs of the model    │
                                               │    (attention matmuls, softmax)               │
                                               ├───────────────────────────────────────────────┤
                                               │ Neural-ART NPU: MobileCLIP-S0 IMAGE encoder   │
                                               │  int8, 218 hardware epochs; whole predict     │
                                               │  145.7 ms                                     │
                                               │  weights 11.5 MB in octo-flash (XIP)          │
                                               │  activations: 1.6 MB internal SRAM            │
                                               │               + 1.5 MB external hyperRAM      │
                                               └───────────────────────────────────────────────┘
```

- **NPU:** the image encoder (MCi0, 256×256 input, 512-d output), compiled by `stedgeai`.
  - Of 300 epochs, 218 are pure hardware.
  - The 64 software and 18 hybrid epochs, mostly attention and softmax, run on the Cortex-M55.
- **Memory:**
  - OpenMV gives the NPU all four NPU SRAM banks (4 × 448 KB) during `predict()`.
  - The peak of live activations is 2 MB, in the high-resolution stem (128×128 and 64×64). So 1.5 MB spills to
    hyperRAM in the first 23 of 300 epochs.
  - From then on, everything stays internal [from the compiler report].
- **CPU, per frame** (180.8 ms total, 5.53 FPS) [measured]:

  | step | ms |
  |---|---|
  | capture | 1.7 |
  | resize | 23.1 |
  | int8 conversion | 1.1 |
  | predict | 148.8 |
  | scoring | 1.3 |
  | top-3 and smoothing | 1.8 |
  | overlay | 2.5 |
  | rest | 0.6 |

- **PC:** the text encoder runs only when a new word is typed.
  - An exact vector (80 templates) reaches the board in about **2.6 s**.
  - With `--quick-first`, a one-template vector arrives in about **0.47 s** and is then replaced by the exact one
    [measured].
  - The board only reads the file. A new word becomes one more row of the word matrix, so it is ranked, smoothed and
    shown like the 43 built-in words.

## Repository layout

| folder | what is inside |
|---|---|
| `pipeline/00_export` | PyTorch checkpoint → FP32 ONNX (image encoder, 256×256) |
| `pipeline/01_quantize` | Percentile 99.95 quantization, the qkv → 1×1 conv rewrite, the A2 zero-point rule |
| `pipeline/02_stem_ranges_bias` | Bias correction, GELU-aware stem ranges, table-aware ranges; `specs/` = the exact ranges used |
| `pipeline/03_gelu_ranges` | GELU-chain ranges for the rest of the network (W3t, W3u); `specs/` = the exact ranges used |
| `pipeline/04_split` | The stem depthwise split (splitdw4) |
| `pipeline/05_compile` | The `stedgeai` command, the Neural-ART profile, compiled-table extraction |
| `pipeline/06_romfs` | ROMFS listing and 8-byte alignment tools |
| `emulator/` | The board-like PC emulator (fused GELU tables, integer softmax, board habits) and its validation scripts |
| `eval/` | Zero-shot ImageNet evaluation (80 templates), data-split builder |
| `text/` | Text-encoder scripts, the 43-word vocabulary and the board text-file builder |
| `board/src`, `board/build` | MicroPython sources and the builders that assemble the single-file board scripts |
| `board/scripts` | Ready-to-run board scripts: `rom_check.py`, `demo_v2.py` (43 words), `vocab_add_demo.py` (add words live), `live_demo.py` |
| `pc/live_word.py` | Type a word on the PC; it is encoded and sent to the board |
| `media/` | Demo videos |

The code is the research code as it was run. Paths that pointed at my own disk are replaced by the `MCLIP_ROOT`
environment variable (the working folder; default: the current folder). Some scripts still expect the working-folder
layout of the original experiments, and their docstrings say which inputs they need.

## How to reproduce

### 1. Install (PC)
```bash
python -m venv .venv && . .venv/Scripts/activate      # Linux/macOS: . .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/apple/ml-mobileclip && pip install -e ml-mobileclip
```
Tools for the board side:
- OpenMV IDE, with firmware 5.0.1 on the N6.
- ST Edge AI Core v4.0.0 (`stedgeai`). The copy bundled with OpenMV IDE works.

### 2. Download the weights (not in this repo)
```bash
mkdir -p checkpoints
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt -P checkpoints
```

### 3. Convert (PC)
Set `MCLIP_ROOT` to your working folder. Then, in order:

| step | script | result |
|---|---|---|
| export | `pipeline/00_export/export_mci0.py` | FP32 ONNX of the image encoder |
| quantize | `pipeline/01_quantize/quantize_mci0.py`, then `build_variant.py P99_95_24 Percentile calib_inputs.npy 99.95` | int8 QDQ model with the qkv conv rewrite and the A2 rule |
| bias + stem ranges | `pipeline/02_stem_ranges_bias/scripts/bias_corr.py`, `build_ranges.py ... specs/spec_S5r.json`, `transplant_bias.py` | model "S5" |
| GELU ranges | `pipeline/03_gelu_ranges/scripts/build_cand.py <S5> <out> @specs/spec_cl2t_27.txt`, then `@specs/spec_inner_fix.txt` | model "W3u" |
| split | `pipeline/04_split/stem_splitdw.py <W3u> <out> 4` | `W3u_..._splitdw4.onnx` |
| compile | `STEDGEAI=... ARM_GCC_BIN=... bash pipeline/05_compile/compile.sh <model.onnx> <out>` | `<out>/gout/network_rel.bin` |

The chosen ranges are saved in the `specs/` files. Rebuilding does **not** need the probe compiles that found them.

Calibration needs a few dozen photos. Evaluation needs ImageNet (validation) and ImageNetV2; `eval/build_data.py`
builds the arrays.

### 4. Run on the board
1. Put `network_rel.bin` into the board's ROMFS as `mci0_256_w3u_splitdw4`, at an **8-byte-aligned** address
   (OpenMV IDE → Tools → ROM File System; check with `pipeline/06_romfs/romfs_ls.py`).
2. In OpenMV IDE, run `board/scripts/rom_check.py`. It must print `ROM CHECK: ALL OK`.
3. Run `board/scripts/demo_v2.py` for the 43-word demo, or `board/scripts/vocab_add_demo.py` to add words live.
4. For live words, on the PC: `python -I pc/live_word.py`. Wait for `word>`, type a word and press Enter.
   - On a new board, create the file once with `--create <word>` while no script is running.

Every board script starts a 10 s watchdog, so the board resets about 10 s after you press Stop.

## Limitations

- **Accuracy:** 54.8 % zero-shot ImageNet top-1, which is 92 % of FP32. On a cluttered desk the top three words are
  often within 0.005 cosine. The demo therefore smooths over about 9 frames and shows "?" when the lead is small.
- **Model limits:** some failures are the model's, not the board's. A book seen edge-on reads as "desk", and a mirror
  as the TV it reflects. FP32 makes the same choices.
- **Software epochs:** 64 software epochs still run on the CPU. Moving the attention matmuls fully onto the NPU
  depends on the zero-point fix in ST's software convolution.
- **Two vendor bugs** are worked around, not fixed: ST's ignored zero point (problem 2) and OpenMV's 4-byte ROMFS
  alignment (problem 8).
- **Board specifics:** tested only on one OpenMV N6 with firmware 5.0.1 and ST Edge AI Core v4.0.0. There is no
  current or power API on the board, so power was not measured.

## What's next

- **MobileCLIP2-S0:** +4.4 pp top-1 in FP32 with the same building blocks. It has not been tried in int8 yet.
- **224×224 input:** about 104–110 ms per image, for about −2.9 pp accuracy [estimate].
- **Faster demo code:** move scoring and resize out of Python. The model is 82 % of the frame time.
- **Vendor fixes:** report the two bugs upstream, so the workarounds can go away.

## License

- **This repo's code:** MIT ([LICENSE](LICENSE)).
- **Not covered by MIT:**
  - Apple's MobileCLIP code and weights stay under Apple's licenses (see
    [apple/ml-mobileclip](https://github.com/apple/ml-mobileclip)).
  - ST Edge AI Core and every file it generates stay under STMicroelectronics' licenses.
  - OpenMV firmware and IDE files stay under OpenMV's licenses.
- **Not in this repo:** no weights, no converted models and no compiler outputs.
- **The one derived file:** the 43 word vectors embedded in `board/scripts/demo_v2.py` were computed with Apple's
  text encoder.
