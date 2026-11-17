#!/usr/bin/env bash
# Compile an int8 ONNX model for the STM32N6 NPU as an OpenMV relocatable model.
# usage: bash pipeline/05_compile/compile.sh <model.onnx> <out_dir>
#   -> <out_dir>/gout/network_rel.bin (the file that goes into ROMFS), <out_dir>/gws (compiler workspace),
#      <out_dir>/compile.log
# Needs:
#   STEDGEAI     path to stedgeai(.exe) from ST Edge AI Core (tested: v4.0.0-20500, the copy bundled with OpenMV IDE)
#   ARM_GCC_BIN  folder with arm-none-eabi-gcc (needed for the --relocatable link step; OpenMV IDE bundles one)
#   stm32n6.mpool next to neuralart_mci0.json: take it unchanged from OpenMV, lib/stai/scripts/stm32n6.mpool
set -eu
M="$1"; B="$2"; mkdir -p "$B"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="${ARM_GCC_BIN:?set ARM_GCC_BIN}:$PATH"
cd "$HERE"
set +e
"${STEDGEAI:?set STEDGEAI}" generate -m "$M" --target stm32n6 \
  --st-neural-art softmax_gelu@neuralart_mci0.json --relocatable -w "$B/gws" -o "$B/gout" > "$B/compile.log" 2>&1
RC=$?
set -e
[ -f "$B/gws/neural_art__network/c_info.json" ] && cp "$B/gws/neural_art__network/c_info.json" "$B/gout/c_info.json"
echo "compile $M -> $B exit $RC"
exit $RC
