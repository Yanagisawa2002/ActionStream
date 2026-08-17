#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
OUTPUT_ROOT="${ACTIONSTREAM_CAPTURE_ROOT:-$ROOT/outputs/m4_paired_capture/task0_seed144_state4}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export ACTIONSTREAM_SOURCE_COMMIT="${ACTIONSTREAM_SOURCE_COMMIT:-d84cb64e9e48b681e083828b24df5a38709c78c8}"

"$VENV/bin/python" -m actionstream.libero_config --config-dir "$LIBERO_CONFIG_PATH"

"$VENV/bin/python" -m actionstream.m4_capture capture \
  --mode async_naive \
  --task-id 0 \
  --episode-index 2 \
  --initial-state-index 4 \
  --base-seed 142 \
  --episode-seed 144 \
  --injected-delay-ms 950 \
  --episode-length 800 \
  --replan-interval-steps 10 \
  --expected-success false \
  --expected-steps 800 \
  --run-id m4-capture-task0-seed144-state4-naive \
  --output-dir "$OUTPUT_ROOT/before_naive"

"$VENV/bin/python" -m actionstream.m4_capture capture \
  --mode async_aligned \
  --task-id 0 \
  --episode-index 2 \
  --initial-state-index 4 \
  --base-seed 142 \
  --episode-seed 144 \
  --injected-delay-ms 950 \
  --episode-length 800 \
  --replan-interval-steps 10 \
  --expected-success true \
  --expected-steps 150 \
  --run-id m4-capture-task0-seed144-state4-aligned \
  --output-dir "$OUTPUT_ROOT/after_aligned"

"$VENV/bin/python" -m actionstream.m4_capture compose \
  --before-receipt "$OUTPUT_ROOT/before_naive/async_naive.receipt.json" \
  --after-receipt "$OUTPUT_ROOT/after_aligned/async_aligned.receipt.json" \
  --output "$OUTPUT_ROOT/m4_task0_seed144_state4_paired.mp4" \
  --receipt-output "$OUTPUT_ROOT/m4_task0_seed144_state4_paired.receipt.json"
