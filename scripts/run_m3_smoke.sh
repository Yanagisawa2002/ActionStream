#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
OUTPUT_ROOT="${ACTIONSTREAM_OUTPUT_ROOT:-$ROOT/outputs}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

"$VENV/bin/python" -m actionstream.libero_config --config-dir "$LIBERO_CONFIG_PATH"

for delay_ms in 0 200; do
  for mode in sync async_naive async_aligned; do
    "$VENV/bin/python" -m actionstream.benchmark \
      --mode="$mode" \
      --task-ids=0 \
      --episodes-per-task=2 \
      --initial-state-indices=0,2 \
      --seed=142 \
      --episode-length=800 \
      --injected-delay-ms="$delay_ms" \
      --realtime \
      --replan-interval-steps=10 \
      --output="$OUTPUT_ROOT/m3_smoke/${mode}_delay${delay_ms}/episodes.jsonl"
  done
done
