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
  "$VENV/bin/python" -m actionstream.benchmark \
    --mode=sync_hold \
    --task-ids=0,1,2 \
    --episodes-per-task=10 \
    --initial-state-indices=0,2,4,6,8,10,12,14,16,18 \
    --seed=142 \
    --episode-length=800 \
    --injected-delay-ms="$delay_ms" \
    --realtime \
    --replan-interval-steps=10 \
    --output="$OUTPUT_ROOT/m4/sync_hold/sync_hold_delay${delay_ms}/episodes.jsonl"
done
