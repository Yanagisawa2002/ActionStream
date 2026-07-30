#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

"$VENV/bin/python" -m actionstream.libero_config --config-dir "$LIBERO_CONFIG_PATH"

read -r -a modes <<< "${ACTIONSTREAM_MODES:-sync async_naive async_aligned}"
read -r -a delays <<< "${ACTIONSTREAM_DELAYS_MS:-0 200}"

for delay_ms in "${delays[@]}"; do
  for mode in "${modes[@]}"; do
    "$VENV/bin/python" -m actionstream.benchmark \
      --mode="$mode" \
      --task-ids=0,1,2 \
      --episodes-per-task=10 \
      --initial-state-indices=0,2,4,6,8,10,12,14,16,18 \
      --seed=142 \
      --episode-length=800 \
      --injected-delay-ms="$delay_ms" \
      --realtime \
      --replan-interval-steps=10 \
      --output="$ROOT/outputs/m3/${mode}_delay${delay_ms}/episodes.jsonl"
  done
done
