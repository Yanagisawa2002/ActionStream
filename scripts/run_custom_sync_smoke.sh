#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
OUTPUT_ROOT="${ACTIONSTREAM_OUTPUT_ROOT:-$ROOT/outputs}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

"$VENV/bin/python" -m actionstream.libero_config --config-dir "$LIBERO_CONFIG_PATH"
"$VENV/bin/python" -m actionstream.benchmark \
  --mode=sync \
  --task-ids=0 \
  --episodes-per-task=3 \
  --initial-state-indices=0,2,4 \
  --seed=142 \
  --episode-length=800 \
  --injected-delay-ms=0 \
  --output="$OUTPUT_ROOT/custom_sync_smoke/episodes.jsonl"
