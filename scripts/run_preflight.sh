#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
OUTPUT_ROOT="${ACTIONSTREAM_OUTPUT_ROOT:-$ROOT/outputs}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "$OUTPUT_ROOT/preflight"

"$VENV/bin/python" -m actionstream.libero_config \
  --config-dir "$LIBERO_CONFIG_PATH"

"$VENV/bin/actionstream-preflight" \
  --output "$OUTPUT_ROOT/preflight/preflight.json" \
  "$@" 2>&1 | tee "$OUTPUT_ROOT/preflight/preflight.log"
