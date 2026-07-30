#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "$ROOT/outputs/preflight"

"$VENV/bin/python" -m actionstream.libero_config \
  --config-dir "$LIBERO_CONFIG_PATH"

"$VENV/bin/actionstream-preflight" \
  --output "$ROOT/outputs/preflight/preflight.json" \
  "$@" 2>&1 | tee "$ROOT/outputs/preflight/preflight.log"
