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
  --task-ids=0,1,2 \
  --episodes-per-task=10 \
  --initial-state-indices=0,2,4,6,8,10,12,14,16,18 \
  --seed=142 \
  --episode-length=800 \
  --injected-delay-ms=0 \
  --output="$OUTPUT_ROOT/custom_sync_parity/episodes.jsonl"

ACTIONSTREAM_GATE_ROOT="$OUTPUT_ROOT" \
  "$VENV/bin/python" -c '
import os
from pathlib import Path
from actionstream.results import write_custom_parity_manifest
root = Path(os.environ["ACTIONSTREAM_GATE_ROOT"])
write_custom_parity_manifest(
    root / "xvla_baseline" / "manifest.json",
    root / "custom_sync_parity" / "episodes.jsonl",
    root / "custom_sync_parity" / "manifest.json",
)
'
