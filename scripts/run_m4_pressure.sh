#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
OUTPUT_ROOT="${ACTIONSTREAM_OUTPUT_ROOT:-$ROOT/outputs}"
M4_ROOT="$OUTPUT_ROOT/m4"
SELECTION="$M4_ROOT/calibration/selection.json"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

if [[ ! -f "$SELECTION" ]]; then
  printf 'Missing frozen pressure selection: %s\n' "$SELECTION" >&2
  exit 1
fi

read -r selection_status pressure_delay <<< "$(
  "$VENV/bin/python" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print(p["status"], p["selected_pressure_delay_ms"])' \
    "$SELECTION"
)"
if [[ "$selection_status" != "selected" || "$pressure_delay" == "None" ]]; then
  printf 'Calibration did not select a pressure point: %s\n' "$selection_status" >&2
  exit 1
fi

"$VENV/bin/python" -m actionstream.libero_config --config-dir "$LIBERO_CONFIG_PATH"

for mode in sync_hold async_naive async_aligned; do
  "$VENV/bin/python" -m actionstream.benchmark \
    --mode="$mode" \
    --task-ids=0,1,2 \
    --episodes-per-task=10 \
    --initial-state-indices=0,2,4,6,8,10,12,14,16,18 \
    --seed=142 \
    --episode-length=800 \
    --injected-delay-ms="$pressure_delay" \
    --realtime \
    --replan-interval-steps=10 \
    --output="$M4_ROOT/pressure/${mode}_delay${pressure_delay}/episodes.jsonl"
done
