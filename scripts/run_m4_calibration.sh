#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
OUTPUT_ROOT="${ACTIONSTREAM_OUTPUT_ROOT:-$ROOT/outputs}"
M4_ROOT="$OUTPUT_ROOT/m4"
PLAN="$M4_ROOT/calibration/calibration_plan.json"
SELECTION="$M4_ROOT/calibration/selection.json"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

"$VENV/bin/python" -m actionstream.libero_config --config-dir "$LIBERO_CONFIG_PATH"

"$VENV/bin/python" -m actionstream.m4_calibration plan \
  "$OUTPUT_ROOT"/m3/*/episodes.jsonl \
  --m3-manifest "$OUTPUT_ROOT/summary/manifest.json" \
  --output "$PLAN"

read -r -a candidate_delays <<< "$(
  "$VENV/bin/python" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print(" ".join(str(x["injected_delay_ms"]) for x in p["candidate_derivation"]["initial_candidates"]))' \
    "$PLAN"
)"

run_delay() {
  local delay_ms="$1"
  local mode
  for mode in async_naive async_aligned; do
    "$VENV/bin/python" -m actionstream.benchmark \
      --mode="$mode" \
      --task-ids=3 \
      --episodes-per-task=3 \
      --initial-state-indices=0,2,4 \
      --seed=142 \
      --episode-length=800 \
      --injected-delay-ms="$delay_ms" \
      --realtime \
      --replan-interval-steps=10 \
      --output="$M4_ROOT/calibration/runs/${mode}_delay${delay_ms}/episodes.jsonl"
  done
}

for delay_ms in "${candidate_delays[@]}"; do
  run_delay "$delay_ms"
done

"$VENV/bin/python" -m actionstream.m4_calibration select \
  "$M4_ROOT"/calibration/runs/*/episodes.jsonl \
  --plan "$PLAN" \
  --output "$SELECTION"

selection_status="$(
  "$VENV/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$SELECTION"
)"
if [[ "$selection_status" == "needs_extension" ]]; then
  extension_delay="$(
    "$VENV/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["extension_delay_ms"])' \
      "$SELECTION"
  )"
  run_delay "$extension_delay"
  "$VENV/bin/python" -m actionstream.m4_calibration select \
    "$M4_ROOT"/calibration/runs/*/episodes.jsonl \
    --plan "$PLAN" \
    --output "$SELECTION"
fi
