#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
OUTPUT_ROOT="${ACTIONSTREAM_OUTPUT_ROOT:-$ROOT/outputs}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
SMOKE_OUTPUT="$OUTPUT_ROOT/m5_g0/audit/runtime_smoke_formal_ready"
resume_args=()
if [[ -f "$SMOKE_OUTPUT/run_manifest.json" ]]; then
  resume_args=(--resume-after-crash)
elif [[ -e "$SMOKE_OUTPUT" ]]; then
  printf 'Refusing untracked smoke output: %s\n' "$SMOKE_OUTPUT" >&2
  exit 1
fi

"$VENV/bin/python" -m actionstream.libero_config --config-dir "$LIBERO_CONFIG_PATH"
"$VENV/bin/python" -m actionstream.m5_benchmark \
  --phase=audit \
  --task-id=0 \
  --displacement-mm=50 \
  --seed-manifest="$ROOT/outputs/m5_g0/audit/runtime_smoke_seed_manifest.json" \
  --config="$ROOT/configs/m5_g0.json" \
  --task-audit="$ROOT/outputs/m5_g0/audit/task_entity_audit.json" \
  --output-dir="$SMOKE_OUTPUT" \
  "${resume_args[@]}"
