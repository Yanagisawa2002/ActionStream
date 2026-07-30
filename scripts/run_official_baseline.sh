#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
OUTPUT_ROOT="${ACTIONSTREAM_OUTPUT_ROOT:-$ROOT/outputs}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

mkdir -p "$OUTPUT_ROOT/logs"
"$VENV/bin/python" -m actionstream.libero_config --config-dir "$LIBERO_CONFIG_PATH"

"$VENV/bin/lerobot-eval" \
  --policy.path=lerobot/xvla-libero \
  --policy.device=cuda \
  --policy.pretrained_revision=12e8783e996944f5c97e490d37d4c145484ed70a \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0,1,2]' \
  --env.control_mode=absolute \
  --env.episode_length=800 \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --seed=142 \
  --output_dir="$OUTPUT_ROOT/xvla_baseline" \
  2>&1 | tee "$OUTPUT_ROOT/logs/xvla_baseline.log"

GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
ACTIONSTREAM_GATE_ROOT="$OUTPUT_ROOT" \
ACTIONSTREAM_GATE_COMMIT="$GIT_COMMIT" \
  "$VENV/bin/python" -c '
import os
from pathlib import Path
from actionstream.results import (
    EXPECTED_MODEL_REVISION,
    audit_official_log,
    write_official_manifest,
)
root = Path(os.environ["ACTIONSTREAM_GATE_ROOT"])
log = root / "logs" / "xvla_baseline.log"
write_official_manifest(
    root / "xvla_baseline" / "eval_info.json",
    root / "xvla_baseline" / "manifest.json",
    model_revision_sha=EXPECTED_MODEL_REVISION,
    git_commit=os.environ["ACTIONSTREAM_GATE_COMMIT"],
    command="bash scripts/run_official_baseline.sh",
    contract_audit=audit_official_log(log, exit_code=0),
)
'
