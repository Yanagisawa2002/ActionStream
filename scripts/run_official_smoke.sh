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
  --env.task_ids='[0]' \
  --env.control_mode=absolute \
  --env.episode_length=800 \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --seed=142 \
  --output_dir="$OUTPUT_ROOT/xvla_smoke" \
  2>&1 | tee "$OUTPUT_ROOT/logs/xvla_smoke.log"
