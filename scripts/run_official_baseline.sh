#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

mkdir -p "$ROOT/outputs/logs"
"$VENV/bin/python" -m actionstream.libero_config --config-dir "$LIBERO_CONFIG_PATH"

"$VENV/bin/lerobot-eval" \
  --policy.path=lerobot/xvla-libero \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0,1,2]' \
  --env.control_mode=absolute \
  --env.episode_length=800 \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --seed=142 \
  --output_dir="$ROOT/outputs/xvla_baseline" \
  2>&1 | tee "$ROOT/outputs/logs/xvla_baseline.log"
