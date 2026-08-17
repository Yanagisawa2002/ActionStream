#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
UV="${ACTIONSTREAM_UV:-$HOME/.local/bin/uv}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"

# AutoDL/SeetaCloud provides this opt-in network accelerator. Source it before
# uv performs any download, while keeping the script portable elsewhere.
if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

if [[ -x "$VENV/bin/python" ]]; then
  "$VENV/bin/python" -c \
    'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
else
  "$UV" venv --python /usr/bin/python3.12 "$VENV"
fi
UV_PROJECT_ENVIRONMENT="$VENV" \
  "$UV" sync --project "$ROOT" --locked --no-dev

LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
  "$VENV/bin/python" -m actionstream.libero_config \
  --config-dir "$LIBERO_CONFIG_PATH"

"$UV" pip check --python "$VENV/bin/python"

printf 'ActionStream environment ready.\n'
printf 'Venv: %s\n' "$VENV"
printf 'LIBERO_CONFIG_PATH: %s\n' "$LIBERO_CONFIG_PATH"
