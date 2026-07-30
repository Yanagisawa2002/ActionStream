#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
UV="${ACTIONSTREAM_UV:-$HOME/.local/bin/uv}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"

if [[ -x "$VENV/bin/python" ]]; then
  "$VENV/bin/python" -c \
    'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
else
  "$UV" venv --python /usr/bin/python3.12 "$VENV"
fi
"$UV" pip install \
  --python "$VENV/bin/python" \
  "lerobot[xvla,libero,evaluation]==0.6.0"
"$UV" pip install --python "$VENV/bin/python" --no-deps --editable "$ROOT"

LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
  "$VENV/bin/python" -m actionstream.libero_config \
  --config-dir "$LIBERO_CONFIG_PATH"

"$UV" pip check --python "$VENV/bin/python"

printf 'ActionStream environment ready.\n'
printf 'Venv: %s\n' "$VENV"
printf 'LIBERO_CONFIG_PATH: %s\n' "$LIBERO_CONFIG_PATH"
