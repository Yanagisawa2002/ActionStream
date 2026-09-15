#!/usr/bin/env bash
# Launch in an isolated task directory. This never shuts down the machine.
set -euo pipefail
BASE=${1:?Provide the isolated task directory}
exec 9>"$BASE/pipeline.lock"
flock -n 9 || { echo 'This pipeline is already running'; exit 1; }
if [[ -f /etc/network_turbo ]]; then
    source /etc/network_turbo >/dev/null 2>&1
fi
export PYTHONPATH="$BASE/source/src"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export TORCH_HOME="$BASE/cache/torch" HF_HOME="$BASE/cache/huggingface"
export WANDB_MODE=disabled HF_HUB_DISABLE_TELEMETRY=1
PYTHON="$BASE/venv/bin/python"
SCRIPT="$BASE/source/scripts/engineering/train_visual_completion.py"
PHASE=waiting_for_data
status() {
    printf '{"phase":"%s","exit_code":%s,"pid":%s}\n' "$PHASE" "$1" "$$" > "$BASE/pipeline-status.tmp"
    mv "$BASE/pipeline-status.tmp" "$BASE/pipeline-status.json"
}
trap 'status "$?"' EXIT
status 0
for ((attempt=0; attempt<240; attempt++)); do
    [[ -f "$BASE/data/tomato-xet.hdf5" ]] && break
    sleep 5
done
[[ -f "$BASE/data/tomato-xet.hdf5" ]] || { echo 'Dataset did not finish downloading'; exit 1; }
PHASE=preparing
status 0
"$PYTHON" "$SCRIPT" prepare --hdf5 "$BASE/data/tomato-xet.hdf5" --output "$BASE/data/rendered-v1"
PHASE=smoke
status 0
"$PYTHON" "$SCRIPT" train --data "$BASE/data/rendered-v1" --output "$BASE/runs/smoke-v1" --smoke --epochs 1
[[ -f "$BASE/runs/smoke-v1/smoke_pass.json" ]]
PHASE=training
status 0
"$PYTHON" "$SCRIPT" train --data "$BASE/data/rendered-v1" --output "$BASE/runs/completion-v1" --epochs 30 --batch-size 64 --lr 0.0001
PHASE=completed
status 0
