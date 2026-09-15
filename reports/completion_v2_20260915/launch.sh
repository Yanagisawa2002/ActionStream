#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/actionstream-agent-20260915/source
export PYTHONPATH=src
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/root/autodl-tmp/actionstream-agent-20260915/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COMMON=(--base /root/autodl-tmp/actionstream-agent-20260915 --protocol configs/completion_v2.json)
/root/autodl-tmp/actionstream-agent-20260915/venv/bin/python -u scripts/engineering/completion_v2.py prepare "${COMMON[@]}"
/root/autodl-tmp/actionstream-agent-20260915/venv/bin/python -u scripts/engineering/completion_v2.py train --smoke "${COMMON[@]}"
/root/autodl-tmp/actionstream-agent-20260915/venv/bin/python -u scripts/engineering/completion_v2.py train "${COMMON[@]}"
/root/autodl-tmp/actionstream-agent-20260915/runtime-venv/bin/python -u scripts/engineering/completion_v2.py fresh "${COMMON[@]}"
/root/autodl-tmp/actionstream-agent-20260915/venv/bin/python -u scripts/engineering/completion_v2.py evaluate "${COMMON[@]}"
printf 'COMPLETION_V2_PIPELINE_FINISHED\n'
