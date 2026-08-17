#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ACTIONSTREAM_VENV:-$HOME/.venvs/actionstream}"
OUTPUT_ROOT="${ACTIONSTREAM_OUTPUT_ROOT:-$ROOT/outputs}"
M5_ROOT="$OUTPUT_ROOT/m5_g0"
PYTHON="$VENV/bin/python"
CONFIG="$ROOT/configs/m5_g0.json"
TASK_AUDIT="$ROOT/outputs/m5_g0/audit/task_entity_audit.json"
CALIBRATION_MANIFEST="$ROOT/outputs/m5_g0/protocol/calibration_seed_manifest.json"
NO_SHIFT_MANIFEST="$ROOT/outputs/m5_g0/protocol/no_shift_seed_manifest.json"
SEALED_MANIFEST="$ROOT/outputs/m5_g0/protocol/sealed_seed_manifest.json"
PROTOCOL_DECISION="$M5_ROOT/protocol/frozen_experiment.json"
NO_SHIFT_DECISION="$M5_ROOT/no_shift/decision.json"
VALIDATION="$M5_ROOT/report/formal_validation.json"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.cache/actionstream/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

"$PYTHON" -m actionstream.libero_config --config-dir "$LIBERO_CONFIG_PATH"
"$PYTHON" -m actionstream.m5_protocol "$ROOT/outputs/m5_g0/protocol"

json_field() {
  "$PYTHON" -c \
    'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); print(value[sys.argv[2]])' \
    "$1" "$2"
}

run_benchmark() {
  local output_dir="$1"
  shift
  local resume_args=()
  if [[ -f "$output_dir/run_manifest.json" ]]; then
    resume_args=(--resume-after-crash)
  elif [[ -e "$output_dir" ]]; then
    printf 'Refusing untracked nonempty/partial output: %s\n' "$output_dir" >&2
    exit 1
  fi
  "$PYTHON" -m actionstream.m5_benchmark \
    "$@" \
    --config="$CONFIG" \
    --task-audit="$TASK_AUDIT" \
    --output-dir="$output_dir" \
    "${resume_args[@]}"
}

make_calibration_decision() {
  local task_id="$1"
  local output="$2"
  shift 2
  "$PYTHON" -m actionstream.m5_calibration \
    "$@" \
    --task-id="$task_id" \
    --output="$output"
}

CANDIDATE_FINAL=""
CANDIDATE_STATUS=""
run_candidate() {
  local task_id="$1"
  local previous_task_decision="${2:-}"
  local task_root="$M5_ROOT/calibration/task${task_id}"
  local previous_args=()
  if [[ -n "$previous_task_decision" ]]; then
    previous_args=(--previous-task-decision="$previous_task_decision")
  fi

  local run50="$task_root/magnitude50"
  run_benchmark "$run50" \
    --phase=calibration \
    --task-id="$task_id" \
    --displacement-mm=50 \
    --seed-manifest="$CALIBRATION_MANIFEST" \
    "${previous_args[@]}"
  local decision="$task_root/decision_after_50.json"
  make_calibration_decision "$task_id" "$decision" "$run50/episodes.jsonl"
  local status
  status="$(json_field "$decision" status)"

  if [[ "$status" == "needs_70_mm" || "$status" == "needs_30_mm" ]]; then
    local magnitude
    magnitude="$(json_field "$decision" next_magnitude_mm)"
    local next_run="$task_root/magnitude${magnitude}"
    run_benchmark "$next_run" \
      --phase=calibration \
      --task-id="$task_id" \
      --displacement-mm="$magnitude" \
      --seed-manifest="$CALIBRATION_MANIFEST" \
      --calibration-directive="$decision" \
      "${previous_args[@]}"
    local next_decision="$task_root/decision_after_50_${magnitude}.json"
    make_calibration_decision \
      "$task_id" \
      "$next_decision" \
      "$run50/episodes.jsonl" \
      "$next_run/episodes.jsonl"
    decision="$next_decision"
    status="$(json_field "$decision" status)"
  fi

  if [[ "$status" == "needs_30_mm" ]]; then
    local run30="$task_root/magnitude30"
    run_benchmark "$run30" \
      --phase=calibration \
      --task-id="$task_id" \
      --displacement-mm=30 \
      --seed-manifest="$CALIBRATION_MANIFEST" \
      --calibration-directive="$decision" \
      "${previous_args[@]}"
    local decision30="$task_root/decision_after_50_70_30.json"
    make_calibration_decision \
      "$task_id" \
      "$decision30" \
      "$run50/episodes.jsonl" \
      "$task_root/magnitude70/episodes.jsonl" \
      "$run30/episodes.jsonl"
    decision="$decision30"
    status="$(json_field "$decision" status)"
  fi

  if [[ "$status" != "selected" && "$status" != "candidate_no_go" ]]; then
    printf 'Calibration did not reach a final state: %s (%s)\n' "$status" "$decision" >&2
    exit 1
  fi
  CANDIDATE_FINAL="$decision"
  CANDIDATE_STATUS="$status"
}

run_candidate 0
TASK0_DECISION="$CANDIDATE_FINAL"
TASK0_STATUS="$CANDIDATE_STATUS"
FINAL_DECISIONS=("$TASK0_DECISION")
if [[ "$TASK0_STATUS" == "candidate_no_go" ]]; then
  run_candidate 2 "$TASK0_DECISION"
  TASK2_DECISION="$CANDIDATE_FINAL"
  TASK2_STATUS="$CANDIDATE_STATUS"
  FINAL_DECISIONS+=("$TASK2_DECISION")
fi

"$PYTHON" -m actionstream.m5_freeze \
  --config="$CONFIG" \
  --task-audit="$TASK_AUDIT" \
  --calibration-decision "${FINAL_DECISIONS[@]}" \
  --seed-manifest \
    "$CALIBRATION_MANIFEST" \
    "$NO_SHIFT_MANIFEST" \
    "$SEALED_MANIFEST" \
  --output="$PROTOCOL_DECISION"

PROTOCOL_STATUS="$(json_field "$PROTOCOL_DECISION" status)"
NO_SHIFT_STATUS="not_run_calibration_hard_stop"
NO_SHIFT_RUN=""
SEALED_RUN=""
EPISODE_EVIDENCE=()
ACTION_EVIDENCE=()
REQUEST_EVIDENCE=()
EVENT_EVIDENCE=()

if [[ "$PROTOCOL_STATUS" == "selected" ]]; then
  NO_SHIFT_TASK="$(json_field "$PROTOCOL_DECISION" no_shift_task_id)"
  NO_SHIFT_RUN="$M5_ROOT/no_shift/task${NO_SHIFT_TASK}"
  run_benchmark "$NO_SHIFT_RUN" \
    --phase=no_shift \
    --task-id="$NO_SHIFT_TASK" \
    --displacement-mm=0 \
    --seed-manifest="$NO_SHIFT_MANIFEST" \
    --protocol-decision="$PROTOCOL_DECISION"

  "$PYTHON" -m actionstream.m5_no_shift \
    --protocol-decision="$PROTOCOL_DECISION" \
    --episodes="$NO_SHIFT_RUN/episodes.jsonl" \
    --actions="$NO_SHIFT_RUN/actions.jsonl" \
    --output="$NO_SHIFT_DECISION"
  NO_SHIFT_STATUS="$(json_field "$NO_SHIFT_DECISION" status)"
  EPISODE_EVIDENCE+=("$NO_SHIFT_RUN/episodes.jsonl")
  ACTION_EVIDENCE+=("$NO_SHIFT_RUN/actions.jsonl")
  REQUEST_EVIDENCE+=("$NO_SHIFT_RUN/requests.jsonl")
  EVENT_EVIDENCE+=("$NO_SHIFT_RUN/events.jsonl")

  if [[ "$NO_SHIFT_STATUS" == "pass" ]]; then
    SEALED_TASK="$(json_field "$PROTOCOL_DECISION" selected_task_id)"
    SEALED_MAGNITUDE="$(json_field "$PROTOCOL_DECISION" selected_displacement_magnitude_mm)"
    SEALED_RUN="$M5_ROOT/sealed/task${SEALED_TASK}_magnitude${SEALED_MAGNITUDE}"
    run_benchmark "$SEALED_RUN" \
      --phase=sealed \
      --task-id="$SEALED_TASK" \
      --displacement-mm="$SEALED_MAGNITUDE" \
      --seed-manifest="$SEALED_MANIFEST" \
      --protocol-decision="$PROTOCOL_DECISION" \
      --no-shift-decision="$NO_SHIFT_DECISION"
    EPISODE_EVIDENCE+=("$SEALED_RUN/episodes.jsonl")
    ACTION_EVIDENCE+=("$SEALED_RUN/actions.jsonl")
    REQUEST_EVIDENCE+=("$SEALED_RUN/requests.jsonl")
    EVENT_EVIDENCE+=("$SEALED_RUN/events.jsonl")
  fi
elif [[ "$PROTOCOL_STATUS" != "calibration_no_go" ]]; then
  printf 'Unexpected frozen protocol status: %s\n' "$PROTOCOL_STATUS" >&2
  exit 1
fi

STATISTICS="$M5_ROOT/report/statistical_tests_input.json"
validation_args=(
  --config="$CONFIG"
  --task-audit="$TASK_AUDIT"
  --seed-manifest
    "$CALIBRATION_MANIFEST"
    "$NO_SHIFT_MANIFEST"
    "$SEALED_MANIFEST"
  --protocol-decision="$PROTOCOL_DECISION"
  --run-tests
  --output="$VALIDATION"
)
if [[ -f "$NO_SHIFT_DECISION" && "$PROTOCOL_STATUS" == "selected" ]]; then
  validation_args+=(--no-shift-decision="$NO_SHIFT_DECISION")
fi
if (( ${#EPISODE_EVIDENCE[@]} )); then
  validation_args+=(--episodes "${EPISODE_EVIDENCE[@]}")
  validation_args+=(--actions "${ACTION_EVIDENCE[@]}")
  validation_args+=(--requests "${REQUEST_EVIDENCE[@]}")
  validation_args+=(--events "${EVENT_EVIDENCE[@]}")
fi
"$PYTHON" -m actionstream.m5_validation "${validation_args[@]}"

analysis_args=(
  "${EPISODE_EVIDENCE[@]}"
  --output="$STATISTICS"
  --validation="$VALIDATION"
)
if [[ "$PROTOCOL_STATUS" == "calibration_no_go" ]]; then
  HARD_STOP_REASON="$(json_field "$CANDIDATE_FINAL" reason)"
  analysis_args+=(--calibration-no-go-reason="$HARD_STOP_REASON")
fi
"$PYTHON" -m actionstream.m5_analysis "${analysis_args[@]}"

REPORT_ROOT="$M5_ROOT/report"
report_args=(
  --config="$CONFIG"
  --task-audit="$TASK_AUDIT"
  --calibration-decision "${FINAL_DECISIONS[@]}"
  --seed-manifest
    "$CALIBRATION_MANIFEST"
    "$NO_SHIFT_MANIFEST"
    "$SEALED_MANIFEST"
  --statistics="$STATISTICS"
  --protocol-decision="$PROTOCOL_DECISION"
  --validation="$VALIDATION"
  --output-root="$REPORT_ROOT"
  --m4-baseline-intact
)
if [[ -f "$NO_SHIFT_DECISION" && "$PROTOCOL_STATUS" == "selected" ]]; then
  report_args+=(--no-shift-decision="$NO_SHIFT_DECISION")
fi
if (( ${#EPISODE_EVIDENCE[@]} )); then
  report_args+=(--episodes "${EPISODE_EVIDENCE[@]}")
  report_args+=(--actions "${ACTION_EVIDENCE[@]}")
  report_args+=(--events "${EVENT_EVIDENCE[@]}")
fi
"$PYTHON" -m actionstream.m5_report "${report_args[@]}"

printf 'M5-G0 protocol: %s\n' "$PROTOCOL_STATUS"
printf 'M5-G0 no-shift: %s\n' "$NO_SHIFT_STATUS"
printf 'M5-G0 acceptance: %s\n' "$REPORT_ROOT/acceptance.json"
