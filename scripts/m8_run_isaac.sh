#!/usr/bin/env bash
# Build the current checkout and run M8 persistent native-Isaac batches on Linux.
#
# This runner is the Linux/cloud counterpart of m8_run_isaac.ps1.  It is
# deliberately fail-closed: an explicit authorization flag is required, every
# batch is validated from the live checkout before GPU use, the selected GPU is
# sampled before and after the build, canonical outputs may not already exist,
# and cleanup targets only process groups created by this invocation.

set -Eeuo pipefail
IFS=$'\n\t'

readonly GPU_MEMORY_THRESHOLD_MIB=4096
readonly DEFAULT_BATCH_TIMEOUT_SECONDS=14400
readonly MIN_BATCH_TIMEOUT_SECONDS=60
readonly MAX_BATCH_TIMEOUT_SECONDS=86400

SCRIPT_PATH="$(realpath -e -- "${BASH_SOURCE[0]}")"
SCRIPT_DIRECTORY="$(dirname -- "$SCRIPT_PATH")"
REPOSITORY_ROOT="$(realpath -e -- "$SCRIPT_DIRECTORY/..")"
SOURCE_ROOT="$REPOSITORY_ROOT/ros2_ws/src"
SUPPORT_SCRIPT="$SCRIPT_DIRECTORY/m8_linux_runner_support.py"

MATRIX_SUITE_MANIFEST=""
ISAAC_WORKSPACE=""
PIXI_EXE=""
HOST_PYTHON=""
HEADLESS="true"
BATCH_TIMEOUT_SECONDS="$DEFAULT_BATCH_TIMEOUT_SECONDS"
GPU_INDEX=0
SELECTED_GPU_UUID=""
AUTHORIZE_NATIVE_GPU_RUN=0
CAPTURE_SINGLE_EPISODE_VIDEOS=0

SUITE_PATH=""
SUITE_DIRECTORY=""
SUITE_SPLIT=""
FREEZE_PATH=""
LOG_DIRECTORY=""
PREFLIGHT_RECEIPT_PATH=""
FAILURE_RECEIPT_PATH=""
TIMEOUT_RECEIPT_PATH=""
GPU_REFUSAL_RECEIPT_PATH=""
INITIAL_GPU_SNAPSHOT_PATH=""
POST_BUILD_GPU_SNAPSHOT_PATH=""
EXTERNAL_ENVIRONMENT_EVIDENCE_PATH=""
VIDEO_DIRECTORY=""
SOURCE_MANIFEST_SHA256=""
CURRENT_STAGE="argument_parse"
CURRENT_BATCH=""
TIMED_OUT=0
ERROR_HANDLING=0

declare -a BATCH_PATHS=()
declare -a BATCH_STRATEGIES=()
declare -a BATCH_STEMS=()
declare -a BATCH_PROFILES=()
declare -a BATCH_EPISODE_COUNTS=()
declare -a BATCH_VALIDATION_LOGS=()
declare -a PLANNED_PROCESS_LOGS=()
declare -a OWNED_PIDS=()
declare -a OWNED_PGIDS=()
declare -a OWNED_ACTIVE=()
STARTED_PID=""
STARTED_PGID=""


usage() {
    cat <<'EOF'
Usage:
  bash scripts/m8_run_isaac.sh \
    --matrix-suite-manifest PATH \
    --isaac-workspace PATH \
    [--pixi-exe PATH] \
    [--gpu-index INDEX] \
    [--headless true|false] \
    [--batch-timeout-seconds 60..86400] \
    [--capture-single-episode-videos] \
    --authorize-native-gpu-run

The Isaac workspace must contain the official Linux pixi.toml.  The runner
builds this checkout into that workspace and refuses to overwrite M8 replay,
analysis, figures, or any timestamped native-run receipt directory.
EOF
}


die() {
    printf 'm8_run_isaac.sh: %s\n' "$*" >&2
    return 1
}


require_file() {
    local path=$1
    local description=$2
    [[ -f "$path" ]] || die "$description was not found: $path"
}


require_regular_nonempty_file() {
    local path=$1
    local description=$2
    [[ -f "$path" && ! -L "$path" && -s "$path" ]] || \
        die "$description must be a nonempty regular file, not a symlink: $path"
}


require_executable() {
    local path=$1
    local description=$2
    [[ -f "$path" && -x "$path" ]] || die "$description is not executable: $path"
}


require_absent() {
    local path=$1
    local description=$2
    [[ ! -e "$path" ]] || die "refusing to overwrite existing $description: $path"
}


owned_index() {
    local pid=$1
    local pgid=$2
    local index
    for index in "${!OWNED_PIDS[@]}"; do
        if [[ "${OWNED_PIDS[$index]}" == "$pid" && "${OWNED_PGIDS[$index]}" == "$pgid" ]]; then
            printf '%s\n' "$index"
            return 0
        fi
    done
    return 1
}


process_is_running() {
    local pid=$1
    local state
    state="$(ps -o stat= -p "$pid" 2>/dev/null | awk '{print $1}')" || return 1
    [[ -n "$state" && "$state" != Z* ]]
}


process_group_has_live_members() {
    local pgid=$1
    ps -eo pgid=,stat= | awk -v expected="$pgid" '
        $1 == expected && $2 !~ /^Z/ { found = 1 }
        END { exit(found ? 0 : 1) }
    '
}


stop_owned_process_group() {
    local pid=$1
    local pgid=$2
    local index
    index="$(owned_index "$pid" "$pgid")" || return 0
    [[ "${OWNED_ACTIVE[$index]}" == "1" ]] || return 0

    # The recorded PID/PGID pair comes only from a successful setsid launch.
    # Never derive a cleanup target by name and never touch pre-existing jobs.
    if process_group_has_live_members "$pgid"; then
        kill -TERM -- "-$pgid" 2>/dev/null || true
        local attempt
        for ((attempt = 0; attempt < 100; attempt += 1)); do
            process_group_has_live_members "$pgid" || break
            sleep 0.1
        done
        if process_group_has_live_members "$pgid"; then
            kill -KILL -- "-$pgid" 2>/dev/null || true
            for ((attempt = 0; attempt < 20; attempt += 1)); do
                process_group_has_live_members "$pgid" || break
                sleep 0.1
            done
        fi
        if process_group_has_live_members "$pgid"; then
            printf 'Owned process group %s did not terminate after SIGKILL.\n' "$pgid" >&2
            return 1
        fi
    fi
    wait "$pid" 2>/dev/null || true
    OWNED_ACTIVE[$index]=0
}


cleanup_owned_process_groups() {
    local index
    for ((index = ${#OWNED_PIDS[@]} - 1; index >= 0; index -= 1)); do
        stop_owned_process_group "${OWNED_PIDS[$index]}" "${OWNED_PGIDS[$index]}" || true
    done
}


write_failure_receipt() {
    local exit_code=$1
    local line=$2
    local command=$3
    [[ -n "$LOG_DIRECTORY" && -d "$LOG_DIRECTORY" ]] || return 0
    [[ -n "$HOST_PYTHON" && -f "$SUPPORT_SCRIPT" ]] || return 0

    local output=$FAILURE_RECEIPT_PATH
    declare -a arguments=(
        "$SUPPORT_SCRIPT" write-failure
        --repository-root "$REPOSITORY_ROOT"
        --output "$output"
        --log-directory "$LOG_DIRECTORY"
        --stage "$CURRENT_STAGE"
        --suite "$SUITE_PATH"
        --batch-timeout-seconds "$BATCH_TIMEOUT_SECONDS"
        --message "native runner command failed"
        --command "$command"
        --line "$line"
        --exit-code "$exit_code"
    )
    if ((TIMED_OUT)); then
        output=$TIMEOUT_RECEIPT_PATH
        arguments[5]="$output"
        arguments+=(--timed-out)
    fi
    [[ -n "$CURRENT_BATCH" ]] && arguments+=(--current-batch "$CURRENT_BATCH")
    [[ -f "$PREFLIGHT_RECEIPT_PATH" ]] && arguments+=(--preflight-receipt "$PREFLIGHT_RECEIPT_PATH")
    [[ -n "$SOURCE_MANIFEST_SHA256" ]] && arguments+=(--source-manifest-sha256 "$SOURCE_MANIFEST_SHA256")
    [[ -f "$INITIAL_GPU_SNAPSHOT_PATH" ]] && arguments+=(--initial-snapshot "$INITIAL_GPU_SNAPSHOT_PATH")
    [[ -f "$POST_BUILD_GPU_SNAPSHOT_PATH" ]] && arguments+=(--post-snapshot "$POST_BUILD_GPU_SNAPSHOT_PATH")
    require_absent "$output" "native failure/timeout receipt" || return 0
    "$HOST_PYTHON" "${arguments[@]}" || return 0
    printf 'Native M8 attempt failed at stage %s; receipt: %s\n' "$CURRENT_STAGE" "$output" >&2
}


on_error() {
    local exit_code=$1
    local line=$2
    local command=$3
    if ((ERROR_HANDLING)); then
        exit "$exit_code"
    fi
    ERROR_HANDLING=1
    trap - ERR EXIT INT TERM HUP
    set +e
    cleanup_owned_process_groups
    write_failure_receipt "$exit_code" "$line" "$command"
    exit "$exit_code"
}


on_signal() {
    local signal_name=$1
    printf 'Received %s during stage %s.\n' "$signal_name" "$CURRENT_STAGE" >&2
    on_error 130 "$LINENO" "received signal $signal_name"
}


trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP
trap 'cleanup_owned_process_groups' EXIT


parse_arguments() {
    while (($#)); do
        case "$1" in
            --matrix-suite-manifest)
                (($# >= 2)) || die "--matrix-suite-manifest requires a value"
                MATRIX_SUITE_MANIFEST=$2
                shift 2
                ;;
            --isaac-workspace)
                (($# >= 2)) || die "--isaac-workspace requires a value"
                ISAAC_WORKSPACE=$2
                shift 2
                ;;
            --pixi-exe)
                (($# >= 2)) || die "--pixi-exe requires a value"
                PIXI_EXE=$2
                shift 2
                ;;
            --gpu-index)
                (($# >= 2)) || die "--gpu-index requires a value"
                [[ "$2" =~ ^[0-9]+$ ]] || die "--gpu-index must be a non-negative integer"
                GPU_INDEX=$2
                shift 2
                ;;
            --headless)
                (($# >= 2)) || die "--headless requires true or false"
                [[ "$2" == "true" || "$2" == "false" ]] || die "--headless requires true or false"
                HEADLESS=$2
                shift 2
                ;;
            --batch-timeout-seconds)
                (($# >= 2)) || die "--batch-timeout-seconds requires a value"
                [[ "$2" =~ ^[0-9]+$ ]] || die "--batch-timeout-seconds must be an integer"
                BATCH_TIMEOUT_SECONDS=$2
                shift 2
                ;;
            --authorize-native-gpu-run)
                AUTHORIZE_NATIVE_GPU_RUN=1
                shift
                ;;
            --capture-single-episode-videos)
                CAPTURE_SINGLE_EPISODE_VIDEOS=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "unknown argument: $1"
                ;;
        esac
    done
}


resolve_inputs() {
    [[ -n "$MATRIX_SUITE_MANIFEST" ]] || die "--matrix-suite-manifest is required"
    [[ -n "$ISAAC_WORKSPACE" ]] || die "--isaac-workspace is required"
    ((AUTHORIZE_NATIVE_GPU_RUN)) || die "native Isaac execution is fail-closed; pass --authorize-native-gpu-run after reviewing the GPU preflight contract"
    ((BATCH_TIMEOUT_SECONDS >= MIN_BATCH_TIMEOUT_SECONDS && BATCH_TIMEOUT_SECONDS <= MAX_BATCH_TIMEOUT_SECONDS)) || die "batch timeout must be between $MIN_BATCH_TIMEOUT_SECONDS and $MAX_BATCH_TIMEOUT_SECONDS seconds"

    require_file "$SUPPORT_SCRIPT" "Linux runner support module"
    require_file "$SOURCE_ROOT/action_stream_isaac/action_stream_isaac/dynamic_isaac_adapter.py" "current-source dynamic Isaac adapter"
    SUITE_PATH="$(realpath -e -- "$MATRIX_SUITE_MANIFEST")"
    ISAAC_WORKSPACE="$(realpath -e -- "$ISAAC_WORKSPACE")"
    require_file "$ISAAC_WORKSPACE/pixi.toml" "official Isaac Pixi manifest"
    require_file "$ISAAC_WORKSPACE/pixi.lock" "official Isaac Pixi lock"

    if [[ -z "$PIXI_EXE" ]]; then
        PIXI_EXE="$(command -v pixi || true)"
    else
        PIXI_EXE="$(realpath -e -- "$PIXI_EXE")"
    fi
    [[ -n "$PIXI_EXE" ]] || die "pixi was not found on PATH; pass --pixi-exe"
    require_executable "$PIXI_EXE" "Pixi executable"

    HOST_PYTHON="$(command -v python3 || command -v python || true)"
    if [[ -z "$HOST_PYTHON" ]]; then
        local environment_python
        for environment_python in \
            "$ISAAC_WORKSPACE/.pixi/envs/default/bin/python3" \
            "$ISAAC_WORKSPACE/.pixi/envs/default/bin/python"; do
            if [[ -x "$environment_python" ]]; then
                HOST_PYTHON="$(realpath -e -- "$environment_python")"
                break
            fi
        done
    fi
    [[ -n "$HOST_PYTHON" ]] || die "python3/python was not found on the host or in the pinned Pixi default environment"
    "$HOST_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' || die "the Linux runner requires Python 3.9 or newer"
    command -v setsid >/dev/null || die "setsid is required for owned process-group isolation"
    command -v ps >/dev/null || die "ps is required for owned process inspection"
    command -v nvidia-smi >/dev/null || die "nvidia-smi was not found; refusing native GPU execution"
}


load_suite_records() {
    local -a fields=()
    mapfile -d '' -t fields < <(
        "$HOST_PYTHON" "$SUPPORT_SCRIPT" suite-records \
            --repository-root "$REPOSITORY_ROOT" \
            --suite "$SUITE_PATH"
    )
    ((${#fields[@]} >= 7)) || die "matrix suite inspection returned no valid batch inventory"
    SUITE_SPLIT=${fields[0]}
    FREEZE_PATH=${fields[1]}
    local payload_count=$((${#fields[@]} - 2))
    ((payload_count % 5 == 0)) || die "matrix suite inspection returned a malformed batch inventory"
    local batch_count=$((payload_count / 5))
    ((batch_count >= 1)) || die "matrix suite has no persistent batches"
    local index offset
    for ((index = 0; index < batch_count; index += 1)); do
        offset=$((2 + index * 5))
        BATCH_PATHS+=("${fields[$offset]}")
        BATCH_STRATEGIES+=("${fields[$((offset + 1))]}")
        BATCH_STEMS+=("${fields[$((offset + 2))]}")
        BATCH_PROFILES+=("${fields[$((offset + 3))]}")
        BATCH_EPISODE_COUNTS+=("${fields[$((offset + 4))]}")
        BATCH_VALIDATION_LOGS+=("${fields[$((offset + 2))]}.validate_only.log")
    done
}


run_pixi_bash_logged() {
    local log_path=$1
    local script=$2
    shift 2
    require_absent "$log_path" "Pixi command log"
    set +e
    "$PIXI_EXE" run --frozen --manifest-path "$ISAAC_WORKSPACE/pixi.toml" -- \
        bash -c "$script" m8-pixi "$@" 2>&1 | tee -- "$log_path"
    local -a statuses=("${PIPESTATUS[@]}")
    set -e
    ((${statuses[0]} == 0)) || die "Pixi command failed with exit code ${statuses[0]}; see $log_path"
    ((${statuses[1]} == 0)) || die "tee failed with exit code ${statuses[1]}; see $log_path"
}


run_pixi_bash() {
    local script=$1
    shift
    "$PIXI_EXE" run --frozen --manifest-path "$ISAAC_WORKSPACE/pixi.toml" -- \
        bash -c "$script" m8-pixi "$@"
}


start_owned_pixi_bash() {
    local stdout_path=$1
    local stderr_path=$2
    local script=$3
    shift 3
    require_absent "$stdout_path" "owned process stdout log"
    require_absent "$stderr_path" "owned process stderr log"
    setsid "$PIXI_EXE" run --frozen --manifest-path "$ISAAC_WORKSPACE/pixi.toml" -- \
        bash -c "$script" m8-pixi "$@" >"$stdout_path" 2>"$stderr_path" &
    local pid=$!
    sleep 0.1
    local pgid
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    if [[ -z "$pgid" || "$pgid" != "$pid" ]]; then
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        die "owned process did not establish an isolated process group; see $stderr_path"
    fi
    OWNED_PIDS+=("$pid")
    OWNED_PGIDS+=("$pgid")
    OWNED_ACTIVE+=(1)
    STARTED_PID=$pid
    STARTED_PGID=$pgid
}


wait_owned_with_timeout() {
    local pid=$1
    local pgid=$2
    local timeout_seconds=$3
    local description=$4
    local started=$SECONDS
    while process_is_running "$pid"; do
        if ((SECONDS - started >= timeout_seconds)); then
            TIMED_OUT=1
            die "$description exceeded the wall timeout of $timeout_seconds seconds: $CURRENT_BATCH"
        fi
        sleep 1
    done
    local exit_code=0
    if wait "$pid"; then
        exit_code=0
    else
        exit_code=$?
    fi
    # A Pixi/adapter leader can exit while descendants remain in its session.
    # Always drain the exact recorded PGID before marking this job inactive.
    owned_index "$pid" "$pgid" >/dev/null || die "completed process was not in the owned inventory"
    stop_owned_process_group "$pid" "$pgid"
    ((exit_code == 0)) || die "$description exited with code $exit_code"
}


capture_gpu_snapshot() {
    local phase=$1
    local output=$2
    local snapshot_status=0
    local selected_uuid=""
    if selected_uuid="$("$HOST_PYTHON" "$SUPPORT_SCRIPT" gpu-snapshot \
        --nvidia-smi "$(command -v nvidia-smi)" \
        --gpu-index "$GPU_INDEX" \
        --memory-threshold-mib "$GPU_MEMORY_THRESHOLD_MIB" \
        --phase "$phase" \
        --output "$output")"; then
        [[ "$selected_uuid" == GPU-* ]] || die "nvidia-smi returned an invalid selected GPU UUID: $selected_uuid"
        if [[ -z "$SELECTED_GPU_UUID" ]]; then
            SELECTED_GPU_UUID=$selected_uuid
        elif [[ "$SELECTED_GPU_UUID" != "$selected_uuid" ]]; then
            die "selected GPU identity changed between preflights: $SELECTED_GPU_UUID -> $selected_uuid"
        fi
        return 0
    else
        snapshot_status=$?
    fi
    if ((snapshot_status == 3)); then
        "$HOST_PYTHON" "$SUPPORT_SCRIPT" write-gpu-refusal \
            --snapshot "$output" \
            --memory-threshold-mib "$GPU_MEMORY_THRESHOLD_MIB" \
            --output "$GPU_REFUSAL_RECEIPT_PATH"
        die "GPU $GPU_INDEX is occupied; no process was killed and Isaac was not started (receipt: $GPU_REFUSAL_RECEIPT_PATH)"
    fi
    return "$snapshot_status"
}


validate_current_source_inputs() {
    local benchmark_source="$SOURCE_ROOT/action_stream_benchmark"
    local isaac_source="$SOURCE_ROOT/action_stream_isaac"
    local policy_source="$SOURCE_ROOT/action_stream_policy"
    if [[ "$SUITE_SPLIT" == "frozen_holdout" ]]; then
        CURRENT_STAGE="freeze_validation"
        run_pixi_bash_logged "$LOG_DIRECTORY/freeze_validate.log" '
set -Eeuo pipefail
repository_root=$1
freeze_manifest=$2
benchmark_source=$3
export PYTHONPATH="$benchmark_source${PYTHONPATH:+:$PYTHONPATH}"
cd "$repository_root"
python -m action_stream_benchmark.m8_cli freeze-validate \
    --repository-root "$repository_root" \
    --manifest "$freeze_manifest"
' "$REPOSITORY_ROOT" "$FREEZE_PATH" "$benchmark_source"
    fi

    CURRENT_STAGE="batch_source_validate_only"
    local index log_path
    for index in "${!BATCH_PATHS[@]}"; do
        CURRENT_BATCH=${BATCH_PATHS[$index]}
        log_path="$LOG_DIRECTORY/${BATCH_VALIDATION_LOGS[$index]}"
        printf 'Validating current-source M8 batch before GPU/build: %s\n' "$CURRENT_BATCH"
        run_pixi_bash_logged "$log_path" '
set -Eeuo pipefail
repository_root=$1
batch_manifest=$2
strategy=$3
benchmark_source=$4
isaac_source=$5
policy_source=$6
export PYTHONPATH="$benchmark_source:$isaac_source:$policy_source${PYTHONPATH:+:$PYTHONPATH}"
cd "$repository_root"
python -m action_stream_isaac.dynamic_isaac_adapter \
    --matrix-manifest "$batch_manifest" \
    --expected-strategy "$strategy" \
    --validate-only
' "$REPOSITORY_ROOT" "${BATCH_PATHS[$index]}" "${BATCH_STRATEGIES[$index]}" \
            "$benchmark_source" "$isaac_source" "$policy_source"
    done
    CURRENT_BATCH=""
}


validate_external_environment_inputs() {
    CURRENT_STAGE="external_environment_validation"
    "$HOST_PYTHON" "$SUPPORT_SCRIPT" validate-external-environment \
        --isaac-workspace "$ISAAC_WORKSPACE" \
        --pixi-exe "$PIXI_EXE"
}


write_external_environment_evidence() {
    CURRENT_STAGE="external_environment_receipt"
    "$HOST_PYTHON" "$SUPPORT_SCRIPT" write-external-environment \
        --isaac-workspace "$ISAAC_WORKSPACE" \
        --pixi-exe "$PIXI_EXE" \
        --output "$EXTERNAL_ENVIRONMENT_EVIDENCE_PATH"
}


build_current_source() {
    CURRENT_STAGE="source_fingerprint"
    local -a source_fields=()
    mapfile -d '' -t source_fields < <(
        "$HOST_PYTHON" "$SUPPORT_SCRIPT" source-manifest \
            --repository-root "$REPOSITORY_ROOT" \
            --source-root "$SOURCE_ROOT" \
            --output "$LOG_DIRECTORY/source_manifest.txt"
    )
    ((${#source_fields[@]} == 3)) || die "source fingerprint helper returned malformed output"
    local source_file_count=${source_fields[0]}
    SOURCE_MANIFEST_SHA256=${source_fields[1]}
    SOURCE_ROOT_RELATIVE=${source_fields[2]}
    printf 'Current ActionStream ROS source manifest SHA256: %s\n' "$SOURCE_MANIFEST_SHA256"

    CURRENT_STAGE="colcon_build"
    printf 'Building the current repository ROS sources into the native Isaac workspace...\n'
    run_pixi_bash_logged "$LOG_DIRECTORY/colcon_build.log" '
set -Eeuo pipefail
workspace=$1
source_root=$2
cd "$workspace"
colcon build \
    --base-paths "$source_root" \
    --packages-select \
        action_stream_msgs \
        action_stream_executor \
        action_stream_policy \
        action_stream_benchmark \
        action_stream_isaac \
    --merge-install \
    --cmake-args \
        -DBUILD_TESTING=ON \
        -DPython_FIND_VIRTUALENV=ONLY \
        -DPython3_FIND_VIRTUALENV=ONLY
' "$ISAAC_WORKSPACE" "$SOURCE_ROOT"
    SOURCE_FILE_COUNT=$source_file_count
}


validate_post_build_activation_scripts() {
    CURRENT_STAGE="post_build_activation_validation"
    require_regular_nonempty_file \
        "$ISAAC_WORKSPACE/install/setup.bash" \
        "colcon-generated workspace setup"
    require_regular_nonempty_file \
        "$ISAAC_WORKSPACE/install/local_setup.bash" \
        "colcon-generated workspace local setup"
}


bind_installed_runtime_and_write_preflight() {
    CURRENT_STAGE="installed_runtime_binding"
    INSTALL_SETUP="$ISAAC_WORKSPACE/install/local_setup.bash"
    EXECUTOR="$ISAAC_WORKSPACE/install/lib/action_stream_executor/action_stream_executor_node"
    require_file "$INSTALL_SETUP" "fresh colcon local overlay setup"
    require_executable "$EXECUTOR" "fresh C++ ActionStream executor"

    ROUTER="$("$PIXI_EXE" run --frozen --manifest-path "$ISAAC_WORKSPACE/pixi.toml" -- \
        bash -c 'set -Eeuo pipefail; source "$1"; type -P rmw_zenohd' m8-pixi "$INSTALL_SETUP")"
    ROUTER="$(printf '%s\n' "$ROUTER" | tail -n 1)"
    [[ "$ROUTER" == /* ]] || die "rmw_zenohd did not resolve to an absolute executable path"
    require_executable "$ROUTER" "Zenoh router executable"

    local source_adapter="$SOURCE_ROOT/action_stream_isaac/action_stream_isaac/dynamic_isaac_adapter.py"
    local source_adapter_hash
    source_adapter_hash="$("$HOST_PYTHON" "$SUPPORT_SCRIPT" sha256 --path "$source_adapter")"
    local -a installed_adapters=()
    mapfile -d '' -t installed_adapters < <(
        find "$ISAAC_WORKSPACE/install" -type f -name dynamic_isaac_adapter.py -print0 | sort -z
    )
    ((${#installed_adapters[@]} >= 1)) || die "fresh install does not contain dynamic_isaac_adapter.py"
    INSTALLED_ADAPTER=""
    local candidate candidate_hash
    for candidate in "${installed_adapters[@]}"; do
        candidate_hash="$("$HOST_PYTHON" "$SUPPORT_SCRIPT" sha256 --path "$candidate")"
        if [[ "$candidate_hash" == "$source_adapter_hash" ]]; then
            INSTALLED_ADAPTER=$candidate
            break
        fi
    done
    [[ -n "$INSTALLED_ADAPTER" ]] || die "installed dynamic adapter does not match the current checkout after colcon build"

    INSTALLED_ADAPTER_EVIDENCE="$LOG_DIRECTORY/installed_dynamic_adapter.py"
    EXECUTOR_EVIDENCE="$LOG_DIRECTORY/action_stream_executor_node"
    RUNNER_EVIDENCE="$LOG_DIRECTORY/m8_run_isaac.runner.sh"
    RUNNER_SUPPORT_EVIDENCE="$LOG_DIRECTORY/m8_linux_runner_support.py"
    "$HOST_PYTHON" "$SUPPORT_SCRIPT" copy-evidence \
        --source "$INSTALLED_ADAPTER" \
        --destination "$INSTALLED_ADAPTER_EVIDENCE" >/dev/null
    "$HOST_PYTHON" "$SUPPORT_SCRIPT" copy-evidence \
        --source "$EXECUTOR" \
        --destination "$EXECUTOR_EVIDENCE" >/dev/null
    "$HOST_PYTHON" "$SUPPORT_SCRIPT" copy-evidence \
        --source "$SCRIPT_PATH" \
        --destination "$RUNNER_EVIDENCE" >/dev/null
    "$HOST_PYTHON" "$SUPPORT_SCRIPT" copy-evidence \
        --source "$SUPPORT_SCRIPT" \
        --destination "$RUNNER_SUPPORT_EVIDENCE" >/dev/null

    PLANNED_PROCESS_LOGS=(
        "colcon_build.log"
        "replay_validate.log"
        "router.stdout.log"
        "router.stderr.log"
    )
    PLANNED_PROCESS_LOGS+=("${BATCH_VALIDATION_LOGS[@]}")
    if [[ "$SUITE_SPLIT" == "frozen_holdout" ]]; then
        PLANNED_PROCESS_LOGS+=("freeze_validate.log" "analysis_figures.log")
    fi
    local stem
    for stem in "${BATCH_STEMS[@]}"; do
        PLANNED_PROCESS_LOGS+=(
            "$stem.executor.stdout.log"
            "$stem.executor.stderr.log"
            "$stem.adapter.stdout.log"
            "$stem.adapter.stderr.log"
        )
    done

    declare -a arguments=(
        "$SUPPORT_SCRIPT" write-preflight
        --repository-root "$REPOSITORY_ROOT"
        --output "$PREFLIGHT_RECEIPT_PATH"
        --suite "$SUITE_PATH"
        --source-root "$SOURCE_ROOT"
        --source-file-count "$SOURCE_FILE_COUNT"
        --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256"
        --external-environment "$EXTERNAL_ENVIRONMENT_EVIDENCE_PATH"
        --memory-threshold-mib "$GPU_MEMORY_THRESHOLD_MIB"
        --gpu-index "$GPU_INDEX"
        --initial-snapshot "$INITIAL_GPU_SNAPSHOT_PATH"
        --post-snapshot "$POST_BUILD_GPU_SNAPSHOT_PATH"
        --installed-adapter "$INSTALLED_ADAPTER"
        --installed-adapter-evidence "$INSTALLED_ADAPTER_EVIDENCE"
        --executor "$EXECUTOR"
        --executor-evidence "$EXECUTOR_EVIDENCE"
        --router "$ROUTER"
        --runner "$SCRIPT_PATH"
        --runner-evidence "$RUNNER_EVIDENCE"
        --runner-support "$SUPPORT_SCRIPT"
        --runner-support-evidence "$RUNNER_SUPPORT_EVIDENCE"
        --batch-timeout-seconds "$BATCH_TIMEOUT_SECONDS"
    )
    if [[ "$SUITE_SPLIT" == "frozen_holdout" ]]; then
        arguments+=(--frozen-live-inputs-validated)
    else
        arguments+=(--no-frozen-live-inputs-validated)
    fi
    local value
    for value in "${BATCH_VALIDATION_LOGS[@]}"; do
        arguments+=(--validation-log "$value")
    done
    for value in "${PLANNED_PROCESS_LOGS[@]}"; do
        arguments+=(--planned-log "$value")
    done
    CURRENT_STAGE="preflight_receipt"
    "$HOST_PYTHON" "${arguments[@]}"
}


start_router() {
    CURRENT_STAGE="router_start"
    start_owned_pixi_bash \
        "$LOG_DIRECTORY/router.stdout.log" \
        "$LOG_DIRECTORY/router.stderr.log" '
set -Eeuo pipefail
workspace=$1
router=$2
gpu_uuid=$3
cd "$workspace"
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
exec "$router"
' "$ISAAC_WORKSPACE" "$ROUTER" "$SELECTED_GPU_UUID"
    ROUTER_PID=$STARTED_PID
    ROUTER_PGID=$STARTED_PGID
    sleep 2
    process_is_running "$ROUTER_PID" || die "owned Zenoh router exited during startup; see $LOG_DIRECTORY/router.stderr.log"
}


run_native_batches() {
    local index stem strategy executor_pid executor_pgid adapter_pid adapter_pgid
    local video_output
    if ((CAPTURE_SINGLE_EPISODE_VIDEOS)); then
        mkdir -- "$VIDEO_DIRECTORY"
    fi
    for index in "${!BATCH_PATHS[@]}"; do
        CURRENT_STAGE="batch_prepare"
        CURRENT_BATCH=${BATCH_PATHS[$index]}
        strategy=${BATCH_STRATEGIES[$index]}
        stem=${BATCH_STEMS[$index]}
        video_output=""
        if ((CAPTURE_SINGLE_EPISODE_VIDEOS)); then
            video_output="$VIDEO_DIRECTORY/$stem.mp4"
            require_absent "$video_output" "native viewport video"
        fi
        printf 'Starting native M8 batch: %s / %s (%s episodes)\n' \
            "${BATCH_PROFILES[$index]}" "$strategy" "${BATCH_EPISODE_COUNTS[$index]}"

        CURRENT_STAGE="batch_executor_start"
        start_owned_pixi_bash \
            "$LOG_DIRECTORY/$stem.executor.stdout.log" \
            "$LOG_DIRECTORY/$stem.executor.stderr.log" '
set -Eeuo pipefail
install_setup=$1
executor=$2
strategy=$3
gpu_uuid=$4
source "$install_setup"
export OMNI_KIT_ACCEPT_EULA=YES
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
exec "$executor" --ros-args \
    -p use_sim_time:=true \
    -p strategy:="$strategy" \
    -p action_dimension:=7 \
    -p "safe_hold_command:=[0.307015,0.0,0.589907,3.141592653589793,0.0,0.0,1.0]" \
    -p sync_periodic_replan:=true \
    -p observation_topic:=/action_stream/observation \
    -p inference_request_topic:=/action_stream/inference_request \
    -p action_chunk_topic:=/action_stream/action_chunk \
    -p robot_command_topic:=/action_stream/robot_command \
    -p diagnostics_topic:=/action_stream/diagnostics \
    -p runtime_event_topic:=/action_stream/events \
    -p episode_control_topic:=/action_stream/episode_control
' "$INSTALL_SETUP" "$EXECUTOR" "$strategy" "$SELECTED_GPU_UUID"
        executor_pid=$STARTED_PID
        executor_pgid=$STARTED_PGID
        sleep 2
        process_is_running "$executor_pid" || die "C++ executor exited during startup; see $LOG_DIRECTORY/$stem.executor.stderr.log"

        CURRENT_STAGE="batch_adapter_start"
        start_owned_pixi_bash \
            "$LOG_DIRECTORY/$stem.adapter.stdout.log" \
            "$LOG_DIRECTORY/$stem.adapter.stderr.log" '
set -Eeuo pipefail
install_setup=$1
repository_root=$2
batch_manifest=$3
strategy=$4
headless=$5
gpu_uuid=$6
video_output=$7
source "$install_setup"
export OMNI_KIT_ACCEPT_EULA=YES
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
cd "$repository_root"
video_arguments=()
if [[ -n "$video_output" ]]; then
    video_arguments+=(--video-output "$video_output")
fi
exec python -m action_stream_isaac.dynamic_isaac_adapter \
    --matrix-manifest "$batch_manifest" \
    --expected-strategy "$strategy" \
    --headless "$headless" \
    "${video_arguments[@]}"
' "$INSTALL_SETUP" "$REPOSITORY_ROOT" "$CURRENT_BATCH" "$strategy" \
            "$HEADLESS" "$SELECTED_GPU_UUID" "$video_output"
        adapter_pid=$STARTED_PID
        adapter_pgid=$STARTED_PGID

        CURRENT_STAGE="batch_adapter_wait"
        wait_owned_with_timeout "$adapter_pid" "$adapter_pgid" "$BATCH_TIMEOUT_SECONDS" "dynamic Isaac adapter"
        if ((CAPTURE_SINGLE_EPISODE_VIDEOS)); then
            require_regular_nonempty_file "$video_output" "native viewport video"
        fi
        stop_owned_process_group "$executor_pid" "$executor_pgid"

        CURRENT_STAGE="batch_artifact_validation"
        "$HOST_PYTHON" "$SUPPORT_SCRIPT" validate-batch-artifacts \
            --repository-root "$REPOSITORY_ROOT" \
            --batch "$CURRENT_BATCH"
    done
    CURRENT_BATCH=""
}


run_replay_analysis_and_log_archive() {
    CURRENT_STAGE="replay_validation"
    run_pixi_bash_logged "$LOG_DIRECTORY/replay_validate.log" '
set -Eeuo pipefail
install_setup=$1
repository_root=$2
suite=$3
replay=$4
source "$install_setup"
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
cd "$repository_root"
python -m action_stream_benchmark.m8_cli validate \
    --manifest "$suite" \
    --output "$replay"
' "$INSTALL_SETUP" "$REPOSITORY_ROOT" "$SUITE_PATH" "$REPLAY_PATH"

    if [[ "$SUITE_SPLIT" == "frozen_holdout" ]]; then
        CURRENT_STAGE="analysis_and_figures"
        run_pixi_bash_logged "$LOG_DIRECTORY/analysis_figures.log" '
set -Eeuo pipefail
install_setup=$1
repository_root=$2
suite=$3
replay=$4
analysis=$5
figures=$6
source "$install_setup"
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
cd "$repository_root"
python -m action_stream_benchmark.m8_cli analyze \
    --manifest "$suite" \
    --replay "$replay" \
    --output "$analysis" \
    --bootstrap-resamples 20000
python -m action_stream_benchmark.m8_cli figures \
    --analysis "$analysis" \
    --output-dir "$figures"
' "$INSTALL_SETUP" "$REPOSITORY_ROOT" "$SUITE_PATH" "$REPLAY_PATH" "$ANALYSIS_PATH" "$FIGURE_DIRECTORY"
    fi

    CURRENT_STAGE="process_log_archive"
    PROCESS_LOG_ARCHIVE_PATH="$LOG_DIRECTORY/process_logs.tar.gz"
    PROCESS_LOG_MANIFEST_PATH="$LOG_DIRECTORY/process_logs.manifest.json"
    local member_list="$LOG_DIRECTORY/.process_logs.members.tmp"
    require_absent "$PROCESS_LOG_ARCHIVE_PATH" "process-log archive"
    require_absent "$PROCESS_LOG_MANIFEST_PATH" "process-log archive manifest"
    require_absent "$member_list" "temporary process-log member list"
    local -a members=()
    mapfile -t members < <(
        find "$LOG_DIRECTORY" -maxdepth 1 -type f -name '*.log' -printf '%f\n' | LC_ALL=C sort
    )
    ((${#members[@]} >= 1)) || die "native run produced no process/command logs to archive"
    (set -o noclobber; printf '%s\n' "${members[@]}" >"$member_list")
    run_pixi_bash '
set -Eeuo pipefail
install_setup=$1
repository_root=$2
log_directory=$3
member_list=$4
archive=$5
manifest=$6
source "$install_setup"
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
cd "$repository_root"
python -m action_stream_benchmark.m8_cli archive \
    --root "$log_directory" \
    --member-list "$member_list" \
    --archive "$archive" \
    --manifest "$manifest"
python -m action_stream_benchmark.m8_cli archive-validate \
    --archive "$archive" \
    --manifest "$manifest"
' "$INSTALL_SETUP" "$REPOSITORY_ROOT" "$LOG_DIRECTORY" "$member_list" \
        "$PROCESS_LOG_ARCHIVE_PATH" "$PROCESS_LOG_MANIFEST_PATH"
    rm -f -- "$member_list"
    require_file "$PROCESS_LOG_ARCHIVE_PATH" "validated process-log archive"
    require_file "$PROCESS_LOG_MANIFEST_PATH" "validated process-log archive manifest"

    # Completion records the exact archived member count rather than counting
    # receipt/evidence files created later.
    PROCESS_LOG_MEMBER_COUNT=${#members[@]}
}


write_completion_receipt() {
    CURRENT_STAGE="completion_receipt"
    declare -a arguments=(
        "$SUPPORT_SCRIPT" write-completion
        --repository-root "$REPOSITORY_ROOT"
        --output "$COMPLETION_RECEIPT_PATH"
        --preflight-receipt "$PREFLIGHT_RECEIPT_PATH"
        --replay "$REPLAY_PATH"
        --process-log-archive "$PROCESS_LOG_ARCHIVE_PATH"
        --process-log-manifest "$PROCESS_LOG_MANIFEST_PATH"
        --process-log-member-count "$PROCESS_LOG_MEMBER_COUNT"
    )
    [[ -n "$ANALYSIS_PATH" ]] && arguments+=(--analysis "$ANALYSIS_PATH")
    "$HOST_PYTHON" "${arguments[@]}"
}


main() {
    parse_arguments "$@"
    resolve_inputs
    load_suite_records

    if ((CAPTURE_SINGLE_EPISODE_VIDEOS)); then
        local episode_count
        for episode_count in "${BATCH_EPISODE_COUNTS[@]}"; do
            [[ "$episode_count" == "1" ]] || \
                die "--capture-single-episode-videos requires exactly one episode per batch"
        done
    fi

    CURRENT_STAGE="output_guard"
    SUITE_DIRECTORY="$(dirname -- "$SUITE_PATH")"
    if ((CAPTURE_SINGLE_EPISODE_VIDEOS)); then
        VIDEO_DIRECTORY="$SUITE_DIRECTORY/videos"
        require_absent "$VIDEO_DIRECTORY" "native viewport video directory"
    fi
    REPLAY_PATH="$SUITE_DIRECTORY/replay_validation.json"
    ANALYSIS_PATH=""
    FIGURE_DIRECTORY=""
    require_absent "$REPLAY_PATH" "replay validation output"
    if [[ "$SUITE_SPLIT" == "frozen_holdout" ]]; then
        ANALYSIS_PATH="$SUITE_DIRECTORY/analysis.json"
        FIGURE_DIRECTORY="$SUITE_DIRECTORY/figures"
        require_absent "$ANALYSIS_PATH" "analysis output"
        require_absent "$FIGURE_DIRECTORY" "figure output directory"
    fi

    local log_root="$SUITE_DIRECTORY/native_run_logs"
    local run_id
    run_id="$(date -u +%Y%m%dT%H%M%S%NZ)"
    LOG_DIRECTORY="$log_root/$run_id"
    require_absent "$LOG_DIRECTORY" "native run log directory"
    mkdir -p -- "$log_root"
    mkdir -- "$LOG_DIRECTORY"
    PREFLIGHT_RECEIPT_PATH="$LOG_DIRECTORY/preflight_receipt.json"
    COMPLETION_RECEIPT_PATH="$LOG_DIRECTORY/completion_receipt.json"
    FAILURE_RECEIPT_PATH="$LOG_DIRECTORY/failure_receipt.json"
    TIMEOUT_RECEIPT_PATH="$LOG_DIRECTORY/timeout_receipt.json"
    GPU_REFUSAL_RECEIPT_PATH="$LOG_DIRECTORY/gpu_refusal_receipt.json"
    INITIAL_GPU_SNAPSHOT_PATH="$LOG_DIRECTORY/initial_gpu_preflight.json"
    POST_BUILD_GPU_SNAPSHOT_PATH="$LOG_DIRECTORY/post_build_gpu_preflight.json"
    EXTERNAL_ENVIRONMENT_EVIDENCE_PATH="$LOG_DIRECTORY/external_environment.json"

    validate_external_environment_inputs
    validate_current_source_inputs

    CURRENT_STAGE="gpu_preflight_initial"
    capture_gpu_snapshot "initial_pre_build" "$INITIAL_GPU_SNAPSHOT_PATH"

    build_current_source
    validate_post_build_activation_scripts
    write_external_environment_evidence

    CURRENT_STAGE="gpu_preflight_post_build"
    capture_gpu_snapshot "post_build_pre_launch" "$POST_BUILD_GPU_SNAPSHOT_PATH"

    bind_installed_runtime_and_write_preflight
    start_router
    run_native_batches
    stop_owned_process_group "$ROUTER_PID" "$ROUTER_PGID"
    run_replay_analysis_and_log_archive
    write_completion_receipt

    CURRENT_STAGE="complete"
    trap - ERR EXIT INT TERM HUP
    cleanup_owned_process_groups
    printf 'Completed native M8 suite; independent replay audit: %s\n' "$REPLAY_PATH"
    printf 'Preflight receipt: %s\n' "$PREFLIGHT_RECEIPT_PATH"
    printf 'Completion receipt: %s\n' "$COMPLETION_RECEIPT_PATH"
}


main "$@"
