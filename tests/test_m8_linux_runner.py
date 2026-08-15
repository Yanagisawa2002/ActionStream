from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
LINUX_RUNNER = ROOT / "scripts" / "m8_run_isaac.sh"
SUPPORT = ROOT / "scripts" / "m8_linux_runner_support.py"
WINDOWS_RUNNER = ROOT / "scripts" / "m8_run_isaac.ps1"
BASELINE_MATRIX = ROOT / "outputs" / "m8_g0" / "baseline_gate" / "candidate_0" / "matrix.json"


def _linux_source() -> str:
    return LINUX_RUNNER.read_text(encoding="utf-8")


def _support(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(SUPPORT), *arguments],
        capture_output=True,
        check=False,
        timeout=30,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bash_executable() -> str | None:
    candidates = []
    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidates.extend(
            (
                program_files / "Git" / "bin" / "bash.exe",
                program_files / "Git" / "usr" / "bin" / "bash.exe",
            )
        )
    discovered = shutil.which("bash")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        completed = subprocess.run(
            [str(candidate), "--version"],
            capture_output=True,
            check=False,
            timeout=10,
        )
        if completed.returncode == 0:
            return str(candidate)
    return None


def test_linux_runner_has_fail_closed_runtime_order() -> None:
    source = _linux_source()
    main = source[source.index("main() {") :]
    output_guard = main.index('CURRENT_STAGE="output_guard"')
    environment_validation = main.index("validate_external_environment_inputs")
    source_validation = main.index("validate_current_source_inputs")
    first_gpu_gate = main.index(
        'capture_gpu_snapshot "initial_pre_build" "$INITIAL_GPU_SNAPSHOT_PATH"'
    )
    build = main.index("build_current_source")
    activation_validation = main.index("validate_post_build_activation_scripts")
    environment_receipt = main.index("write_external_environment_evidence")
    second_gpu_gate = main.index(
        'capture_gpu_snapshot "post_build_pre_launch" "$POST_BUILD_GPU_SNAPSHOT_PATH"'
    )
    runtime_binding = main.index("bind_installed_runtime_and_write_preflight")
    router = main.index("start_router")
    batches = main.index("run_native_batches")
    replay = main.index("run_replay_analysis_and_log_archive")
    completion = main.index("write_completion_receipt")
    assert (
        output_guard
        < environment_validation
        < source_validation
        < first_gpu_gate
        < build
        < activation_validation
        < environment_receipt
        < second_gpu_gate
        < runtime_binding
        < router
        < batches
        < replay
        < completion
    )
    assert "--authorize-native-gpu-run" in source
    assert "AUTHORIZE_NATIVE_GPU_RUN=0" in source
    assert "native Isaac execution is fail-closed" in source
    assert 'CURRENT_STAGE="post_build_activation_validation"' in source
    assert '"$ISAAC_WORKSPACE/install/setup.bash"' in source
    assert '"$ISAAC_WORKSPACE/install/local_setup.bash"' in source
    assert 'INSTALL_SETUP="$ISAAC_WORKSPACE/install/setup.bash"' in source
    assert '[[ -f "$path" && ! -L "$path" && -s "$path" ]]' in source
    assert source.count("capture_gpu_snapshot ") == 2
    assert "--validate-only" in source
    assert "freeze-validate" in source
    assert "colcon build" in source
    assert "--cmake-clean-cache" in source
    assert "-DPython_FIND_VIRTUALENV=ONLY" in source
    assert "-DPython3_FIND_VIRTUALENV=ONLY" in source
    assert "action_stream_benchmark.m8_cli validate" in source
    assert "action_stream_benchmark.m8_cli analyze" in source
    assert "action_stream_benchmark.m8_cli figures" in source
    assert "action_stream_benchmark.m8_cli archive-validate" in source
    assert source.count('run --frozen --manifest-path "$ISAAC_WORKSPACE/pixi.toml"') == 4
    assert 'require_file "$ISAAC_WORKSPACE/pixi.lock"' in source


def test_linux_runner_uses_only_the_pinned_pixi_python_when_host_python_is_absent() -> None:
    source = _linux_source()
    resolve_inputs = source[source.index("resolve_inputs() {") : source.index("load_suite_records() {")]
    assert 'HOST_PYTHON="$(command -v python3 || command -v python || true)"' in resolve_inputs
    assert 'if [[ -z "$HOST_PYTHON" ]]; then' in resolve_inputs
    assert '"$ISAAC_WORKSPACE/.pixi/envs/default/bin/python3"' in resolve_inputs
    assert '"$ISAAC_WORKSPACE/.pixi/envs/default/bin/python"' in resolve_inputs
    assert 'HOST_PYTHON="$(realpath -e -- "$environment_python")"' in resolve_inputs
    assert "the pinned Pixi default environment" in resolve_inputs
    assert "sys.version_info >= (3, 9)" in resolve_inputs


def test_linux_runner_owns_only_isolated_process_groups() -> None:
    source = _linux_source()
    assert "setsid \"$PIXI_EXE\"" in source
    assert "OWNED_PIDS" in source
    assert "OWNED_PGIDS" in source
    assert 'pgid" != "$pid' in source
    assert 'kill -TERM -- "-$pgid"' in source
    assert 'kill -KILL -- "-$pgid"' in source
    assert 'stop_owned_process_group "$pid" "$pgid"' in source
    assert "stopped_only_owned_process_groups" in SUPPORT.read_text(encoding="utf-8")
    lowered = source.casefold()
    assert "pkill" not in lowered
    assert "killall" not in lowered
    assert "--gpu-reset" not in lowered
    assert "powershell" not in lowered
    assert "local_setup.ps1" not in lowered
    assert "action_stream_executor_node.exe" not in lowered
    assert "rmw_zenohd.exe" not in lowered
    assert 'export PYTHONPATH="$benchmark_source:$isaac_source:$policy_source' in source


def test_linux_runner_launches_only_the_preflighted_gpu_uuid() -> None:
    source = _linux_source()
    assert 'SELECTED_GPU_UUID=""' in source
    assert '[[ "$selected_uuid" == GPU-* ]]' in source
    assert "selected GPU identity changed between preflights" in source
    assert "' \"$ISAAC_WORKSPACE\" \"$ROUTER\" \"$SELECTED_GPU_UUID\"" in source
    assert (
        "' \"$INSTALL_SETUP\" \"$EXECUTOR\" \"$strategy\" \"$SELECTED_GPU_UUID\""
        in source
    )
    assert '"$HEADLESS" "$SELECTED_GPU_UUID"' in source
    assert 'export CUDA_VISIBLE_DEVICES="$gpu_uuid"' in source
    assert 'export CUDA_VISIBLE_DEVICES="$gpu_index"' not in source


def test_linux_runner_resolves_the_standard_ros_package_executable_layout() -> None:
    source = _linux_source()
    assert "ros2 pkg prefix rmw_zenoh_cpp" in source
    assert 'router="$package_prefix/lib/rmw_zenoh_cpp/rmw_zenohd"' in source
    assert '[[ -x "$router" ]]' in source
    assert 'set +u\nsource "$1"\nset -u' in source
    assert source.count('set +u\nsource "$install_setup"\nset -u') == 5


def test_linux_runner_captures_only_unambiguous_single_episode_batches() -> None:
    source = _linux_source()
    assert "--capture-single-episode-videos" in source
    assert (
        'die "--capture-single-episode-videos requires exactly one episode per batch"'
        in source
    )
    assert 'require_absent "$VIDEO_DIRECTORY" "native viewport video directory"' in source
    assert 'video_arguments+=(--video-output "$video_output")' in source
    assert (
        'require_regular_nonempty_file "$video_output" "native viewport video"'
        in source
    )


def _load_support_module():
    spec = importlib.util.spec_from_file_location("m8_linux_runner_support", SUPPORT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_environment_constants_are_exactly_pinned() -> None:
    module = _load_support_module()
    assert module.ISAAC_WORKSPACE_REPOSITORY_URL == (
        "https://github.com/isaac-sim/IsaacSim-ros_workspaces.git"
    )
    assert module.ISAAC_WORKSPACE_COMMIT == (
        "dd3eeede7912755996a18f4884285d9f50843f79"
    )
    assert module.ISAAC_WORKSPACE_RELATIVE_PATH == "jazzy_ws"
    assert module.OFFICIAL_LINUX_PIXI_VERSION_OUTPUT == "pixi 0.75.0"
    assert module.OFFICIAL_LINUX_PIXI_EXECUTABLE_SIZE_BYTES == 77_311_024
    assert module.OFFICIAL_LINUX_PIXI_EXECUTABLE_SHA256 == (
        "4383aed18b2d5569cf34a19638daf954aa4415cc87ad3a9da9f34059cc4a004c"
    )
    assert module.OFFICIAL_WORKSPACE_FILE_SPECS["pixi.toml"] == {
        "canonical_lf_size_bytes": 5_467,
        "canonical_lf_sha256": (
            "b4e7a34c264e88f19ba6bfb3c7a72ee46b0843f3e6eb7e75619dc0ebb87b313d"
        ),
        "exact_crlf_size_bytes": 5_618,
        "exact_crlf_sha256": (
            "9649bf57644781a1fe0203ed6b80828ccb42ea07555475080e5d11a9b0c3e1ae"
        ),
        "evidence_name": "isaac_workspace.pixi.toml",
    }
    assert module.OFFICIAL_WORKSPACE_FILE_SPECS["pixi.lock"] == {
        "canonical_lf_size_bytes": 1_491_808,
        "canonical_lf_sha256": (
            "ba8e59eef962cbf49a1ff06ff947ed5eaa4547d018389b048771a1e7d8bb890d"
        ),
        "exact_crlf_size_bytes": 1_533_045,
        "exact_crlf_sha256": (
            "2c2f9097b129847735b5abb805045a22076a731e2138c8192caac95d09a2866e"
        ),
        "evidence_name": "isaac_workspace.pixi.lock",
    }
    assert module.EXPECTED_RUNTIME_VERSIONS == {
        "python_version": "3.12.13",
        "isaacsim": "6.0.1.0",
        "isaacsim-app": "6.0.1.0",
        "isaacsim-core": "6.0.1.0",
        "isaacsim-robot": "6.0.1.0",
        "isaacsim-ros2": "6.0.1.0",
        "rclpy": "7.1.9",
        "rosgraph-msgs": "2.0.3",
        "ros_distribution": "jazzy",
        "rmw_implementation": "rmw_zenoh_cpp",
        "rmw_zenoh_cpp": "0.2.9",
    }


def test_support_accepts_only_exact_lf_or_all_crlf_workspace_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_support_module()
    lf = b"first\nsecond\n"
    crlf = lf.replace(b"\n", b"\r\n")
    monkeypatch.setattr(
        module,
        "OFFICIAL_WORKSPACE_FILE_SPECS",
        {
            "pixi.toml": {
                "canonical_lf_size_bytes": len(lf),
                "canonical_lf_sha256": hashlib.sha256(lf).hexdigest(),
                "exact_crlf_size_bytes": len(crlf),
                "exact_crlf_sha256": hashlib.sha256(crlf).hexdigest(),
                "evidence_name": "isaac_workspace.pixi.toml",
            }
        },
    )
    manifest = tmp_path / "pixi.toml"
    manifest.write_bytes(lf)
    assert module._official_workspace_file_record(
        manifest, filename="pixi.toml"
    )["byte_form"] == "lf"
    manifest.write_bytes(crlf)
    assert module._official_workspace_file_record(
        manifest, filename="pixi.toml"
    )["byte_form"] == "crlf"
    manifest.write_bytes(lf.replace(b"\n", b"\r\n", 1))
    with pytest.raises(ValueError, match="exact LF Git blob or its exact all-CRLF"):
        module._official_workspace_file_record(manifest, filename="pixi.toml")


def test_support_rejects_mixed_workspace_newline_pair() -> None:
    module = _load_support_module()
    with pytest.raises(ValueError, match="same exact newline form"):
        module._require_matching_workspace_byte_forms(
            {
                "pixi.toml": {"byte_form": "lf"},
                "pixi.lock": {"byte_form": "crlf"},
            }
        )


def test_gpu_snapshot_blocks_unknown_memory_compute_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_support_module()
    executable = tmp_path / "nvidia-smi"
    executable.write_bytes(b"fake")

    def fake_nvidia_smi(_executable: Path, arguments: tuple[str, ...]) -> str:
        if arguments[0].startswith("--query-gpu="):
            return "0, NVIDIA RTX 5090, GPU-selected, 999.1, 32607, 100, 0\n"
        return "GPU-selected, 4242, python, N/A\n"

    monkeypatch.setattr(module, "_run_nvidia_smi", fake_nvidia_smi)
    output = tmp_path / "snapshot.json"
    result = module._gpu_snapshot(
        SimpleNamespace(
            nvidia_smi=str(executable),
            gpu_index=0,
            memory_threshold_mib=4096,
            phase="test",
            output=str(output),
        )
    )
    assert result == 3
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["blocking_compute_processes"][0]["process_id"] == 4242
    assert payload["unknown_memory_compute_processes"][0]["used_memory_mib"] is None


def test_gpu_snapshot_rejects_malformed_compute_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_support_module()
    executable = tmp_path / "nvidia-smi"
    executable.write_bytes(b"fake")

    def fake_nvidia_smi(_executable: Path, arguments: tuple[str, ...]) -> str:
        if arguments[0].startswith("--query-gpu="):
            return "0, NVIDIA RTX 5090, GPU-selected, 999.1, 32607, 100, 0\n"
        return "GPU-selected, missing-fields\n"

    monkeypatch.setattr(module, "_run_nvidia_smi", fake_nvidia_smi)
    with pytest.raises(ValueError, match="unexpected nvidia-smi compute-process row"):
        module._gpu_snapshot(
            SimpleNamespace(
                nvidia_smi=str(executable),
                gpu_index=0,
                memory_threshold_mib=4096,
                phase="test",
                output=str(tmp_path / "snapshot.json"),
            )
        )


def test_linux_runner_emits_formal_portable_evidence_and_windows_is_nonformal() -> None:
    linux = _linux_source()
    windows = WINDOWS_RUNNER.read_text(encoding="utf-8")
    helper = SUPPORT.read_text(encoding="utf-8")
    for field in (
        "installed_dynamic_adapter_evidence",
        "installed_dynamic_adapter_evidence_sha256",
        "runner_source",
        "runner_evidence",
        "runner_evidence_sha256",
        "runner_sha256",
    ):
        assert f'"{field}"' in helper
        assert field in windows
    assert '"runner": str(runner)' in helper
    assert "runner = $PSCommandPath" in windows
    assert "scripts/m8_run_isaac.sh" in linux
    assert "copy-evidence" in linux
    assert "m8_run_isaac.runner.ps1" in windows
    for field in (
        "runner_support_source",
        "runner_support_evidence",
        "runner_support_evidence_sha256",
        "runner_support_sha256",
    ):
        assert f'"{field}"' in helper
        assert field not in windows
    assert "m8_linux_runner_support.py" in linux
    assert "external_environment_evidence" in helper
    assert "write-external-environment" in linux
    assert "external_environment_evidence" not in windows


def test_linux_runner_parses_with_bash() -> None:
    bash = _bash_executable()
    if bash is None:
        pytest.skip("a working Bash executable is unavailable")
    completed = subprocess.run(
        [bash, "-n", str(LINUX_RUNNER)],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_support_extracts_exact_frozen_baseline_inventory() -> None:
    completed = _support(
        "suite-records",
        "--repository-root",
        str(ROOT),
        "--suite",
        str(BASELINE_MATRIX),
    )
    assert completed.returncode == 0, completed.stderr.decode()
    fields = completed.stdout.rstrip(b"\0").decode("utf-8").split("\0")
    assert fields[0:2] == ["baseline_gate", ""]
    assert len(fields) == 7
    assert Path(fields[2]).resolve() == (
        BASELINE_MATRIX.parent / "batch_profile_0_sanity_sync_hold.json"
    ).resolve()
    assert fields[3:] == [
        "sync_hold",
        "batch_profile_0_sanity_sync_hold",
        "profile_0_sanity",
        "20",
    ]


def test_support_accepts_in_repository_parent_reference_for_holdout_freeze(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    holdout = root / "outputs" / "holdout"
    protocol = root / "outputs" / "protocol"
    holdout.mkdir(parents=True)
    protocol.mkdir(parents=True)
    freeze = protocol / "freeze_manifest.json"
    freeze.write_text("{}\n", encoding="utf-8")
    batch = holdout / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "batch_strategy": "sync_hold",
                "batch_profile_id": "profile_0_sanity",
                "expected_episode_count": 1,
                "episodes": [{}],
            }
        ),
        encoding="utf-8",
    )
    suite = holdout / "matrix.json"
    suite.write_text(
        json.dumps(
            {
                "milestone": "M8-G0",
                "split": "frozen_holdout",
                "native_results_status_at_creation": "not_run",
                "freeze_manifest": "../protocol/freeze_manifest.json",
                "persistent_batch_count": 1,
                "batch_manifests": [{"path": "batch.json"}],
            }
        ),
        encoding="utf-8",
    )

    completed = _support(
        "suite-records",
        "--repository-root",
        str(root),
        "--suite",
        str(suite),
    )
    assert completed.returncode == 0, completed.stderr.decode()
    fields = completed.stdout.rstrip(b"\0").decode("utf-8").split("\0")
    assert fields[0] == "frozen_holdout"
    assert Path(fields[1]) == freeze.resolve()


def test_support_rejects_batch_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    suite = root / "matrix.json"
    suite.write_text(
        json.dumps(
            {
                "milestone": "M8-G0",
                "native_results_status_at_creation": "not_run",
                "split": "baseline_gate",
                "persistent_batch_count": 1,
                "batch_manifests": [{"path": "../outside.json"}],
            }
        ),
        encoding="utf-8",
    )
    completed = _support(
        "suite-records",
        "--repository-root",
        str(root),
        "--suite",
        str(suite),
    )
    assert completed.returncode != 0
    assert b"not an allowed portable relative path" in completed.stderr


@pytest.mark.parametrize("unsafe_path", [r"..\outside.json", r"C:\outside.json"])
def test_support_rejects_cross_platform_batch_path_escape(
    tmp_path: Path, unsafe_path: str
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    suite = root / "matrix.json"
    suite.write_text(
        json.dumps(
            {
                "milestone": "M8-G0",
                "native_results_status_at_creation": "not_run",
                "split": "baseline_gate",
                "persistent_batch_count": 1,
                "batch_manifests": [{"path": unsafe_path}],
            }
        ),
        encoding="utf-8",
    )
    completed = _support(
        "suite-records",
        "--repository-root",
        str(root),
        "--suite",
        str(suite),
    )
    assert completed.returncode != 0
    assert b"portable relative path" in completed.stderr


def test_support_source_manifest_matches_frozen_algorithm(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    source_root = root / "ros2_ws" / "src"
    package = source_root / "package"
    package.mkdir(parents=True)
    first = package / "a.py"
    second = package / "b.txt"
    first.write_bytes(b"alpha\n")
    second.write_bytes(b"beta\n")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ignored")
    output = tmp_path / "source_manifest.txt"
    completed = _support(
        "source-manifest",
        "--repository-root",
        str(root),
        "--source-root",
        str(source_root),
        "--output",
        str(output),
    )
    assert completed.returncode == 0, completed.stderr.decode()
    fields = completed.stdout.rstrip(b"\0").decode("utf-8").split("\0")
    records = [
        f"ros2_ws/src/package/a.py|{_sha256(first)}",
        f"ros2_ws/src/package/b.txt|{_sha256(second)}",
    ]
    expected_digest = hashlib.sha256("\n".join(records).encode()).hexdigest()
    assert fields == ["2", expected_digest, "ros2_ws/src"]
    assert output.read_text(encoding="utf-8").splitlines() == records


def test_support_writes_hash_bound_portable_preflight(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    receipt_dir = root / "outputs" / "native_run_logs" / "run"
    source_root = root / "ros2_ws" / "src"
    scripts = root / "scripts"
    receipt_dir.mkdir(parents=True)
    source_root.mkdir(parents=True)
    scripts.mkdir()
    suite = root / "matrix.json"
    suite.write_text("{}\n", encoding="utf-8")
    runner = scripts / "m8_run_isaac.sh"
    runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    runner_support = scripts / "m8_linux_runner_support.py"
    runner_support.write_text("SUPPORT = True\n", encoding="utf-8")
    installed_adapter = tmp_path / "install" / "dynamic_isaac_adapter.py"
    installed_adapter.parent.mkdir()
    installed_adapter.write_text("adapter = True\n", encoding="utf-8")
    executor = tmp_path / "install" / "action_stream_executor_node"
    router = tmp_path / "install" / "rmw_zenohd"
    executor.write_bytes(b"executor")
    router.write_bytes(b"router")
    adapter_evidence = receipt_dir / "installed_dynamic_adapter.py"
    executor_evidence = receipt_dir / "action_stream_executor_node"
    runner_evidence = receipt_dir / "m8_run_isaac.runner.sh"
    runner_support_evidence = receipt_dir / "m8_linux_runner_support.py"
    for source, destination in (
        (installed_adapter, adapter_evidence),
        (executor, executor_evidence),
        (runner, runner_evidence),
        (runner_support, runner_support_evidence),
    ):
        copied = _support(
            "copy-evidence",
            "--source",
            str(source),
            "--destination",
            str(destination),
        )
        assert copied.returncode == 0, copied.stderr.decode()
    snapshot_payload = {
        "phase": "initial_pre_build",
        "selected_gpu_index": 0,
        "passed": True,
        "gpu_inventory": [
            {
                "index": 0,
                "name": "RTX 5090",
                "uuid": "GPU-test",
                "driver_version": "999.1",
                "memory_total_mib": 32607,
                "memory_used_mib": 0,
                "utilization_gpu_percent": 0,
            }
        ],
        "reported_compute_processes": [],
        "actionable_compute_processes": [],
        "blocking_compute_processes": [],
        "unknown_memory_compute_processes": [],
        "occupied_gpus": [],
    }
    initial = receipt_dir / "initial.json"
    post = receipt_dir / "post.json"
    initial.write_text(json.dumps(snapshot_payload), encoding="utf-8")
    post_payload = {**snapshot_payload, "phase": "post_build_pre_launch"}
    post.write_text(json.dumps(post_payload), encoding="utf-8")
    external_environment = receipt_dir / "external_environment.json"
    external_environment.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "milestone": "M8-G0",
                "evidence_kind": "native_external_environment",
            }
        ),
        encoding="utf-8",
    )
    receipt = receipt_dir / "preflight_receipt.json"
    completed = _support(
        "write-preflight",
        "--repository-root",
        str(root),
        "--output",
        str(receipt),
        "--suite",
        str(suite),
        "--source-root",
        str(source_root),
        "--source-file-count",
        "0",
        "--source-manifest-sha256",
        hashlib.sha256(b"").hexdigest(),
        "--external-environment",
        str(external_environment),
        "--memory-threshold-mib",
        "4096",
        "--gpu-index",
        "0",
        "--initial-snapshot",
        str(initial),
        "--post-snapshot",
        str(post),
        "--no-frozen-live-inputs-validated",
        "--validation-log",
        "batch.validate_only.log",
        "--installed-adapter",
        str(installed_adapter),
        "--installed-adapter-evidence",
        str(adapter_evidence),
        "--executor",
        str(executor),
        "--executor-evidence",
        str(executor_evidence),
        "--router",
        str(router),
        "--runner",
        str(runner),
        "--runner-evidence",
        str(runner_evidence),
        "--runner-support",
        str(runner_support),
        "--runner-support-evidence",
        str(runner_support_evidence),
        "--batch-timeout-seconds",
        "14400",
        "--planned-log",
        "batch.validate_only.log",
    )
    assert completed.returncode == 0, completed.stderr.decode()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["runner_source"] == "scripts/m8_run_isaac.sh"
    assert payload["runner"] == str(runner.resolve())
    assert payload["runner_evidence"] == "m8_run_isaac.runner.sh"
    assert payload["runner_sha256"] == payload["runner_evidence_sha256"]
    assert payload["runner_support_source"] == "scripts/m8_linux_runner_support.py"
    assert payload["runner_support_evidence"] == "m8_linux_runner_support.py"
    assert payload["runner_support_sha256"] == payload[
        "runner_support_evidence_sha256"
    ]
    assert payload["installed_dynamic_adapter_evidence"] == (
        "installed_dynamic_adapter.py"
    )
    assert payload["installed_dynamic_adapter_sha256"] == payload[
        "installed_dynamic_adapter_evidence_sha256"
    ]
    assert payload["executor_evidence"] == "action_stream_executor_node"
    assert payload["executor_sha256"] == payload["executor_evidence_sha256"]
    assert payload["preexisting_compute_processes"] == []
    assert payload["selected_gpu_uuid"] == "GPU-test"
    assert payload["selected_gpu_identity"]["name"] == "RTX 5090"
    assert payload["external_environment_evidence"] == "external_environment.json"


def test_copy_evidence_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("source\n", encoding="utf-8")
    destination.write_text("existing\n", encoding="utf-8")
    completed = _support(
        "copy-evidence",
        "--source",
        str(source),
        "--destination",
        str(destination),
    )
    assert completed.returncode != 0
    assert destination.read_text(encoding="utf-8") == "existing\n"
