from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/actionstream_transport_h0_exact_lerobot_canary.json"
ORCHESTRATOR_PATH = (
    ROOT / "scripts/experiments/run_transport_h0_exact_lerobot_canary.py"
)
RESUME_V2_PATH = (
    ROOT / "scripts/experiments/run_transport_h0_exact_lerobot_canary_v2.py"
)
RESUME_V2_CONFIG_PATH = (
    ROOT / "configs/actionstream_transport_h0_exact_lerobot_canary_v2.json"
)
REPORT_ROOT = ROOT / "reports/actionstream_transport_h0_closeout"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module():
    spec = importlib.util.spec_from_file_location(
        "transport_h0_canary", ORCHESTRATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v2_module():
    spec = importlib.util.spec_from_file_location(
        "transport_h0_canary_v2", RESUME_V2_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transport_h0_canary_freezes_only_the_import_prerequisite() -> None:
    module = _module()
    raw = module._validate_protocol(PROTOCOL_PATH)
    assert raw["candidate"]["orchestrator_sha256"] == _sha256(ORCHESTRATOR_PATH)
    assert raw["runtimes"] == ["actionstream_backend_pipelined_aligned"]
    assert raw["environment"]["initial_state_indices"] == [47]
    assert raw["compact_matrix"]["episodes_per_task"] == 1
    assert raw["evidence_boundary"]["formal_holdout"] is False
    assert raw["evidence_boundary"]["performance_or_superiority_claim_allowed"] is False
    assert raw["decision_rule"]["task_success_required"] is False
    assert raw["decision_rule"]["no_h1_performance_conclusion"] is True


def test_transport_h0_canary_places_exact_lerobot_source_first(tmp_path: Path) -> None:
    module = _module()
    lerobot = tmp_path / "lerobot"
    inherited = os.pathsep.join([str(tmp_path / "old"), str(tmp_path / "other")])
    entries = module._pythonpath(lerobot, inherited).split(os.pathsep)
    assert entries[:2] == [
        str((lerobot / "src").resolve()),
        str((ROOT / "src").resolve()),
    ]
    assert entries[2:] == [str(tmp_path / "old"), str(tmp_path / "other")]


def test_transport_h0_canary_result_gate_is_not_a_task_success_gate(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "output"
    traces = output / "traces"
    videos = output / "videos"
    traces.mkdir(parents=True)
    videos.mkdir()
    trace = traces / "episode.json"
    telemetry = traces / "episode.telemetry.jsonl"
    video = videos / "episode.mp4"
    trace.write_text("{}", encoding="utf-8")
    telemetry.write_text("{}\n", encoding="utf-8")
    video.write_bytes(b"video")
    receipt = {
        "selection": {
            "runtimes": ["actionstream_backend_pipelined_aligned"],
            "profiles": ["fixed_0950_transport_h0_canary"],
        },
        "system_gpu": {
            "sample_count": 2,
            "phases": {
                "steady_state": {
                    "resident_sample_count": 1,
                    "process_gpu_memory_mib_max": 100.0,
                }
            },
        },
    }
    (output / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    row = {
        "status": "completed",
        "runtime": "actionstream_backend_pipelined_aligned",
        "inference_completed": 2,
        "environment_steps": 3,
        "action_discontinuity_max_l2": 0.5,
        "responses_scheduled": 2,
        "responses_delivered": 1,
        "inference_errors": 0,
        "success": False,
        "trace_path": str(trace),
        "telemetry_jsonl_path": str(telemetry),
        "video_path": str(video),
    }
    (output / "episodes.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = module._validate_result(output)
    assert result["status"] == "PASS"
    assert all(result["gates"].values())
    assert result["task_success_observed_not_gated"] is False


def test_transport_h0_canary_v2_is_bounded_to_preflight_resume() -> None:
    raw = json.loads(RESUME_V2_CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = raw["frozen_artifacts"]
    assert raw["status"] == "frozen_preflight_resume_before_gpu_child"
    assert frozen["v1_config_sha256"] == _sha256(PROTOCOL_PATH)
    assert frozen["v1_orchestrator_sha256"] == _sha256(ORCHESTRATOR_PATH)
    assert frozen["v2_orchestrator_sha256"] == _sha256(RESUME_V2_PATH)
    assert raw["preflight_failure"]["gpu_child_process_started"] is False
    assert raw["preflight_failure"]["development_episode_started"] is False
    assert raw["resume_contract"]["gpu_child_processes_allowed"] == 1
    assert raw["resume_contract"]["development_episodes_allowed"] == 1
    assert raw["resume_contract"]["retry_after_gpu_child_start_allowed"] is False
    assert all(raw["unchanged_scientific_contract"].values())


def test_transport_h0_canary_v2_preserves_venv_entrypoint(monkeypatch) -> None:
    module = _v2_module()
    captured = {}

    def fake_preflight(**kwargs):
        captured.update(kwargs)
        return {"status": "fixture"}

    monkeypatch.setattr(module, "_V1_PREFLIGHT", fake_preflight)
    result = module._venv_preflight(
        python=Path("/incorrect/resolved/python"),
        lerobot_root=Path("/lerobot"),
        environment={"PYTHONPATH": "fixture"},
    )
    assert result == {"status": "fixture"}
    assert captured["python"] == Path(sys.executable)


def test_transport_h0_closeout_keeps_h0_pass_separate_from_h1() -> None:
    summary = json.loads((REPORT_ROOT / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (REPORT_ROOT / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    report = (REPORT_ROOT / "report.md").read_text(encoding="utf-8")
    assert summary["status"] == "H0_PASS_H1_UNRESOLVED"
    assert summary["h0"]["verdict"] == "PASS"
    assert summary["h0"]["formal_holdout"] is False
    assert summary["h0"]["task_success_is_gate"] is False
    assert summary["h1"]["verdict"] == "UNRESOLVED_NOT_PAIRED"
    assert summary["h1"]["paired_identity_count"] == 0
    assert summary["h1"]["performance_conclusion_allowed"] is False
    assert summary["validation"]["focused_protocol_and_closeout_tests_passed"] == 15
    assert summary["validation"]["remote_full_repository_suite"].startswith("PASS")
    assert summary["validation"]["video_content_inspected"] is True
    image = REPORT_ROOT / "qualitative_progression.png"
    assert manifest["tracked_qualitative_progression"]["sha256"] == _sha256(image)
    assert manifest["raw_archive"]["tracked_in_git"] is False
    assert "H0 PASS; H1 UNRESOLVED / NOT PAIRED" in report
    assert "Production soak / real remote service / real robot safety" in report
