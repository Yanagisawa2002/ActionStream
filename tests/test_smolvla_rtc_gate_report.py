from __future__ import annotations

from actionstream.smolvla_rtc_gate_report import (
    _build_artifact_manifest,
    build_canary_table,
    evaluate_gates,
)


def _row(version: str, suite: str, episode: int, success: bool) -> dict:
    task = {"libero_object": 0, "libero_spatial": 0, "libero_goal": 5}[suite]
    return {
        "experiment_id": f"actionstream_backend_gpu_smolvla_rtc_canary_{suite.removeprefix('libero_')}_{version}",
        "model_key": "smolvla",
        "runtime": "sync_hold",
        "delay_profile": "fixed_0000",
        "suite": suite,
        "task_id": task,
        "episode_index": episode,
        "initial_state_index": 45 + episode,
        "success": success,
        "environment_steps": 100 if success else 300,
        "inference_latency_p50_seconds": 0.2,
        "inference_latency_p95_seconds": 0.25,
        "peak_cuda_memory_mib": 1000.0,
    }


def test_v3_gate_fails_if_one_suite_is_one_of_two() -> None:
    rows = []
    for version in ("v2", "v3"):
        for suite in ("libero_object", "libero_spatial", "libero_goal"):
            rows.extend(_row(version, suite, episode, True) for episode in (0, 1))
    rows[-6 + 1]["success"] = False
    rows[-6 + 1]["environment_steps"] = 300

    table = build_canary_table(rows)
    gates = evaluate_gates(rows, table)

    assert gates["smolvla_v2_three_suite_sync_gate"] is True
    assert gates["smolvla_v3_three_suite_sync_gate"] is False
    assert gates["smolvla_v3_suite_successes"]["libero_object"] == "1/2"
    assert gates["formal_holdout_record_count"] == 0
    assert gates["formal_rtc_paired_effect_available"] is False


def test_artifact_manifest_excludes_impossible_self_hash(tmp_path) -> None:
    (tmp_path / "report.md").write_text("evidence\n", encoding="utf-8")
    (tmp_path / "artifact_manifest.json").write_text(
        '{"stale": true}\n', encoding="utf-8"
    )

    manifest = _build_artifact_manifest(tmp_path)

    assert set(manifest) == {"report.md"}
    assert manifest["report.md"]["bytes"] == (tmp_path / "report.md").stat().st_size
