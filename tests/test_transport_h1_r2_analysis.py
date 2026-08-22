from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/actionstream_transport_h1_r2"
ANALYSIS_PATH = ROOT / "scripts/analysis/generate_transport_h1_r2_report.py"


def _analysis():
    spec = importlib.util.spec_from_file_location(
        "generate_transport_h1_r2_report", ANALYSIS_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_h1_r2_tracked_report_preserves_the_frozen_verdict() -> None:
    summary = json.loads((REPORT_ROOT / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "GO"
    assert summary["decision"] == {
        "verdict": "GO",
        "all_gates_passed": True,
        "request_rate_ratio": 11.427425381085444,
        "serialized_depletion_fraction": 0.46589403973509935,
        "pipelined_depletion_fraction": 0.0,
        "depletion_reduction_fraction": 1.0,
        "families_with_lower_depletion": 3,
        "serialized_successes": 10,
        "pipelined_successes": 13,
        "out_of_order_rejections": 0,
        "fallback_activations": 0,
    }
    assert len(summary["gates"]) == 7
    assert all(gate["passed"] for gate in summary["gates"])
    assert summary["integrity"]["raw_manifest"]["entry_count"] == 99
    assert summary["integrity"]["contract"]["row_count"] == 30
    assert summary["evidence"]["videos_local_and_verified"] == 6


def test_h1_r2_report_does_not_hide_the_task_regression_or_compute_cost() -> None:
    summary = json.loads((REPORT_ROOT / "summary.json").read_text(encoding="utf-8"))
    success = next(
        effect
        for effect in summary["paired_effects"]
        if effect["metric"] == "success_difference"
    )
    assert success["bootstrap_95_ci_low"] < 0 < success["bootstrap_95_ci_high"]
    object_rows = {
        row["runtime"]: row
        for row in summary["task_table"]
        if row["family"] == "object"
    }
    assert object_rows["actionstream_backend_aligned"]["successes"] == 5
    assert object_rows["actionstream_backend_pipelined_aligned"]["successes"] == 3
    aligned_calls = summary["aggregates"]["actionstream_backend_aligned"][
        "inference_calls_sum"
    ]
    pipelined_calls = summary["aggregates"]["actionstream_backend_pipelined_aligned"][
        "inference_calls_sum"
    ]
    assert pipelined_calls / aligned_calls > 7.9
    assert "No real robot" in " ".join(summary["limitations"])


def test_h1_r2_curated_manifest_matches_every_listed_file() -> None:
    manifest = json.loads(
        (REPORT_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
    )["artifacts"]
    assert "report.md" in manifest
    assert "summary.json" in manifest
    assert "paired_content_contact.png" in manifest
    for name, receipt in manifest.items():
        path = REPORT_ROOT / name
        assert path.stat().st_size == receipt["bytes"]
        assert _sha256(path) == receipt["sha256"]


def test_h1_r2_paired_bootstrap_is_deterministic() -> None:
    module = _analysis()
    pairs = []
    for index, difference in enumerate((1, 0, -1, 1, 0)):
        serialized = {
            "success": difference < 1,
            "environment_steps": 100,
            "wall_clock_episode_seconds": 10.0,
            "inference_requests_per_second": 1.0,
        }
        pipelined = {
            "success": bool(int(serialized["success"]) + difference),
            "environment_steps": 90 + index,
            "wall_clock_episode_seconds": 8.0,
            "inference_requests_per_second": 5.0 + index,
        }
        pairs.append((("suite", 0, index, index), serialized, pipelined))
    first = module.paired_effects(pairs, resamples=1_000, seed=2026102295)
    second = module.paired_effects(pairs, resamples=1_000, seed=2026102295)
    assert first == second
    assert first[1]["paired_unit"] == (
        "suite + task_id + initial_state_index + policy seed"
    )
