from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from actionstream.current_baselines import load_protocol
from actionstream.transport_h1 import load_transport_h1_protocol


ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return {
        "schema_version": 1,
        "protocol_status": "frozen_before_formal_result",
        "hypothesis_id": "transport_h1_serialized_delivery",
        "shared_contract": {
            "control_frequency_hz": 20.0,
            "chunk_size": 30,
            "compute_latency_milliseconds": 150.0,
            "delivery_delay_milliseconds": 950.0,
            "scored_control_steps": 120,
            "repetitions": 3,
            "bounded_hold_steps": 2,
            "latest_only_fallback": False,
        },
        "variants": {
            "serialized": {"delivery_scheduler_enabled": False},
            "pipelined": {"delivery_scheduler_enabled": True},
        },
        "decision_gates": {
            "minimum_request_rate_ratio": 4.0,
            "minimum_depletion_reduction_fraction": 0.9,
            "maximum_pipelined_depletion_fraction": 0.02,
            "maximum_out_of_order_rejections": 0,
            "maximum_fallback_activations": 0,
        },
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_transport_h1_loader_preserves_single_variable_contrast(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    _write(path, _protocol())

    loaded = load_transport_h1_protocol(path)

    assert loaded.control_frequency_hz == 20.0
    assert loaded.chunk_size == 30
    assert loaded.compute_latency_seconds == 0.15
    assert loaded.delivery_delay_seconds == 0.95
    assert loaded.repetitions == 3


def test_transport_h1_loader_rejects_second_variant_change(tmp_path: Path) -> None:
    raw = _protocol()
    raw["variants"]["pipelined"]["latest_only_fallback"] = True
    path = tmp_path / "invalid.json"
    _write(path, raw)

    with pytest.raises(ValueError, match="only enable"):
        load_transport_h1_protocol(path)


def test_frozen_transport_h1_candidate_hashes_pre_result_code() -> None:
    protocol = load_transport_h1_protocol(
        ROOT / "configs/actionstream_transport_h1.json"
    )
    candidate = protocol.raw["candidate"]
    commit = candidate["commit"]
    paths = {
        "lerobot_inference_sha256": "src/actionstream/lerobot_inference.py",
        "current_baselines_sha256": "src/actionstream/current_baselines.py",
        "transport_h1_sha256": "src/actionstream/transport_h1.py",
    }
    for field, path in paths.items():
        completed = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        assert candidate[field] == hashlib.sha256(completed.stdout).hexdigest()


def test_gpu_canary_is_frozen_on_legal_unused_reset_after_fixture_pass() -> None:
    invalidation = json.loads(
        (
            ROOT
            / "configs/actionstream_transport_h1_gpu_canary_invalidated.json"
        ).read_text(encoding="utf-8")
    )
    protocol = load_protocol(
        ROOT / "configs/actionstream_transport_h1_gpu_canary.json"
    )
    raw = protocol.raw

    assert invalidation["status"] == "invalidated_pre_gpu_result"
    assert invalidation["gpu_episodes_started"] == 0
    assert invalidation["policy_results_observed_before_invalidation"] is False
    assert raw["fixture_gate_receipt"]["verdict"] == "PASS"
    assert raw["environment"]["initial_state_indices"] == [39]
    assert raw["evidence_boundary"][
        "state_39_absent_from_prior_xvla_task5_scored_and_warmup_identities"
    ] is True
    assert set(raw["runtimes"]) == {
        "actionstream_backend_aligned",
        "actionstream_backend_pipelined_aligned",
    }
    assert raw["decision_rule"]["policy_success_is_not_a_gate"] is True


def test_tracked_transport_h1_summary_preserves_canary_claim_boundary() -> None:
    summary = json.loads(
        (
            ROOT / "reports/actionstream_transport_h1/summary.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["status"] == "PASS_WITH_CANARY_SCOPE"
    assert summary["tests"] == {
        "collected": 334,
        "passed": 322,
        "skipped": 12,
        "failed": 0,
        "exit_code": 0,
        "remote_log_sha256": (
            "b5a9b9e0e7ddc9faa3c9bdcba2c3af7a2dad1342b6a45acdaaf738bdfd0cd90b"
        ),
    }
    assert summary["gpu_canary"]["gpu_sessions"] == 1
    assert summary["gpu_canary"]["contrast"][
        "all_frozen_system_gates_passed"
    ] is True
    assert summary["gpu_canary"]["contrast"][
        "policy_success_is_descriptive_only"
    ] is True
    assert summary["artifacts"]["raw_root_git_ignored"] is True
    assert "no multi-task CI" in summary["claim_boundary"]
