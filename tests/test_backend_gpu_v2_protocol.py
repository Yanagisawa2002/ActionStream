from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from actionstream.current_baselines import load_protocol


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_object_sha256(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def test_gpu_v2_registry_is_preserved_as_invalid_seed_freeze() -> None:
    registry = json.loads(
        (ROOT / "configs/actionstream_backend_gpu_v2_registry.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry["candidate_commit"] == ("b3da160f7fe22cc3182e5e1991c27e5cdac5e12a")
    assert registry["reset_identity_audit"]["prior_episode_identity_hits"] == 0
    assert registry["seeded_trace_audit"][
        "disjoint_from_all_other_loadable_config_seeded_traces"
    ]

    for item in registry["protocols"].values():
        path = ROOT / item["path"]
        assert _sha256(path) == item["sha256"]
        with pytest.raises(ValueError, match="NumPy/LeRobot seed range"):
            load_protocol(path)


def test_gpu_v2r1_freezes_valid_new_process_and_disjoint_evidence() -> None:
    registry = json.loads(
        (ROOT / "configs/actionstream_backend_gpu_v2r1_registry.json").read_text(
            encoding="utf-8"
        )
    )
    invalidation = json.loads(
        (ROOT / "configs/actionstream_backend_gpu_v2_invalidated.json").read_text(
            encoding="utf-8"
        )
    )
    assert invalidation["status"] == "invalidated_pre_result"
    assert invalidation["scored_episode_count"] == 0
    assert registry["candidate_commit"] == ("a00f580a0956474612b3639e5eccc3d0cf61db8b")
    assert registry["reset_identity_audit"]["prior_episode_identity_hits"] == 0

    protocols = []
    for item in registry["protocols"].values():
        path = ROOT / item["path"]
        assert _sha256(path) == item["sha256"]
        protocols.append(load_protocol(path))

    assert {item.raw["task_family"] for item in protocols} == {
        "object",
        "spatial",
        "goal",
    }
    candidate = protocols[0].raw["candidate"]
    candidate_commit = registry["candidate_commit"]
    assert candidate["current_baselines_sha256"] == _git_object_sha256(
        candidate_commit, "src/actionstream/current_baselines.py"
    )
    assert candidate["lerobot_inference_sha256"] == _git_object_sha256(
        candidate_commit, "src/actionstream/lerobot_inference.py"
    )
    assert candidate["gpu_measurement_sha256"] == _git_object_sha256(
        candidate_commit, "src/actionstream/gpu_measurement.py"
    )
    assert candidate["orchestrator_sha256"] == _git_object_sha256(
        candidate_commit, "scripts/experiments/run_backend_gpu_v2.py"
    )
    assert candidate["report_sha256"] == _git_object_sha256(
        candidate_commit, "src/actionstream/backend_gpu_v2_report.py"
    )

    seeded_canary: set[str] = set()
    seeded_holdout: set[str] = set()
    for protocol in protocols:
        raw = protocol.raw
        assert raw["benchmark_version"] == 2
        assert raw["protocol_revision"] == 1
        assert 0 <= raw["environment"]["base_seed"] <= 2**32 - 1
        assert 0 <= raw["measurement_v2"]["worker_warmup"]["seed"] <= 2**32 - 1
        assert set(raw["runtimes"]) == {
            "sync_hold",
            "lerobot_weighted_average",
            "lerobot_latest_only",
            "actionstream_backend_aligned",
            "actionstream_backend_guarded",
        }
        assert raw["measurement_v2"]["fresh_process_per_runtime"] is True
        assert raw["measurement_v2"]["worker_warmup"]["inference_calls"] == 2
        assert raw["measurement_v2"]["worker_warmup"]["initial_state_index"] == 45
        seeded = {
            delay.trace_sha256
            for delay in protocol.delays.values()
            if str(delay.definition["kind"]).startswith("seeded_")
        }
        if raw["protocol_status"] == "frozen_canary":
            assert raw["environment"]["initial_state_indices"] == [46]
            assert raw["compact_matrix"]["episodes_per_task"] == 1
            assert "disconnect_recovery_v2_canary" in protocol.delays
            seeded_canary.update(seeded)
        else:
            assert raw["protocol_status"] == "frozen_holdout"
            assert raw["environment"]["initial_state_indices"] == [47, 48, 49]
            assert raw["compact_matrix"]["episodes_per_task"] == 3
            assert "disconnect_recovery_v2_canary" not in protocol.delays
            seeded_holdout.update(seeded)

    assert len(seeded_canary) == 6
    assert len(seeded_holdout) == 6
    assert seeded_canary.isdisjoint(seeded_holdout)
