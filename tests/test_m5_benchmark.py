from __future__ import annotations

import json
from pathlib import Path

import pytest

from actionstream import m5_benchmark


class FakeBackend:
    def __init__(self, **_: object) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeWorker:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def audit_args(root: Path, output: Path, *, resume: bool = False):
    argv = [
        "--phase=audit",
        "--task-id=0",
        "--displacement-mm=50",
        f"--seed-manifest={root / 'outputs/m5_g0/audit/runtime_smoke_seed_manifest.json'}",
        f"--config={root / 'configs/m5_g0.json'}",
        f"--task-audit={root / 'outputs/m5_g0/audit/task_entity_audit.json'}",
        f"--output-dir={output}",
        "--run-id=transaction-test",
    ]
    if resume:
        argv.append("--resume-after-crash")
    return m5_benchmark.parse_args(argv)


def fake_episode(**kwargs):
    condition = kwargs["condition"]
    episode = {
        "condition": condition,
        "success": True,
        "environment_steps": 1,
        "perturbation_valid": True,
        "stale_action_steps": 0,
    }
    return (
        episode,
        [{"condition": condition}],
        [{"condition": condition}],
        [{"condition": condition}],
    )


def test_episode_bundles_resume_same_contract_without_rerunning_completed(
    tmp_path,
    monkeypatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "run"
    calls: list[str] = []
    fail_gate = True

    def flaky_episode(*args, **kwargs):
        nonlocal fail_gate
        condition = kwargs["condition"]
        calls.append(condition)
        if condition == "aligned_oracle_pose_gate_shift" and fail_gate:
            fail_gate = False
            raise RuntimeError("synthetic infrastructure crash")
        return fake_episode(**kwargs)

    monkeypatch.setattr(m5_benchmark, "LeRobotBackend", FakeBackend)
    monkeypatch.setattr(m5_benchmark, "_make_worker", lambda *_: FakeWorker())
    monkeypatch.setattr(m5_benchmark, "_run_episode", flaky_episode)

    with pytest.raises(RuntimeError, match="synthetic infrastructure crash"):
        m5_benchmark.run(audit_args(root, output))
    crashed = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert crashed["status"] == "crashed"
    assert len(list((output / "episode_bundles").glob("*.json"))) == 1

    calls.clear()
    episodes = m5_benchmark.run(audit_args(root, output, resume=True))
    assert calls == ["aligned_oracle_pose_gate_shift"]
    assert len(episodes) == 2
    completed = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["attempt_count"] == 2
    assert len((output / "episodes.jsonl").read_text().splitlines()) == 2
    attempt_statuses = [
        json.loads(line)["status"]
        for line in (output / "attempts.jsonl").read_text().splitlines()
    ]
    assert attempt_statuses == ["started", "crashed", "started", "completed"]

    calls.clear()
    validated = m5_benchmark.run(audit_args(root, output, resume=True))
    assert len(validated) == 2
    assert calls == []
    assert [
        json.loads(line)["status"]
        for line in (output / "attempts.jsonl").read_text().splitlines()
    ] == attempt_statuses


def test_sealed_phase_rejects_calibration_manifest_before_backend(
    tmp_path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    args = m5_benchmark.parse_args(
        [
            "--phase=sealed",
            "--task-id=0",
            "--displacement-mm=50",
            f"--seed-manifest={root / 'outputs/m5_g0/protocol/calibration_seed_manifest.json'}",
            f"--config={root / 'configs/m5_g0.json'}",
            f"--task-audit={root / 'outputs/m5_g0/audit/task_entity_audit.json'}",
            f"--output-dir={tmp_path / 'wrong'}",
        ]
    )
    with pytest.raises(ValueError, match="requires manifest 'sealed_evaluation'"):
        m5_benchmark.run(args)
