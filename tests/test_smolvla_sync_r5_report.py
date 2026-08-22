from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from actionstream.current_baselines import load_protocol
from actionstream.smolvla_sync_r5_report import generate


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(suite: str, task_id: int, state: int, *, success: bool) -> dict[str, object]:
    return {
        "suite": suite,
        "task_id": task_id,
        "initial_state_index": state,
        "runtime": "sync_hold",
        "delay_profile": "fixed_0000",
        "sync_execution_mode": "receding_horizon",
        "sync_execution_horizon_steps": 10,
        "chunk_size": 50,
        "success": success,
    }


def _write_phase(root: Path, rows: list[dict[str, object]], passed: bool) -> None:
    root.mkdir()
    (root / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "orchestrator_receipt.json").write_text(
        json.dumps({"gate_passed": passed}), encoding="utf-8"
    )


def test_r5_protocols_pin_one_change_and_sequential_data_boundary() -> None:
    paths = {
        "repair_spatial": (
            "development_repair_only",
            "libero_spatial",
            0,
            [29],
            2026090200,
        ),
        "repair_goal": (
            "development_repair_only",
            "libero_goal",
            5,
            [30],
            2026090301,
        ),
        "canary_object": (
            "frozen_sync_canary",
            "libero_object",
            3,
            [48, 49],
            2026100100,
        ),
        "canary_spatial": (
            "frozen_sync_canary",
            "libero_spatial",
            0,
            [48, 49],
            2026100200,
        ),
        "canary_goal": (
            "frozen_sync_canary",
            "libero_goal",
            5,
            [48, 49],
            2026100300,
        ),
    }
    canary_identities: set[tuple[str, int, int, int]] = set()
    for key, (status, suite, task_id, states, base_seed) in paths.items():
        protocol = load_protocol(
            ROOT / "configs" / f"actionstream_backend_gpu_smolvla_sync_r5_{key}.json"
        )
        assert protocol.raw["smolvla_sync_protocol_version"] == 5
        assert protocol.raw["protocol_status"] == status
        assert protocol.raw["runtimes"] == ["sync_hold"]
        assert protocol.raw["sync_execution"] == {
            "mode": "receding_horizon",
            "horizon_steps": 10,
            "predicted_chunk_steps": 50,
            "hypothesis_id": "SMOLVLA_SYNC_R5_HORIZON_10",
        }
        assert protocol.models["smolvla"].chunk_size == 50
        assert protocol.raw["environment"]["suite"] == suite
        assert protocol.raw["environment"]["task_ids"] == [task_id]
        assert protocol.raw["environment"]["initial_state_indices"] == states
        assert protocol.raw["environment"]["base_seed"] == base_seed
        assert set(protocol.delays) == {"fixed_0000"}
        candidate = protocol.raw["candidate"]
        assert candidate["base_commit"] == ("162ac3bb6578b7093a21f2d12d457916266c35bb")
        assert candidate["current_baselines_sha256"] == _sha256(
            ROOT / "src/actionstream/current_baselines.py"
        )
        assert candidate["orchestrator_sha256"] == _sha256(
            ROOT / "scripts/experiments/run_smolvla_sync_r5.py"
        )
        assert candidate["report_sha256"] == _sha256(
            ROOT / "src/actionstream/smolvla_sync_r5_report.py"
        )
        if status == "frozen_sync_canary":
            canary_identities.update(
                (suite, task_id, state, base_seed + index)
                for index, state in enumerate(states)
            )

    assert len(canary_identities) == 6
    assert len({identity[3] for identity in canary_identities}) == 6


def test_r5_report_requires_repair_before_canary(tmp_path: Path) -> None:
    repair = tmp_path / "repair"
    _write_phase(
        repair,
        [
            _row("libero_spatial", 0, 29, success=False),
            _row("libero_goal", 5, 30, success=True),
        ],
        False,
    )
    canary = tmp_path / "canary"
    _write_phase(
        canary,
        [
            _row(suite, task_id, state, success=True)
            for suite, task_id in (
                ("libero_object", 3),
                ("libero_spatial", 0),
                ("libero_goal", 5),
            )
            for state in (48, 49)
        ],
        True,
    )

    with pytest.raises(RuntimeError, match="after a failed R5 repair gate"):
        generate(repair, canary, tmp_path / "report")


def test_r5_report_marks_pass_without_claiming_rtc(tmp_path: Path) -> None:
    repair = tmp_path / "repair"
    _write_phase(
        repair,
        [
            _row("libero_spatial", 0, 29, success=True),
            _row("libero_goal", 5, 30, success=True),
        ],
        True,
    )
    canary = tmp_path / "canary"
    _write_phase(
        canary,
        [
            _row(suite, task_id, state, success=True)
            for suite, task_id in (
                ("libero_object", 3),
                ("libero_spatial", 0),
                ("libero_goal", 5),
            )
            for state in (48, 49)
        ],
        True,
    )

    summary = generate(repair, canary, tmp_path / "report")

    assert summary["status"] == "PASS_SYNC_GATE"
    assert summary["formal_rtc_status"] == "ELIGIBLE_FOR_SEPARATE_FREEZE"
    assert "not itself an RTC result" in (tmp_path / "report" / "report.md").read_text(
        encoding="utf-8"
    )
