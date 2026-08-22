from __future__ import annotations

import json
from pathlib import Path

import pytest

from actionstream.smolvla_sync_r5_report import generate


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
