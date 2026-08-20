from __future__ import annotations

import json

from actionstream.smolvla_rtc_v4_report import (
    PROFILE_ORDER,
    _effects,
    _gate_rows,
)


def _row(
    *,
    suite: str,
    state: int,
    runtime: str,
    profile: str,
    success: bool = True,
    steps: int = 100,
) -> dict[str, object]:
    return {
        "suite": suite,
        "task_id": 0,
        "initial_state_index": state,
        "seed": 1000 + state,
        "runtime": runtime,
        "delay_profile": profile,
        "success": success,
        "environment_steps": steps,
    }


def test_v4_gate_requires_two_successful_resets_in_three_families(tmp_path) -> None:
    rows = [
        _row(
            suite=suite,
            state=state,
            runtime="sync_hold",
            profile="fixed_0000",
        )
        for suite in ("libero_object", "libero_spatial", "libero_goal")
        for state in (29, 30)
    ]
    (tmp_path / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    loaded, gate = _gate_rows(tmp_path)
    assert len(loaded) == 6
    assert all(gate.values())


def test_v4_effects_are_paired_across_fifteen_resets_per_profile() -> None:
    rows: list[dict[str, object]] = []
    for profile in PROFILE_ORDER:
        for suite_index, suite in enumerate(
            ("libero_object", "libero_spatial", "libero_goal")
        ):
            for state in range(31, 36):
                for runtime, steps in (
                    ("sync_hold", 100),
                    ("lerobot_latest_only", 110),
                    ("lerobot_rtc", 90),
                ):
                    row = _row(
                        suite=suite,
                        state=state,
                        runtime=runtime,
                        profile=profile,
                        steps=steps + suite_index,
                    )
                    row["task_id"] = suite_index
                    rows.append(row)
    effects = _effects(rows)  # type: ignore[arg-type]
    assert len(effects) == 2 * len(PROFILE_ORDER)
    assert all(item["paired_n"] == 15 for item in effects)
    primary = next(
        item
        for item in effects
        if item["contrast"] == "primary" and item["profile"] == "fixed_0950"
    )
    assert primary["environment_steps_difference"] == -20.0
