"""Report the bounded SmolVLA R5 receding-horizon sync hypothesis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPECTED_REPAIR_IDENTITIES = {
    ("libero_spatial", 0, 29),
    ("libero_goal", 5, 30),
}
EXPECTED_CANARY_IDENTITIES = {
    (suite, task_id, state)
    for suite, task_id in (
        ("libero_object", 3),
        ("libero_spatial", 0),
        ("libero_goal", 5),
    )
    for state in (48, 49)
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _identity(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["suite"]),
        int(row["task_id"]),
        int(row["initial_state_index"]),
    )


def _validate_rows(
    rows: list[dict[str, Any]], expected: set[tuple[str, int, int]], label: str
) -> None:
    identities = {_identity(row) for row in rows}
    if identities != expected or len(rows) != len(expected):
        raise RuntimeError(
            f"Unexpected {label} identities: expected={sorted(expected)}, "
            f"actual={sorted(identities)} records={len(rows)}"
        )
    for row in rows:
        if row.get("runtime") != "sync_hold":
            raise RuntimeError(f"{label} contains a non-sync runtime")
        if row.get("delay_profile") != "fixed_0000":
            raise RuntimeError(f"{label} contains a nonzero/stochastic delay")
        if row.get("sync_execution_mode") != "receding_horizon":
            raise RuntimeError(f"{label} did not use receding-horizon sync")
        if int(row.get("sync_execution_horizon_steps", 0)) != 10:
            raise RuntimeError(f"{label} execution horizon drifted")
        if int(row.get("chunk_size", 0)) != 50:
            raise RuntimeError(f"{label} predicted chunk size drifted")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(
    repair_root: Path, canary_root: Path | None, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    repair_rows = _read_jsonl(repair_root / "episodes.jsonl")
    repair_receipt = _read_json(repair_root / "orchestrator_receipt.json")
    _validate_rows(repair_rows, EXPECTED_REPAIR_IDENTITIES, "repair")
    repair_passed = bool(
        repair_receipt.get("gate_passed") is True
        and all(bool(row.get("success")) for row in repair_rows)
    )

    canary_rows: list[dict[str, Any]] = []
    canary_passed: bool | None = None
    if canary_root is not None:
        if not repair_passed:
            raise RuntimeError("Canary evidence exists after a failed R5 repair gate")
        canary_rows = _read_jsonl(canary_root / "episodes.jsonl")
        canary_receipt = _read_json(canary_root / "orchestrator_receipt.json")
        _validate_rows(canary_rows, EXPECTED_CANARY_IDENTITIES, "canary")
        canary_passed = bool(
            canary_receipt.get("gate_passed") is True
            and all(bool(row.get("success")) for row in canary_rows)
        )

    if not repair_passed:
        status = "NO_GO_REPAIR"
    elif canary_root is None:
        status = "REPAIR_PASS_CANARY_SEALED"
    elif canary_passed:
        status = "PASS_SYNC_GATE"
    else:
        status = "NO_GO_CANARY"
    formal_rtc_status = (
        "ELIGIBLE_FOR_SEPARATE_FREEZE"
        if status == "PASS_SYNC_GATE"
        else "NOT_RUN_BY_SYNC_GATE"
    )
    summary = {
        "schema_version": 1,
        "hypothesis_id": "SMOLVLA_SYNC_R5_HORIZON_10",
        "status": status,
        "single_changed_variable": {
            "from": "50-step full-chunk open-loop sync",
            "to": "10-step receding-horizon sync",
            "predicted_chunk_steps_unchanged": 50,
        },
        "repair": {
            "records": len(repair_rows),
            "successes": sum(bool(row.get("success")) for row in repair_rows),
            "passed": repair_passed,
        },
        "canary": {
            "records": len(canary_rows),
            "successes": sum(bool(row.get("success")) for row in canary_rows),
            "passed": canary_passed,
        },
        "formal_rtc_status": formal_rtc_status,
        "claim_boundary": (
            "This gate can establish only zero-delay SmolVLA sync stability under "
            "a 10-step receding horizon on three pinned LIBERO tasks. It cannot "
            "establish an RTC effect, latency robustness, or cross-task generality."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_csv(output_dir / "repair_table.csv", repair_rows)
    if canary_rows:
        _write_csv(output_dir / "canary_table.csv", canary_rows)
    report = f"""# SmolVLA sync R5: one-hypothesis gate

Verdict: **{status}**.

## Hypothesis

The v4 failures were caused by executing all 50 predicted actions open loop.
R5 changes only the execution horizon to 10 steps; checkpoint, predicted chunk,
tasks, observations, action space, success predicate, and zero-delay profile are
unchanged.

## Sequential gates

- Known-failure repair: {summary["repair"]["successes"]}/{summary["repair"]["records"]}.
- Unseen three-family canary: {summary["canary"]["successes"]}/{summary["canary"]["records"]}.
- Formal RTC: `{formal_rtc_status}`.

If repair is not 2/2, canary must not run. If canary is not 6/6, RTC must not
run. Passing this gate permits only a separately frozen RTC hypothesis; it is
not itself an RTC result.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--canary-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = generate(
        args.repair_root.resolve(),
        args.canary_root.resolve() if args.canary_root else None,
        args.output_dir.resolve(),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
