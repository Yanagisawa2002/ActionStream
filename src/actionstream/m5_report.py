"""Build the auditable ActionStream M5-G0 report and publication figures.

The reporter is deliberately downstream of the simulator/runtime.  It never
changes experiment evidence and it does not infer a positive result from raw
plots.  A reviewed ``m5_analysis`` payload may be supplied verbatim; otherwise
the same analysis implementation is run over sealed/no-shift episode rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from actionstream.m5_analysis import (
    ALIGNED_NO_SHIFT,
    ALIGNED_SHIFT,
    CONDITIONS,
    GATED_NO_SHIFT,
    GATED_SHIFT,
    build_m5_analysis,
)
from actionstream.m5_protocol import (
    canonical_sha256,
    validate_disjoint,
    validate_manifest,
)


REPORT_SCHEMA_VERSION = 1
CLASSIFICATION_LABELS = {
    "FULL GO",
    "MECHANISM PASS / OUTCOME INCONCLUSIVE",
    "NO-GO",
}
CONDITION_ORDER = (
    ALIGNED_SHIFT,
    GATED_SHIFT,
    ALIGNED_NO_SHIFT,
    GATED_NO_SHIFT,
)
CONDITION_LABELS = {
    ALIGNED_SHIFT: "Aligned shift",
    GATED_SHIFT: "Oracle-gated shift",
    ALIGNED_NO_SHIFT: "Aligned no shift",
    GATED_NO_SHIFT: "Oracle-gated no shift",
}
FIGURE_FILENAMES = (
    "representative_paired_timeline.png",
    "representative_paired_timeline.pdf",
    "mechanism_outcome.png",
    "mechanism_outcome.pdf",
)


def _read_json(path: Path | str, *, role: str) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{role} is not valid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must be a JSON object: {source}")
    return payload


def _read_jsonl(
    paths: Iterable[Path | str],
    *,
    role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{role} contains invalid JSON at {path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{role} row at {path}:{line_number} must be an object"
                    )
                rows.append(row)
    return rows


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_entry(
    path: Path | str,
    *,
    role: str,
    row_count: int | None = None,
) -> dict[str, Any]:
    source = Path(path)
    entry: dict[str, Any] = {
        "role": role,
        "path": str(source),
        "sha256": _sha256(source),
        "bytes": source.stat().st_size,
    }
    if row_count is not None:
        entry["row_count"] = row_count
    return entry


def _ensure_milestone(payload: Mapping[str, Any], *, role: str) -> None:
    milestone = payload.get("milestone")
    if milestone is not None and milestone != "M5-G0":
        raise ValueError(f"{role} has unexpected milestone {milestone!r}")


def _validate_calibration_decision(
    payload: dict[str, Any],
    *,
    role: str,
) -> None:
    _ensure_milestone(payload, role=role)
    expected = payload.get("selection_sha256")
    if expected is not None:
        core = {
            key: value for key, value in payload.items() if key != "selection_sha256"
        }
        actual = canonical_sha256(core)
        if expected != actual:
            raise ValueError(
                f"{role} selection hash mismatch: expected {expected}, computed {actual}"
            )
    status = payload.get("status")
    if status not in {
        "selected",
        "needs_30_mm",
        "needs_70_mm",
        "candidate_no_go",
        "no_go",
    }:
        raise ValueError(f"{role} has unsupported calibration status {status!r}")


def _validate_canonical_hash(
    payload: dict[str, Any],
    *,
    hash_field: str,
    role: str,
) -> None:
    expected = payload.get(hash_field)
    core = {key: value for key, value in payload.items() if key != hash_field}
    actual = canonical_sha256(core)
    if expected != actual:
        raise ValueError(
            f"{role} hash mismatch: expected {expected}, computed {actual}"
        )


def _validate_action_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Check evaluator staleness labels against the recorded world epochs."""

    stale_executed = 0
    executed = 0
    discarded = 0
    discard_reasons: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        condition = row.get("condition")
        if condition not in CONDITIONS:
            raise ValueError(
                f"Action row {index} has unexpected condition {condition!r}"
            )
        condition_counts[str(condition)] += 1
        is_discarded = bool(row.get("discarded", row.get("record_type") == "discarded"))
        if is_discarded:
            discarded += 1
            reason = row.get("discard_reason")
            if reason:
                discard_reasons[str(reason)] += 1
            current_epoch = row.get("current_world_epoch_at_discard")
        else:
            executed += 1
            current_epoch = row.get("current_world_epoch_at_execution")
        observation_epoch = row.get("world_epoch_at_observation")
        if observation_epoch is None or current_epoch is None:
            raise ValueError(f"Action row {index} lacks exact world-epoch provenance")
        derived_stale = int(observation_epoch) < int(current_epoch)
        if "stale" in row and type(row["stale"]) is not bool:
            raise ValueError(f"Action row {index} has a non-boolean stale label")
        if "stale" in row and bool(row["stale"]) != derived_stale:
            raise ValueError(
                f"Action row {index} stale={row['stale']!r} disagrees with "
                f"world epochs {observation_epoch} < {current_epoch}"
            )
        if not is_discarded and derived_stale:
            stale_executed += 1
    return {
        "record_count": len(rows),
        "executed_action_count": executed,
        "discarded_action_count": discarded,
        "stale_executed_action_count": stale_executed,
        "discard_reason_counts": dict(sorted(discard_reasons.items())),
        "condition_record_counts": dict(sorted(condition_counts.items())),
        "world_epoch_staleness_recomputed": True,
    }


def _finite_values(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                values.append(number)
    return values


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "total": None,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    vector = np.asarray(values, dtype=np.float64)
    return {
        "count": int(vector.size),
        "total": float(vector.sum()),
        "mean": float(vector.mean()),
        "median": float(np.median(vector)),
        "minimum": float(vector.min()),
        "maximum": float(vector.max()),
    }


def _condition_summaries(
    episode_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        rows = [row for row in episode_rows if row.get("condition") == condition]
        invalid = [
            {
                "seed": row.get("seed"),
                "initial_state_index": row.get("initial_state_index"),
                "reason": row.get("invalid_reason"),
            }
            for row in rows
            if row.get("perturbation_valid") is False
        ]
        valid_rows = [
            row
            for row in rows
            if condition in {ALIGNED_NO_SHIFT, GATED_NO_SHIFT}
            or row.get("perturbation_valid") is not False
        ]
        successes = [row for row in valid_rows if row.get("success") is True]
        summaries.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "episode_count": len(rows),
                "valid_episode_count": len(valid_rows),
                "success_count": len(successes),
                "success_rate": (
                    len(successes) / len(valid_rows) if valid_rows else None
                ),
                "stale_action_steps": _numeric_summary(
                    _finite_values(rows, "stale_action_steps")
                ),
                "stale_action_duration_seconds": _numeric_summary(
                    _finite_values(rows, "stale_action_duration_seconds")
                ),
                "environment_steps": _numeric_summary(
                    _finite_values(rows, "environment_steps")
                ),
                "simulated_completion_time_seconds": _numeric_summary(
                    _finite_values(rows, "simulated_completion_time_seconds")
                ),
                "wall_clock_episode_seconds": _numeric_summary(
                    _finite_values(rows, "wall_clock_episode_seconds")
                ),
                "stale_policy_results_discarded": _numeric_summary(
                    _finite_values(rows, "stale_policy_results_discarded")
                ),
                "stale_queued_actions_discarded": _numeric_summary(
                    _finite_values(rows, "stale_queued_actions_discarded")
                ),
                "fresh_replan_count": _numeric_summary(
                    _finite_values(rows, "fresh_replan_count")
                ),
                "hold_steps_introduced_by_gate": _numeric_summary(
                    _finite_values(rows, "hold_steps_introduced_by_gate")
                ),
                "maximum_queue_depth": _numeric_summary(
                    _finite_values(rows, "maximum_queue_depth")
                ),
                "action_source_observation_age_steps_mean": _numeric_summary(
                    _finite_values(rows, "action_source_observation_age_steps_mean")
                ),
                "time_from_shift_to_gate_trigger_seconds": _numeric_summary(
                    _finite_values(rows, "time_from_shift_to_gate_trigger_seconds")
                ),
                "time_from_shift_to_queue_clear_seconds": _numeric_summary(
                    _finite_values(rows, "time_from_shift_to_queue_clear_seconds")
                ),
                "time_from_shift_to_first_fresh_action_seconds": _numeric_summary(
                    _finite_values(
                        rows, "time_from_shift_to_first_fresh_action_seconds"
                    )
                ),
                "detector_max_translation_delta_m": _numeric_summary(
                    _finite_values(rows, "detector_max_translation_delta_m")
                ),
                "detector_max_rotation_delta_rad": _numeric_summary(
                    _finite_values(rows, "detector_max_rotation_delta_rad")
                ),
                "scene_gate_trigger_count": sum(
                    int(row.get("scene_gate_trigger_count", 0)) for row in rows
                ),
                "queue_invalidation_count": sum(
                    int(row.get("queue_invalidation_count", 0)) for row in rows
                ),
                "false_gate_count": sum(
                    int(row.get("false_gate_count", 0)) for row in rows
                ),
                "moved_state_recovered_count": sum(
                    bool(row.get("moved_state_recovered")) for row in rows
                ),
                "termination_reason_counts": dict(
                    sorted(
                        Counter(
                            str(row.get("termination_reason", "unknown"))
                            for row in rows
                        ).items()
                    )
                ),
                "invalid_episode_count": len(invalid),
                "invalid_episodes": invalid,
            }
        )
    return summaries


def _selected_task(
    task_audit: Mapping[str, Any],
    config: Mapping[str, Any],
    selected_task_id: int | None = None,
) -> dict[str, Any]:
    selected = task_audit.get("selected_task")
    if isinstance(selected, dict) and (
        selected_task_id is None or int(selected.get("task_id", -1)) == selected_task_id
    ):
        return dict(selected)
    if selected_task_id is not None:
        for candidate in config.get("candidate_tasks", []):
            if (
                isinstance(candidate, dict)
                and int(candidate.get("task_id", -1)) == selected_task_id
            ):
                return dict(candidate)
    for candidate in task_audit.get("candidates_inspected", []):
        if isinstance(candidate, dict) and candidate.get("assessment") == "selected":
            return dict(candidate)
    for candidate in config.get("candidate_tasks", []):
        if isinstance(candidate, dict):
            return dict(candidate)
    return {}


def _calibration_summary(
    decisions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    selected = next(
        (
            decision
            for decision in reversed(decisions)
            if decision["status"] == "selected"
        ),
        None,
    )
    if selected is not None:
        return {
            "status": "selected",
            "selected_task_id": selected.get("task_id"),
            "selected_displacement_magnitude_mm": selected.get(
                "selected_displacement_magnitude_mm"
            ),
            "decision_count": len(decisions),
            "decisions": list(decisions),
            "hard_stop_reason": None,
        }
    terminal = decisions[-1]
    status = str(terminal["status"])
    if status in {"candidate_no_go", "no_go"}:
        reason = str(
            terminal.get("reason")
            or "No task/magnitude met the frozen behavioral challenge rule"
        )
        normalized_status = "no_go"
    else:
        reason = (
            f"Calibration is incomplete: {status}; sealed evaluation is not authorized"
        )
        normalized_status = "incomplete"
    return {
        "status": normalized_status,
        "selected_task_id": None,
        "selected_displacement_magnitude_mm": None,
        "decision_count": len(decisions),
        "decisions": list(decisions),
        "hard_stop_reason": reason,
    }


def _validate_protocol_calibration_decisions(
    protocol: Mapping[str, Any],
    decision_paths: Sequence[Path | str],
    decisions: Sequence[Mapping[str, Any]],
) -> None:
    expected = protocol.get("calibration_decisions")
    if not isinstance(expected, list):
        raise ValueError("Frozen protocol lacks calibration decision bindings")
    if len(expected) != len(decision_paths) or len(expected) != len(decisions):
        raise ValueError(
            "Supplied calibration decision count does not match the frozen protocol"
        )
    for index, (binding, path_value, decision) in enumerate(
        zip(expected, decision_paths, decisions, strict=True)
    ):
        if not isinstance(binding, Mapping):
            raise ValueError(
                f"Frozen calibration decision binding {index} is incomplete"
            )
        supplied_path = Path(path_value).resolve()
        recorded_path = Path(str(binding.get("path", ""))).resolve()
        checks = {
            "path": supplied_path == recorded_path,
            "sha256": binding.get("sha256") == _sha256(supplied_path),
            "selection_sha256": binding.get("selection_sha256")
            == decision.get("selection_sha256"),
            "task_id": binding.get("task_id") == decision.get("task_id"),
            "status": binding.get("status") == decision.get("status"),
        }
        failed = sorted(field for field, passed in checks.items() if not passed)
        if failed:
            raise ValueError(
                f"Calibration decision {index} does not match the frozen protocol: "
                f"{failed}"
            )


def _manifest_seed_sets(
    manifests: Sequence[dict[str, Any]],
) -> dict[str, set[tuple[int, int]]]:
    return {
        str(manifest["manifest_name"]): {
            (int(pair["seed"]), int(pair["initial_state_index"]))
            for pair in manifest["pairs"]
        }
        for manifest in manifests
    }


def _analysis_episode_rows(
    episode_rows: Sequence[dict[str, Any]],
    seed_sets: Mapping[str, set[tuple[int, int]]],
) -> list[dict[str, Any]]:
    """Admit only exact sealed/no-shift manifest pairs and declared phases."""

    expected = {
        ALIGNED_SHIFT: ("sealed_evaluation", "sealed"),
        GATED_SHIFT: ("sealed_evaluation", "sealed"),
        ALIGNED_NO_SHIFT: ("no_shift_regression", "no_shift"),
        GATED_NO_SHIFT: ("no_shift_regression", "no_shift"),
    }
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(episode_rows):
        condition = row.get("condition")
        if condition not in expected:
            raise ValueError(
                f"Episode evidence row {index} has unexpected condition {condition!r}"
            )
        manifest_name, phase = expected[str(condition)]
        key = (int(row.get("seed", -1)), int(row.get("initial_state_index", -1)))
        if key not in seed_sets.get(manifest_name, set()):
            raise ValueError(
                f"Episode evidence row {index} seed/state {key} is not in "
                f"{manifest_name}"
            )
        if row.get("phase") != phase:
            raise ValueError(
                f"Episode evidence row {index} phase {row.get('phase')!r} "
                f"does not match {phase!r}"
            )
        selected.append(row)
    return selected


def _extract_m4_baseline_flag(task_audit: Mapping[str, Any]) -> bool:
    direct = task_audit.get("m4_baseline_intact")
    if type(direct) is bool:
        return direct
    smoke = task_audit.get("m4_baseline_smoke")
    if isinstance(smoke, Mapping) and type(smoke.get("passed")) is bool:
        return bool(smoke["passed"])
    return False


def _validate_statistics(payload: dict[str, Any]) -> None:
    _ensure_milestone(payload, role="statistics")
    classification = payload.get("classification")
    if not isinstance(classification, dict):
        raise ValueError("Statistics payload has no classification object")
    label = classification.get("label")
    if label not in CLASSIFICATION_LABELS:
        raise ValueError(f"Statistics has unsupported classification {label!r}")


def _first_number(row: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _representative_pair(
    episode_rows: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in episode_rows:
        if row.get("condition") not in {ALIGNED_SHIFT, GATED_SHIFT}:
            continue
        key = (
            int(row.get("seed", -1)),
            int(row.get("initial_state_index", -1)),
            str(row["condition"]),
        )
        indexed[key] = row
    keys = sorted({(seed, state) for seed, state, _ in indexed})
    pairs: list[dict[str, Any]] = []
    for seed, state in keys:
        aligned = indexed.get((seed, state, ALIGNED_SHIFT))
        gate = indexed.get((seed, state, GATED_SHIFT))
        if aligned is not None and gate is not None:
            pairs.append(
                {
                    "seed": seed,
                    "initial_state_index": state,
                    ALIGNED_SHIFT: aligned,
                    GATED_SHIFT: gate,
                }
            )
    if not pairs:
        return None
    # Prefer the most causally legible gate-only success, then any discordance.
    pairs.sort(
        key=lambda pair: (
            not (
                pair[GATED_SHIFT].get("success") is True
                and pair[ALIGNED_SHIFT].get("success") is False
            ),
            pair[GATED_SHIFT].get("success") == pair[ALIGNED_SHIFT].get("success"),
            pair["seed"],
        )
    )
    return pairs[0]


def _row_matches_episode(
    row: Mapping[str, Any],
    episode: Mapping[str, Any],
) -> bool:
    if row.get("condition") != episode.get("condition"):
        return False
    episode_id = episode.get("episode_id")
    if episode_id is not None and row.get("episode_id") is not None:
        return row.get("episode_id") == episode_id
    return row.get("seed") == episode.get("seed") and (
        row.get("initial_state_index") is None
        or row.get("initial_state_index") == episode.get("initial_state_index")
    )


def _event_kind(row: Mapping[str, Any]) -> str:
    for key in ("event_type", "type", "event", "name"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value.lower().replace("-", "_").replace(" ", "_")
    return ""


def _event_step(row: Mapping[str, Any]) -> int | None:
    return _first_number(
        row,
        (
            "control_step",
            "event_step",
            "step",
            "shift_step",
            "observation_step",
            "observation_control_step",
            "result_arrival_step",
            "policy_result_arrival_step",
            "queue_execution_step",
            "discard_step",
        ),
    )


def _timeline_data(
    episode: Mapping[str, Any],
    *,
    action_rows: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    actions = [row for row in action_rows if _row_matches_episode(row, episode)]
    events = [row for row in event_rows if _row_matches_episode(row, episode)]
    queue_points: dict[int, float] = {}
    for row in actions:
        step = _first_number(row, ("queue_execution_step", "discard_step"))
        if step is None:
            continue
        depth = _first_number(
            row,
            (
                "queue_depth_before_action",
                "queue_depth_before_invalidation",
                "queue_depth_after_action",
            ),
        )
        if depth is not None:
            queue_points[step] = float(depth)

    markers: dict[tuple[int, str], dict[str, Any]] = {}

    def add_marker(
        step: int | None,
        label: str,
        color: str,
        marker: str,
    ) -> None:
        if step is not None and step >= 0:
            markers.setdefault(
                (step, label),
                {"step": step, "label": label, "color": color, "marker": marker},
            )

    # Preserve explicit runtime event records first.
    for event in events:
        kind = _event_kind(event)
        step = _event_step(event)
        if "perturb" in kind or kind in {"scene_shift", "shift"}:
            add_marker(step, "perturbation", "#D55E00", "X")
        elif "gate" in kind and ("trigger" in kind or "detect" in kind):
            add_marker(step, "gate trigger", "#CC79A7", "D")
        elif "queue" in kind and ("clear" in kind or "invalid" in kind):
            add_marker(step, "queue clear", "#009E73", "s")
        elif "result" in kind and ("arriv" in kind or "return" in kind):
            add_marker(step, "result arrival", "#E69F00", "v")
        elif "request" in kind or "observation" in kind:
            label = (
                "fresh request"
                if event.get("world_epoch_at_observation", 0)
                else "observation/request"
            )
            add_marker(step, label, "#0072B2", "^")
        elif "fresh" in kind and "action" in kind:
            add_marker(step, "first fresh action", "#009E73", "o")

    request_rows: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in actions:
        request_id = str(row.get("request_id", ""))
        generation = int(row.get("request_generation_id", 0))
        request_rows.setdefault((request_id, generation), row)
    for row in request_rows.values():
        observation_step = _first_number(
            row, ("observation_control_step", "observation_step")
        )
        is_fresh = int(row.get("world_epoch_at_observation", 0)) > 0
        add_marker(
            observation_step,
            "fresh request" if is_fresh else "observation/request",
            "#0072B2",
            "^",
        )
        add_marker(
            _first_number(row, ("policy_result_arrival_step", "result_arrival_step")),
            "fresh result arrival" if is_fresh else "result arrival",
            "#E69F00",
            "v",
        )

    shift_step = _first_number(
        episode,
        ("shift_step", "perturbation_step", "scene_shift_step"),
    )
    add_marker(shift_step, "perturbation", "#D55E00", "X")
    add_marker(
        _first_number(
            episode,
            ("gate_trigger_step", "scene_gate_trigger_step", "detection_step"),
        ),
        "gate trigger",
        "#CC79A7",
        "D",
    )
    add_marker(
        _first_number(
            episode,
            ("queue_clear_step", "queue_invalidation_step"),
        ),
        "queue clear",
        "#009E73",
        "s",
    )

    gate_rows = [row for row in actions if row.get("gate_triggered") is True]
    gate_steps = [
        step
        for row in gate_rows
        if (step := _first_number(row, ("discard_step", "queue_execution_step")))
        is not None
    ]
    if gate_steps:
        gate_step = min(gate_steps)
        add_marker(gate_step, "gate trigger", "#CC79A7", "D")
        clear_rows = [
            row for row in gate_rows if row.get("queue_depth_after_invalidation") == 0
        ]
        clear_steps = [
            step
            for row in clear_rows
            if (step := _first_number(row, ("discard_step",))) is not None
        ]
        if clear_steps:
            clear_step = min(clear_steps)
            add_marker(clear_step, "queue clear", "#009E73", "s")

    fresh_executed = [
        row
        for row in actions
        if not bool(row.get("discarded", row.get("record_type") == "discarded"))
        and int(row.get("world_epoch_at_observation", 0)) > 0
        and not bool(row.get("stale", False))
    ]
    if fresh_executed:
        first_fresh = min(
            fresh_executed,
            key=lambda row: int(row.get("queue_execution_step", 10**12)),
        )
        add_marker(
            _first_number(first_fresh, ("queue_execution_step",)),
            "first fresh action",
            "#009E73",
            "o",
        )

    terminal_step = _first_number(
        episode,
        ("terminal_step", "environment_steps"),
    )
    add_marker(
        terminal_step,
        "success" if episode.get("success") is True else "failure",
        "#009E73" if episode.get("success") is True else "#D55E00",
        "*" if episode.get("success") is True else "P",
    )
    return {
        "queue_points": sorted(queue_points.items()),
        "markers": sorted(
            markers.values(), key=lambda item: (item["step"], item["label"])
        ),
        "terminal_step": terminal_step,
    }


def _set_publication_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)


def _plot_representative_timeline(
    *,
    pair: dict[str, Any] | None,
    action_rows: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    output_dir: Path,
    unavailable_reason: str | None,
) -> None:
    _set_publication_style()
    if pair is None:
        fig, ax = plt.subplots(figsize=(7.2, 2.2))
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "Representative paired timeline unavailable",
            ha="center",
            va="center",
            weight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.38,
            unavailable_reason or "No complete shifted pair is present.",
            ha="center",
            va="center",
            wrap=True,
            transform=ax.transAxes,
        )
        _save_figure(fig, output_dir, "representative_paired_timeline")
        return

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    legend_specs: dict[str, tuple[str, str]] = {}
    for ax, condition in zip(axes, (ALIGNED_SHIFT, GATED_SHIFT), strict=True):
        episode = pair[condition]
        timeline = _timeline_data(
            episode,
            action_rows=action_rows,
            event_rows=event_rows,
        )
        points = timeline["queue_points"]
        if points:
            x, y = zip(*points, strict=True)
            ax.step(x, y, where="post", color="#4C78A8", linewidth=1.3)
            queue_top = max(y)
        else:
            terminal = timeline["terminal_step"] or 1
            ax.plot([0, terminal], [0, 0], color="#4C78A8", linewidth=1.0)
            queue_top = 0
        marker_gap = max(1.0, queue_top * 0.15)
        marker_y = queue_top + marker_gap
        occupied_levels: defaultdict[int, list[int]] = defaultdict(list)
        labeled_markers: list[tuple[dict[str, Any], int]] = []
        for marker in timeline["markers"]:
            step = int(marker["step"])
            level = 0
            while any(
                abs(step - occupied_step) <= 7
                for occupied_step in occupied_levels[level]
            ):
                level += 1
            occupied_levels[level].append(step)
            labeled_markers.append((marker, level))
        for marker, level in labeled_markers:
            step = int(marker["step"])
            y_value = marker_y + level * marker_gap
            ax.scatter(
                [step],
                [y_value],
                color=marker["color"],
                marker=marker["marker"],
                s=34,
                zorder=4,
            )
            legend_specs.setdefault(
                marker["label"],
                (marker["color"], marker["marker"]),
            )
        ax.set_ylabel(f"{CONDITION_LABELS[condition]}\nqueue depth")
        ax.grid(axis="x", color="0.90", linewidth=0.6)
        maximum_level = max((level for _, level in labeled_markers), default=0)
        top = marker_y + (maximum_level + 1.8) * marker_gap
        ax.set_ylim(-0.5, top)
    axes[-1].set_xlabel("Control step")
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle="none",
            markersize=5.5,
            label=label,
        )
        for label, (color, marker) in legend_specs.items()
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(5, len(handles)),
            frameon=False,
            columnspacing=0.9,
            handletextpad=0.3,
            fontsize=6.5,
        )
    fig.subplots_adjust(hspace=0.18, top=0.86)
    _save_figure(fig, output_dir, "representative_paired_timeline")


def _statistics_plot_values(
    statistics: Mapping[str, Any],
    condition_summaries: Sequence[dict[str, Any]],
) -> tuple[list[float | None], list[float | None]]:
    shift = statistics.get("shift_comparison")
    if isinstance(shift, Mapping):
        success = shift.get("success")
        stale = shift.get("stale_action_duration_seconds")
        if isinstance(success, Mapping) and isinstance(stale, Mapping):
            return (
                [
                    success.get("aligned", {}).get("success_rate"),
                    success.get("gate", {}).get("success_rate"),
                ],
                [
                    stale.get("reference", {}).get("mean"),
                    stale.get("estimate", {}).get("mean"),
                ],
            )
    by_condition = {summary["condition"]: summary for summary in condition_summaries}
    return (
        [
            by_condition[ALIGNED_SHIFT]["success_rate"],
            by_condition[GATED_SHIFT]["success_rate"],
        ],
        [
            by_condition[ALIGNED_SHIFT]["stale_action_duration_seconds"]["mean"],
            by_condition[GATED_SHIFT]["stale_action_duration_seconds"]["mean"],
        ],
    )


def _plot_mechanism_outcome(
    *,
    statistics: Mapping[str, Any],
    condition_summaries: Sequence[dict[str, Any]],
    calibration: Mapping[str, Any],
    output_dir: Path,
) -> None:
    _set_publication_style()
    success_rates, stale_means = _statistics_plot_values(
        statistics, condition_summaries
    )
    labels = ["Aligned\nshift", "Oracle gate\nshift"]
    colors = ["#4C78A8", "#E45756"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))

    if all(value is not None for value in success_rates):
        values = [100.0 * float(value) for value in success_rates]
        bars = axes[0].bar(labels, values, color=colors, width=0.64)
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel("Task success (%)")
        for bar, value in zip(bars, values, strict=True):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 2,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    else:
        evaluation = None
        for decision in reversed(calibration.get("decisions", [])):
            evaluations = decision.get("evaluations", [])
            if evaluations:
                evaluation = evaluations[-1]
                break
        if evaluation is not None:
            values = [
                20.0 * float(evaluation.get("aligned_success_count", 0)),
                20.0 * float(evaluation.get("gate_success_count", 0)),
            ]
            bars = axes[0].bar(labels, values, color=colors, width=0.64)
            axes[0].set_ylim(0, 105)
            axes[0].set_ylabel("Calibration success (%)")
            for bar, value in zip(bars, values, strict=True):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 2,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        else:
            axes[0].axis("off")
            axes[0].text(
                0.5,
                0.5,
                "Task outcome unavailable",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )

    if all(value is not None for value in stale_means):
        values = [float(value) for value in stale_means]
        bars = axes[1].bar(labels, values, color=colors, width=0.64)
        axes[1].set_ylabel("Mean stale-action duration (s)")
        upper = max(values) if values else 0.0
        axes[1].set_ylim(0, max(0.1, upper * 1.18))
        for bar, value in zip(bars, values, strict=True):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + max(0.01, upper * 0.03),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.55,
            "Mechanism metric unavailable",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            weight="bold",
        )
        axes[1].text(
            0.5,
            0.36,
            "Sealed stale-action rows were not run.",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    for ax in axes:
        ax.grid(axis="y", color="0.90", linewidth=0.6)
    fig.subplots_adjust(wspace=0.42)
    _save_figure(fig, output_dir, "mechanism_outcome")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _fmt_ci(value: Any, *, digits: int = 3) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return "n/a"
    if len(value) != 2:
        return "n/a"
    return f"[{_fmt(value[0], digits)}, {_fmt(value[1], digits)}]"


def _pct_ci(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return "n/a"
    if len(value) != 2:
        return "n/a"
    return f"[{_pct(value[0])}, {_pct(value[1])}]"


def _sealed_state(
    statistics: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    shift = statistics.get("shift_comparison")
    valid_pairs = (
        int(shift.get("valid_pair_count", 0)) if isinstance(shift, Mapping) else 0
    )
    if calibration.get("hard_stop_reason") is not None:
        return {
            "status": "not_run_due_to_calibration_hard_stop",
            "valid_pair_count": valid_pairs,
            "required_pair_count": 30,
            "reason": calibration["hard_stop_reason"],
        }
    return {
        "status": "complete" if valid_pairs == 30 else "incomplete",
        "valid_pair_count": valid_pairs,
        "required_pair_count": 30,
        "reason": (
            None if valid_pairs == 30 else "Thirty valid sealed pairs are required"
        ),
    }


def _default_limitations(
    *,
    task_audit: Mapping[str, Any],
    sealed: Mapping[str, Any],
    event_row_count: int,
) -> list[str]:
    limitations = [
        (
            "The oracle detector reads exact simulator entity pose. It is an upper-bound "
            "mechanism test, not deployable RGB/RGB-D perception."
        ),
        (
            "The experiment covers one frozen policy, at most two pre-audited "
            "calibration candidates, one moved task-critical entity class, and one "
            "scripted planar perturbation family; sealed evaluation remains limited "
            "to one selected task."
        ),
        (
            "world_epoch is evaluator-only ground truth for labeling stale actions; the "
            "runtime gate is permitted to read pose displacement only."
        ),
        (
            "ActionStream stale-prefix alignment handles temporal delivery age, whereas "
            "the oracle gate handles semantic invalidation after the physical scene moves."
        ),
        (
            "Paired bootstrap intervals treat episode pairs, not individual actions, as "
            "independent statistical units."
        ),
        (
            "The dynamic receptacle can move again during task interaction; later oracle "
            "detections therefore represent additional measured pose changes and can "
            "introduce gate holds beyond the single scripted shift."
        ),
        "No FoundationPose, PoseLoop, ROS 2, physical robot, new model, or training is included.",
    ]
    for value in task_audit.get("known_limitations", []):
        if isinstance(value, str) and value not in limitations:
            limitations.append(value)
    if sealed["status"] != "complete":
        limitations.append(
            f"Sealed evidence is unavailable or incomplete: {sealed.get('reason')}."
        )
    if event_row_count == 0:
        limitations.append(
            "No standalone event rows were supplied; timeline markers are reconstructed "
            "only from episode and action provenance where available."
        )
    return limitations


def render_m5_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a paper-style, evidence-bounded M5-G0 report."""

    acceptance = report["acceptance"]
    calibration = report["calibration"]
    sealed = report["sealed_evaluation"]
    statistics = report["statistics"]
    selected_task = report["selected_task"]
    shift = statistics.get("shift_comparison")
    no_shift = statistics.get("no_shift_comparison")
    classification = acceptance["classification"]
    by_condition = {
        summary["condition"]: summary
        for summary in report.get("condition_summaries", [])
        if isinstance(summary, Mapping) and summary.get("condition") in CONDITIONS
    }
    calibration_table = [
        "| Task | Shift | Pairs | Aligned success | Oracle-gated success | "
        "Gate-only | Physically valid | Qualifies |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for decision in calibration.get("decisions", []):
        if not isinstance(decision, Mapping):
            continue
        for evaluation in decision.get("evaluations", []):
            if not isinstance(evaluation, Mapping):
                continue
            pair_count = int(evaluation.get("pair_count", 0))
            calibration_table.append(
                f"| {evaluation.get('task_id', decision.get('task_id', 'n/a'))} "
                f"| {evaluation.get('displacement_magnitude_mm', 'n/a')} mm "
                f"| {pair_count} "
                f"| {evaluation.get('aligned_success_count', 'n/a')}/{pair_count} "
                f"| {evaluation.get('gate_success_count', 'n/a')}/{pair_count} "
                f"| {evaluation.get('gate_only_success_count', 'n/a')}/{pair_count} "
                f"| {evaluation.get('all_perturbations_physically_valid', False)} "
                f"| {evaluation.get('qualifies', False)} |"
            )
    if len(calibration_table) == 2:
        calibration_table = ["No calibration evaluation rows were supplied."]
    validation = report.get("formal_evidence_validation")
    source_binding_notes: list[str] = []
    if isinstance(validation, Mapping):
        source_binding = validation.get("source_binding")
        if (
            isinstance(source_binding, Mapping)
            and source_binding.get("mode") != "exact"
        ):
            drift_paths = [
                str(entry.get("path"))
                for entry in source_binding.get("implementation_source_drift", [])
                if isinstance(entry, Mapping)
            ]
            source_binding_notes = [
                "- Post-calibration source-binding mode: "
                f"`{source_binding.get('mode')}`.",
                "- Disclosed downstream-only changed files: "
                + ", ".join(f"`{path}`" for path in drift_paths)
                + ".",
                "- Benchmark runtime and calibration/selection files remained "
                f"unchanged: **{source_binding.get('benchmark_runtime_and_selection_files_unchanged')}**.",
            ]
    lines = [
        "# ActionStream M5-G0: Oracle Scene-Shift Gate",
        "",
        "## Abstract",
        "",
        (
            "M5-G0 asks whether invalidating actions after a task-critical simulator "
            "entity moves, then requesting a fresh policy chunk, improves closed-loop "
            "behavior beyond ActionStream's existing stale-prefix alignment."
        ),
        (
            f"The predeclared classification is **{classification['label']}**. "
            "This is a simulator oracle upper-bound result."
        ),
        "",
        "## 1. Experimental boundary",
        "",
        f"- Task: `{selected_task.get('task_id', 'n/a')}` — "
        f"{selected_task.get('instruction', 'not recorded')}.",
        f"- Moved entity: `{selected_task.get('moved_entity', 'n/a')}` "
        f"(`{selected_task.get('simulator_body', selected_task.get('moved_body', 'n/a'))}`).",
        (
            "- Temporal staleness is handled by ActionStream stale-prefix alignment; "
            "semantic scene invalidation is handled by the oracle pose gate."
        ),
        (
            "- The detector sees ground-truth pose displacement and uses no perturbation "
            "flag or world epoch. `world_epoch` is evaluator-only and labels an executed "
            "action stale exactly when its observation epoch is older than the live epoch."
        ),
        "",
        "## 2. Calibration and frozen protocol",
        "",
        f"- Calibration status: **{calibration['status']}**.",
        f"- Selected displacement: "
        f"{calibration.get('selected_displacement_magnitude_mm') or 'not selected'} mm.",
        f"- Detector translation threshold: "
        f"{_fmt(report['protocol'].get('detector_translation_threshold_m'))} m.",
        f"- Injected delivery delay: "
        f"{report['protocol'].get('injected_delay_ms', 'n/a')} ms.",
        (
            f"- Hard-stop reason: {calibration['hard_stop_reason']}"
            if calibration.get("hard_stop_reason")
            else "- The first qualifying magnitude was frozen before sealed evaluation."
        ),
        "",
        *calibration_table,
        "",
        "## 3. Sealed paired evaluation",
        "",
        f"- Status: **{sealed['status']}**.",
        f"- Valid pairs: {sealed['valid_pair_count']}/{sealed['required_pair_count']}.",
    ]
    if isinstance(shift, Mapping):
        success = shift["success"]
        stale = shift["stale_action_duration_seconds"]
        lines.extend(
            [
                f"- Aligned success: {success['aligned']['success_count']}/"
                f"{success['aligned']['episode_count']} "
                f"({_pct(success['aligned']['success_rate'])}).",
                f"- Oracle-gated success: {success['gate']['success_count']}/"
                f"{success['gate']['episode_count']} "
                f"({_pct(success['gate']['success_rate'])}).",
                "- Paired success difference (gate minus aligned): "
                f"{_pct(success['paired_success_difference_gate_minus_aligned'])}; "
                f"exact two-sided McNemar p = "
                f"{_fmt(success['mcnemar_exact_two_sided_p'], 6)}.",
                "- Paired success-difference bootstrap 95% CI: "
                f"{_pct_ci(success.get('paired_success_difference_95pct_bootstrap_ci'))}.",
                "- Pair outcomes: "
                f"{success['gate_only_success_count']} gate-only, "
                f"{success['aligned_only_success_count']} aligned-only, "
                f"{success['both_success_count']} both-success, "
                f"{success['both_failure_count']} both-failure.",
                "- Stale-action duration mean/median: aligned "
                f"{_fmt(stale.get('reference', {}).get('mean'))}/"
                f"{_fmt(stale.get('reference', {}).get('median'))} s; oracle-gated "
                f"{_fmt(stale.get('estimate', {}).get('mean'))}/"
                f"{_fmt(stale.get('estimate', {}).get('median'))} s.",
                "- Paired stale-duration difference (gate minus aligned): mean "
                f"{_fmt(stale.get('paired_mean_difference'))} s, median "
                f"{_fmt(stale.get('paired_median_difference'))} s, bootstrap 95% CI "
                f"{_fmt_ci(stale.get('paired_mean_difference_95pct_bootstrap_ci'))} s.",
                "- Mean stale-action-duration reduction: "
                f"{_pct(stale.get('mean_reduction_fraction'))}; paired mean reduction "
                f"{_fmt(stale.get('paired_mean_reduction'))} s, 95% bootstrap CI "
                f"{_fmt_ci(stale.get('paired_mean_reduction_95pct_bootstrap_ci'))} s.",
            ]
        )
        fresh = shift["time_from_shift_to_first_fresh_action_seconds"]
        lines.append(
            "- Shift-to-first-fresh-action mean/median: aligned "
            f"{_fmt(fresh.get('reference', {}).get('mean'))}/"
            f"{_fmt(fresh.get('reference', {}).get('median'))} s; oracle-gated "
            f"{_fmt(fresh.get('estimate', {}).get('mean'))}/"
            f"{_fmt(fresh.get('estimate', {}).get('median'))} s; paired mean "
            f"gate-minus-aligned difference {_fmt(fresh.get('paired_mean_difference'))} s "
            f"(bootstrap 95% CI "
            f"{_fmt_ci(fresh.get('paired_mean_difference_95pct_bootstrap_ci'))} s, "
            f"{fresh.get('pairs_without_both_values', 0)} pairs missing either value)."
        )
        gated_summary = by_condition.get(GATED_SHIFT, {})
        gated_clear = gated_summary.get("time_from_shift_to_queue_clear_seconds", {})
        lines.append(
            "- Shift-to-queue-clear latency for oracle-gated invalidations: "
            f"n={gated_clear.get('count', 0)}, mean/median "
            f"{_fmt(gated_clear.get('mean'))}/{_fmt(gated_clear.get('median'))} s."
        )
        for field, label, unit in (
            ("environment_steps", "Environment steps", "steps"),
            (
                "simulated_completion_time_seconds",
                "Simulated completion time",
                "s",
            ),
            ("wall_clock_episode_seconds", "Measured wall-clock episode time", "s"),
        ):
            effect = shift[field]
            lines.append(
                f"- {label} mean/median: aligned "
                f"{_fmt(effect.get('reference', {}).get('mean'))}/"
                f"{_fmt(effect.get('reference', {}).get('median'))} {unit}; "
                "oracle-gated "
                f"{_fmt(effect.get('estimate', {}).get('mean'))}/"
                f"{_fmt(effect.get('estimate', {}).get('median'))} {unit}; paired "
                f"mean gate-minus-aligned difference "
                f"{_fmt(effect.get('paired_mean_difference'))} {unit} "
                f"(bootstrap 95% CI "
                f"{_fmt_ci(effect.get('paired_mean_difference_95pct_bootstrap_ci'))} "
                f"{unit})."
            )
    else:
        lines.append(
            "- No sealed outcome or stale-action statistic is reported because the "
            "calibration gate did not authorize sealed rows."
        )
    invalid_episodes = statistics.get("invalid_episodes", [])
    excluded_pairs = statistics.get("excluded_pairs", [])
    lines.append(
        "- Invalid/excluded evidence: "
        f"{len(invalid_episodes)} invalid episode(s), "
        f"{len(excluded_pairs)} excluded pair(s)."
    )
    for episode in invalid_episodes:
        lines.append(
            "- Invalid episode: "
            f"condition `{episode.get('condition')}`, seed {episode.get('seed')}, "
            f"initial state {episode.get('initial_state_index')}: "
            f"{episode.get('reason')}."
        )
    for pair in excluded_pairs:
        reasons = "; ".join(str(reason) for reason in pair.get("reasons", []))
        lines.append(
            "- Excluded pair: "
            f"{pair.get('comparison')} comparison, seed {pair.get('seed')}, "
            f"initial state {pair.get('initial_state_index')}: {reasons}."
        )
    lines.extend(
        [
            "",
            "![Representative paired timeline](plots/representative_paired_timeline.png)",
            "",
            "## 4. Mechanism and task outcome",
            "",
            (
                "The mechanism metric is stale action duration, computed from action-level "
                "world-epoch provenance rather than elapsed time. The task metric is paired "
                "episode success under an identical physical shift."
            ),
            "",
            "![Mechanism and outcome](plots/mechanism_outcome.png)",
            "",
            "## 5. No-shift safety control",
            "",
        ]
    )
    if isinstance(no_shift, Mapping):
        success = no_shift["success"]
        lines.extend(
            [
                f"- Valid pairs: {no_shift['valid_pair_count']}/10.",
                f"- Aligned success: {_pct(success['aligned']['success_rate'])}; "
                f"gated success: {_pct(success['gate']['success_rate'])}.",
                f"- Safety requirements pass: **{no_shift['safety_requirements_pass']}**.",
                f"- Checks: `{json.dumps(no_shift['safety_checks'], sort_keys=True)}`.",
            ]
        )
        no_shift_decision = report.get("no_shift_decision")
        if isinstance(no_shift_decision, Mapping):
            trace = no_shift_decision.get("action_trace_comparison", {})
            lines.append(
                "- Deterministic command traces bit-exact across all pairs: "
                f"**{trace.get('all_action_traces_bit_exact')}**; maximum absolute "
                f"action difference = {_fmt(trace.get('maximum_absolute_action_difference'))}."
            )
    else:
        lines.append(
            "- Not run because calibration did not authorize a frozen task/magnitude; "
            "no no-shift outcome is inferred."
            if calibration.get("hard_stop_reason")
            else "- No complete no-shift comparison is present."
        )
    lines.extend(
        [
            "",
            "## 6. Acceptance decision",
            "",
            f"**{classification['label']}**",
            "",
            *[f"- {reason}" for reason in classification.get("reasons", [])],
            "",
            "The classification uses the predeclared M5-G0 thresholds and is not revised "
            "from visual inspection of the figures.",
            "",
            "## 7. Source provenance",
            "",
            "| role | path | SHA-256 | rows |",
            "|---|---|---|---:|",
            *[
                f"| {source['role']} | `{source['path']}` | "
                f"`{source['sha256']}` | {source.get('row_count', 'n/a')} |"
                for source in report["sources"]
            ],
            "",
            "All hashes above are computed over the exact input bytes consumed by this report.",
            "",
            *source_binding_notes,
            "",
            "## 8. Limitations",
            "",
            *[f"- {limitation}" for limitation in report["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)


def build_m5_report(
    *,
    config_path: Path | str,
    task_audit_path: Path | str,
    calibration_decision_paths: Sequence[Path | str],
    seed_manifest_paths: Sequence[Path | str],
    episode_paths: Sequence[Path | str] = (),
    action_paths: Sequence[Path | str] = (),
    event_paths: Sequence[Path | str] = (),
    statistics_path: Path | str | None = None,
    protocol_decision_path: Path | str | None = None,
    no_shift_decision_path: Path | str | None = None,
    validation_path: Path | str | None = None,
    m4_baseline_intact: bool | None = None,
    source_validation_passed: bool | None = None,
) -> dict[str, Any]:
    """Read, validate, and summarize the complete M5-G0 evidence package."""

    if not calibration_decision_paths:
        raise ValueError("At least one calibration decision is required")
    if not seed_manifest_paths:
        raise ValueError("Seed manifests are required")
    config = _read_json(config_path, role="config")
    task_audit = _read_json(task_audit_path, role="task audit")
    _ensure_milestone(config, role="config")
    _ensure_milestone(task_audit, role="task audit")
    protocol_decision = (
        None
        if protocol_decision_path is None
        else _read_json(protocol_decision_path, role="frozen protocol decision")
    )
    if protocol_decision is not None:
        _validate_canonical_hash(
            protocol_decision,
            hash_field="protocol_decision_sha256",
            role="frozen protocol decision",
        )
    no_shift_decision = (
        None
        if no_shift_decision_path is None
        else _read_json(no_shift_decision_path, role="no-shift decision")
    )
    if no_shift_decision is not None:
        _validate_canonical_hash(
            no_shift_decision,
            hash_field="no_shift_decision_sha256",
            role="no-shift decision",
        )
        if protocol_decision is None:
            raise ValueError("A no-shift decision requires its frozen protocol")
        if no_shift_decision.get("protocol_decision_sha256") != protocol_decision.get(
            "protocol_decision_sha256"
        ):
            raise ValueError("No-shift decision is bound to a different protocol")
    validation = (
        None
        if validation_path is None
        else _read_json(validation_path, role="formal evidence validation")
    )
    if validation is not None:
        _validate_canonical_hash(
            validation,
            hash_field="validation_sha256",
            role="formal evidence validation",
        )
        if validation.get("status") != "pass":
            raise ValueError("Formal evidence validation did not pass")
        if protocol_decision is None:
            raise ValueError("Formal validation requires a frozen protocol")
        if validation.get("protocol_decision_sha256") != protocol_decision.get(
            "protocol_decision_sha256"
        ):
            raise ValueError("Formal validation is bound to a different protocol")

    decisions = [
        _read_json(path, role=f"calibration decision {index}")
        for index, path in enumerate(calibration_decision_paths)
    ]
    for index, decision in enumerate(decisions):
        _validate_calibration_decision(decision, role=f"calibration decision {index}")
    if protocol_decision is not None:
        _validate_protocol_calibration_decisions(
            protocol_decision,
            calibration_decision_paths,
            decisions,
        )
    manifests = [
        _read_json(path, role=f"seed manifest {index}")
        for index, path in enumerate(seed_manifest_paths)
    ]
    for manifest in manifests:
        validate_manifest(manifest)
    validate_disjoint(manifests)
    manifest_names = [str(manifest["manifest_name"]) for manifest in manifests]
    required_manifests = {"calibration", "no_shift_regression", "sealed_evaluation"}
    if set(manifest_names) != required_manifests:
        raise ValueError(
            "Expected calibration, no_shift_regression, and sealed_evaluation "
            f"manifests, got {sorted(manifest_names)}"
        )
    expected_manifest_counts = {
        "calibration": 5,
        "no_shift_regression": 10,
        "sealed_evaluation": 30,
    }
    for manifest in manifests:
        name = str(manifest["manifest_name"])
        actual_count = len(manifest["pairs"])
        if actual_count != expected_manifest_counts[name]:
            raise ValueError(
                f"{name} manifest has {actual_count} pairs; "
                f"expected {expected_manifest_counts[name]}"
            )

    episode_rows = _read_jsonl(episode_paths, role="episode evidence")
    action_rows = _read_jsonl(action_paths, role="action evidence")
    event_rows = _read_jsonl(event_paths, role="event evidence")
    if validation is not None:
        validated_sources = {
            str(Path(str(entry["path"])).resolve()): str(entry["sha256"])
            for entry in validation.get("sources", [])
            if isinstance(entry, Mapping) and "path" in entry and "sha256" in entry
        }
        supplied_sources = [
            Path(config_path),
            Path(task_audit_path),
            *[Path(path) for path in calibration_decision_paths],
            *[Path(path) for path in seed_manifest_paths],
            *[Path(path) for path in episode_paths],
            *[Path(path) for path in action_paths],
            *[Path(path) for path in event_paths],
        ]
        if protocol_decision_path is not None:
            supplied_sources.append(Path(protocol_decision_path))
        if no_shift_decision_path is not None:
            supplied_sources.append(Path(no_shift_decision_path))
        for path in supplied_sources:
            resolved = str(path.resolve())
            if validated_sources.get(resolved) != _sha256(path):
                raise ValueError(
                    "Formal validation does not bind the exact supplied source: "
                    f"{path}"
                )
    action_validation = _validate_action_rows(action_rows)
    calibration = _calibration_summary(decisions)
    seed_sets = _manifest_seed_sets(manifests)
    analysis_rows = _analysis_episode_rows(episode_rows, seed_sets)
    sealed_shift_rows = [
        row
        for row in analysis_rows
        if (
            int(row.get("seed", -1)),
            int(row.get("initial_state_index", -1)),
        )
        in seed_sets["sealed_evaluation"]
        and row.get("condition") in {ALIGNED_SHIFT, GATED_SHIFT}
    ]
    if calibration["hard_stop_reason"] is not None and sealed_shift_rows:
        raise ValueError(
            "Sealed shift rows are present despite the calibration hard stop"
        )

    if validation is None:
        baseline_flag = (
            _extract_m4_baseline_flag(task_audit)
            if m4_baseline_intact is None
            else m4_baseline_intact
        )
        source_flag = (
            True if source_validation_passed is None else source_validation_passed
        )
    else:
        validation_checks = validation.get("checks", {})
        baseline_flag = validation_checks.get("m4_baseline_intact")
        source_flag = all(
            validation_checks.get(field) is True
            for field in (
                "repository_tests_and_compile_passed",
                "frozen_source_binding_passed",
                "episode_schema_and_pair_contract_passed",
                "action_request_event_provenance_passed",
            )
        )
        if m4_baseline_intact is not None and m4_baseline_intact != baseline_flag:
            raise ValueError("Caller M4 flag disagrees with formal validation")
        if (
            source_validation_passed is not None
            and source_validation_passed != source_flag
        ):
            raise ValueError("Caller source flag disagrees with formal validation")
    if type(baseline_flag) is not bool or type(source_flag) is not bool:
        raise ValueError("Validation flags must be booleans")
    recomputed_statistics = build_m5_analysis(
        analysis_rows,
        m4_baseline_intact=baseline_flag,
        source_validation_passed=source_flag,
        calibration_no_go_reason=calibration["hard_stop_reason"],
    )
    if statistics_path is None:
        statistics = recomputed_statistics
        statistics_origin = "computed_by_actionstream.m5_analysis"
    else:
        statistics = _read_json(statistics_path, role="reviewed statistics")
        _validate_statistics(statistics)
        if canonical_sha256(statistics) != canonical_sha256(recomputed_statistics):
            raise ValueError(
                "Reviewed statistics do not canonically match statistics recomputed "
                "from the supplied sealed/no-shift episode rows and validation gates"
            )
        statistics_origin = "reviewed_statistics_input_copied"
    _validate_statistics(statistics)

    condition_summaries = _condition_summaries(episode_rows)
    sealed = _sealed_state(statistics, calibration)
    selected_task = _selected_task(
        task_audit,
        config,
        selected_task_id=calibration.get("selected_task_id"),
    )
    classification = dict(statistics["classification"])
    if (
        calibration["hard_stop_reason"] is not None
        and classification["label"] != "NO-GO"
    ):
        raise ValueError(
            "A calibration hard stop cannot be paired with a positive classification"
        )

    sources: list[dict[str, Any]] = [
        _source_entry(config_path, role="config"),
        _source_entry(task_audit_path, role="task_audit"),
    ]
    sources.extend(
        _source_entry(path, role="calibration_decision")
        for path in calibration_decision_paths
    )
    sources.extend(
        _source_entry(path, role="seed_manifest") for path in seed_manifest_paths
    )
    for path in episode_paths:
        path_rows = _read_jsonl([path], role="episode evidence")
        sources.append(
            _source_entry(path, role="episode_jsonl", row_count=len(path_rows))
        )
    for path in action_paths:
        path_rows = _read_jsonl([path], role="action evidence")
        sources.append(
            _source_entry(path, role="action_jsonl", row_count=len(path_rows))
        )
    for path in event_paths:
        path_rows = _read_jsonl([path], role="event evidence")
        sources.append(
            _source_entry(path, role="event_jsonl", row_count=len(path_rows))
        )
    if statistics_path is not None:
        sources.append(_source_entry(statistics_path, role="reviewed_statistics"))
    if protocol_decision_path is not None:
        sources.append(
            _source_entry(protocol_decision_path, role="frozen_protocol_decision")
        )
    if no_shift_decision_path is not None:
        sources.append(_source_entry(no_shift_decision_path, role="no_shift_decision"))
    if validation_path is not None:
        sources.append(
            _source_entry(validation_path, role="formal_evidence_validation")
        )

    pair = _representative_pair(analysis_rows)
    representative = (
        None
        if pair is None
        else {
            "seed": pair["seed"],
            "initial_state_index": pair["initial_state_index"],
            "selection_rule": "prefer gate-only success, then discordance, then lowest seed",
            "aligned_success": bool(pair[ALIGNED_SHIFT].get("success")),
            "gate_success": bool(pair[GATED_SHIFT].get("success")),
        }
    )
    protocol = {
        "injected_delay_ms": config.get("runtime", {}).get("injected_delay_ms"),
        "control_frequency_hz": config.get("runtime", {}).get("control_frequency_hz"),
        "detector_translation_threshold_m": config.get("detector", {}).get(
            "translation_threshold_m"
        ),
        "perturbation_axis_xyz": config.get("perturbation", {}).get("axis_xyz"),
        "selected_displacement_magnitude_mm": calibration.get(
            "selected_displacement_magnitude_mm"
        ),
    }
    limitations = _default_limitations(
        task_audit=task_audit,
        sealed=sealed,
        event_row_count=len(event_rows),
    )
    acceptance_core = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "milestone": "M5-G0",
        "classification": classification,
        "calibration_status": calibration["status"],
        "sealed_evaluation": sealed,
        "no_shift_safety_requirements_pass": (
            statistics.get("no_shift_comparison", {}).get("safety_requirements_pass")
            if isinstance(statistics.get("no_shift_comparison"), Mapping)
            else False
        ),
        "no_shift_action_trace_equivalence_pass": (
            no_shift_decision.get("action_trace_comparison", {}).get(
                "all_action_traces_bit_exact"
            )
            if isinstance(no_shift_decision, Mapping)
            else None
        ),
        "m4_baseline_intact": baseline_flag,
        "source_validation_passed": source_flag,
        "statistics_origin": statistics_origin,
    }
    acceptance = {
        **acceptance_core,
        "acceptance_sha256": canonical_sha256(acceptance_core),
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "milestone": "M5-G0",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_task": selected_task,
        "protocol": protocol,
        "frozen_protocol_decision": protocol_decision,
        "no_shift_decision": no_shift_decision,
        "formal_evidence_validation": validation,
        "calibration": calibration,
        "sealed_evaluation": sealed,
        "condition_summaries": condition_summaries,
        "action_provenance_validation": action_validation,
        "event_row_count": len(event_rows),
        "analysis_episode_row_count": len(analysis_rows),
        "representative_pair": representative,
        "statistics": statistics,
        "acceptance": acceptance,
        "sources": sources,
        "limitations": limitations,
        "_plot_evidence": {
            "pair": pair,
            "action_rows": action_rows,
            "event_rows": event_rows,
        },
    }


def write_m5_report(
    *,
    config_path: Path | str,
    task_audit_path: Path | str,
    calibration_decision_paths: Sequence[Path | str],
    seed_manifest_paths: Sequence[Path | str],
    episode_paths: Sequence[Path | str] = (),
    action_paths: Sequence[Path | str] = (),
    event_paths: Sequence[Path | str] = (),
    statistics_path: Path | str | None = None,
    protocol_decision_path: Path | str | None = None,
    no_shift_decision_path: Path | str | None = None,
    validation_path: Path | str | None = None,
    output_root: Path | str,
    m4_baseline_intact: bool | None = None,
    source_validation_passed: bool | None = None,
) -> dict[str, Any]:
    """Write JSON, Markdown, and PNG/PDF artifacts to ``output_root``."""

    report = build_m5_report(
        config_path=config_path,
        task_audit_path=task_audit_path,
        calibration_decision_paths=calibration_decision_paths,
        seed_manifest_paths=seed_manifest_paths,
        episode_paths=episode_paths,
        action_paths=action_paths,
        event_paths=event_paths,
        statistics_path=statistics_path,
        protocol_decision_path=protocol_decision_path,
        no_shift_decision_path=no_shift_decision_path,
        validation_path=validation_path,
        m4_baseline_intact=m4_baseline_intact,
        source_validation_passed=source_validation_passed,
    )
    output = Path(output_root)
    plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plot_evidence = report.pop("_plot_evidence")
    unavailable_reason = report["sealed_evaluation"].get("reason")
    _plot_representative_timeline(
        pair=plot_evidence["pair"],
        action_rows=plot_evidence["action_rows"],
        event_rows=plot_evidence["event_rows"],
        output_dir=plots,
        unavailable_reason=unavailable_reason,
    )
    _plot_mechanism_outcome(
        statistics=report["statistics"],
        condition_summaries=report["condition_summaries"],
        calibration=report["calibration"],
        output_dir=plots,
    )
    report["artifacts"] = {
        "aggregate_summary": "aggregate_summary.json",
        "statistical_tests": "statistical_tests.json",
        "acceptance": "acceptance.json",
        "paper_report": "m5_g0_report.md",
        "plots": [f"plots/{name}" for name in FIGURE_FILENAMES],
    }
    (output / "statistical_tests.json").write_text(
        json.dumps(report["statistics"], indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "acceptance.json").write_text(
        json.dumps(report["acceptance"], indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "aggregate_summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "m5_g0_report.md").write_text(
        render_m5_report_markdown(report),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--task-audit",
        "--task-entity-audit",
        dest="task_audit",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration-decision",
        "--calibration-decisions",
        dest="calibration_decisions",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--seed-manifest",
        "--seed-manifests",
        dest="seed_manifests",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--episodes", type=Path, nargs="*", default=[])
    parser.add_argument("--actions", type=Path, nargs="*", default=[])
    parser.add_argument("--events", type=Path, nargs="*", default=[])
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--protocol-decision", type=Path)
    parser.add_argument("--no-shift-decision", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument(
        "--output-root",
        "--output-dir",
        dest="output_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--m4-baseline-intact",
        action="store_true",
        default=None,
        help="Confirm the frozen M4 smoke/tests remain intact.",
    )
    parser.add_argument(
        "--source-validation-failed",
        action="store_true",
        help="Force the source-validation acceptance gate to false.",
    )
    args = parser.parse_args()
    report = write_m5_report(
        config_path=args.config,
        task_audit_path=args.task_audit,
        calibration_decision_paths=args.calibration_decisions,
        seed_manifest_paths=args.seed_manifests,
        episode_paths=args.episodes,
        action_paths=args.actions,
        event_paths=args.events,
        statistics_path=args.statistics,
        protocol_decision_path=args.protocol_decision,
        no_shift_decision_path=args.no_shift_decision,
        validation_path=args.validation,
        output_root=args.output_root,
        m4_baseline_intact=args.m4_baseline_intact,
        source_validation_passed=(False if args.source_validation_failed else None),
    )
    print(f"{args.output_root} ({report['acceptance']['classification']['label']})")


if __name__ == "__main__":
    main()
