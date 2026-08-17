"""Validate M5-G0 evidence and acceptance gates before statistical reporting."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from actionstream.m5_analysis import (
    ALIGNED_NO_SHIFT,
    ALIGNED_SHIFT,
    GATED_NO_SHIFT,
    GATED_SHIFT,
    pair_episode_rows,
    validate_episode_rows,
)
from actionstream.m5_protocol import (
    canonical_sha256,
    file_sha256,
    implementation_source_hash,
    validate_disjoint,
    validate_manifest,
)
from actionstream.m5_report import _validate_action_rows
from actionstream.m5_runtime import summarize_action_records


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must be a JSON object: {path}")
    return payload


def _read_jsonl(paths: Iterable[Path], *, role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"{role} row must be a JSON object: {path}:{line_number}"
                    )
                rows.append(payload)
    return rows


def _validate_canonical_payload(
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


def _validate_m4_baseline(
    repository_root: Path,
    task_audit: dict[str, Any],
) -> dict[str, Any]:
    if task_audit.get("m4_baseline_intact") is not True:
        raise ValueError("Task audit does not affirm an intact M4 baseline")
    smoke = task_audit.get("m4_baseline_smoke")
    if not isinstance(smoke, dict) or smoke.get("passed") is not True:
        raise ValueError("Task audit lacks a passing M4 baseline smoke")
    smoke_path = Path(str(smoke["path"]))
    if not smoke_path.is_absolute():
        smoke_path = repository_root / smoke_path
    actual_sha256 = file_sha256(smoke_path)
    if actual_sha256 != smoke.get("sha256"):
        raise ValueError("M4 baseline smoke evidence hash mismatch")
    rows = _read_jsonl([smoke_path], role="M4 baseline smoke")
    if len(rows) != 1:
        raise ValueError("M4 baseline smoke must contain exactly one episode")
    row = rows[0]
    expected = {
        "runtime_mode": smoke["condition"],
        "task_id": int(smoke["task_id"]),
        "seed": int(smoke["seed"]),
        "initial_state_index": int(smoke["initial_state_index"]),
        "injected_delay_ms": int(smoke["injected_delay_ms"]),
        "success": bool(smoke["success"]),
        "environment_steps": int(smoke["environment_steps"]),
    }
    mismatches = {
        field: {"audit": expected_value, "evidence": row.get(field)}
        for field, expected_value in expected.items()
        if row.get(field) != expected_value
    }
    if mismatches:
        raise ValueError(f"M4 baseline smoke metadata mismatch: {mismatches}")
    return {
        "passed": True,
        "path": str(smoke_path),
        "sha256": actual_sha256,
        "episode": expected,
    }


def _run_repository_checks(repository_root: Path) -> dict[str, Any]:
    commands = (
        [sys.executable, "-m", "pytest", "-q"],
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            str(repository_root / "src" / "actionstream"),
        ],
    )
    evidence: list[dict[str, Any]] = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        entry = {
            "command": command,
            "returncode": completed.returncode,
        }
        evidence.append(entry)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Repository validation failed: {command}\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
    test_sources = [
        {
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for path in sorted((repository_root / "tests").glob("test_*.py"))
    ]
    return {
        "passed": True,
        "commands": evidence,
        "test_sources": test_sources,
    }


def _manifest_index(
    manifest_paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, set[tuple[int, int]]]]:
    manifests: dict[str, dict[str, Any]] = {}
    payloads: list[dict[str, Any]] = []
    pair_sets: dict[str, set[tuple[int, int]]] = {}
    for path in manifest_paths:
        payload = _read_json(path, role="seed manifest")
        validate_manifest(payload)
        name = str(payload["manifest_name"])
        if name in manifests:
            raise ValueError(f"Duplicate seed manifest: {name}")
        manifests[name] = payload
        payloads.append(payload)
        pair_sets[name] = {
            (int(pair["seed"]), int(pair["initial_state_index"]))
            for pair in payload["pairs"]
        }
    validate_disjoint(payloads)
    expected = {
        "calibration": 5,
        "no_shift_regression": 10,
        "sealed_evaluation": 30,
    }
    if set(manifests) != set(expected):
        raise ValueError("Validation requires all three M5-G0 seed manifests")
    for name, count in expected.items():
        if len(pair_sets[name]) != count:
            raise ValueError(f"{name} manifest must contain {count} unique pairs")
    return manifests, pair_sets


def _validate_episode_bindings(
    *,
    rows: list[dict[str, Any]],
    current_source_sha256: str,
    config_sha256: str,
    task_audit_sha256: str,
    manifests: dict[str, dict[str, Any]],
    manifest_pairs: dict[str, set[tuple[int, int]]],
    protocol: dict[str, Any],
    no_shift_decision: dict[str, Any] | None,
) -> dict[str, Any]:
    validate_episode_rows(rows)
    protocol_sha256 = str(protocol["protocol_decision_sha256"])
    no_shift_sha256 = (
        None
        if no_shift_decision is None
        else str(no_shift_decision["no_shift_decision_sha256"])
    )
    expected_by_phase = {
        "no_shift": {
            "manifest": "no_shift_regression",
            "conditions": {ALIGNED_NO_SHIFT, GATED_NO_SHIFT},
            "task_id": int(protocol["no_shift_task_id"]),
            "magnitude": 0,
        },
        "sealed": {
            "manifest": "sealed_evaluation",
            "conditions": {ALIGNED_SHIFT, GATED_SHIFT},
            "task_id": protocol.get("selected_task_id"),
            "magnitude": protocol.get("selected_displacement_magnitude_mm"),
        },
    }
    phase_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        phase = str(row.get("phase"))
        if phase not in expected_by_phase:
            raise ValueError(f"Episode row {index} has unexpected phase {phase!r}")
        expected = expected_by_phase[phase]
        manifest_name = str(expected["manifest"])
        pair = (int(row["seed"]), int(row["initial_state_index"]))
        if pair not in manifest_pairs[manifest_name]:
            raise ValueError(
                f"Episode row {index} seed/state is not in {manifest_name}: {pair}"
            )
        checks = {
            "condition": row["condition"] in expected["conditions"],
            "task_id": int(row["task_id"]) == int(expected["task_id"]),
            "displacement_magnitude_mm": int(row["displacement_magnitude_mm"])
            == int(expected["magnitude"]),
            "implementation_source_sha256": row["implementation_source_sha256"]
            == current_source_sha256,
            "experiment_config_sha256": row["experiment_config_sha256"]
            == config_sha256,
            "task_audit_sha256": row["task_audit_sha256"] == task_audit_sha256,
            "seed_manifest_name": row["seed_manifest_name"] == manifest_name,
            "seed_manifest_sha256": row["seed_manifest_sha256"]
            == manifests[manifest_name]["manifest_sha256"],
            "protocol_decision_sha256": row["protocol_decision_sha256"]
            == protocol_sha256,
            "no_shift_decision_sha256": (
                row["no_shift_decision_sha256"] is None
                if phase == "no_shift"
                else row["no_shift_decision_sha256"] == no_shift_sha256
            ),
        }
        failed = sorted(key for key, passed in checks.items() if not passed)
        if failed:
            raise ValueError(
                f"Episode row {index} failed frozen binding checks: {failed}"
            )
        phase_counts[phase] += 1

    status = str(protocol["status"])
    if status == "calibration_no_go":
        expected_phase_counts: dict[str, int] = {}
    else:
        expected_phase_counts = {"no_shift": 20}
        if no_shift_decision is not None and no_shift_decision.get("status") == "pass":
            expected_phase_counts["sealed"] = 60
    if dict(phase_counts) != expected_phase_counts:
        raise ValueError(
            f"Evidence phase counts {dict(phase_counts)} do not match "
            f"the frozen protocol {expected_phase_counts}"
        )

    pair_checks: dict[str, Any] = {}
    for phase, reference, estimate, expected_pairs in (
        ("no_shift", ALIGNED_NO_SHIFT, GATED_NO_SHIFT, 10),
        ("sealed", ALIGNED_SHIFT, GATED_SHIFT, 30),
    ):
        phase_rows = [row for row in rows if row.get("phase") == phase]
        if not phase_rows:
            continue
        pairing = pair_episode_rows(
            phase_rows,
            reference_condition=reference,
            estimate_condition=estimate,
            require_valid_perturbation=False,
        )
        if pairing["valid_pair_count"] != expected_pairs or pairing["excluded_pairs"]:
            raise ValueError(
                f"{phase} pair contract failed: "
                f"{pairing['valid_pair_count']}/{expected_pairs} valid; "
                f"excluded={pairing['excluded_pairs']}"
            )
        pair_checks[phase] = {
            "pair_count": pairing["valid_pair_count"],
            "pair_contract_fields_identical": True,
        }
    return {
        "passed": True,
        "episode_count": len(rows),
        "phase_episode_counts": dict(sorted(phase_counts.items())),
        "pair_checks": pair_checks,
    }


def _validate_trace_files(
    *,
    episodes: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    action_validation = _validate_action_rows(actions)
    episodes_by_id = {str(row["episode_id"]): row for row in episodes}
    if len(episodes_by_id) != len(episodes):
        raise ValueError("Episode IDs are not unique")
    actions_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    requests_by_episode: Counter[str] = Counter()
    request_ids_by_episode: dict[str, set[str]] = defaultdict(set)
    request_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    events_by_episode: dict[str, Counter[str]] = defaultdict(Counter)
    event_rows_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        episode_id = str(row.get("episode_id"))
        if episode_id not in episodes_by_id:
            raise ValueError(f"Action references unknown episode {episode_id}")
        if row.get("condition") != episodes_by_id[episode_id]["condition"]:
            raise ValueError(f"Action condition mismatch for {episode_id}")
        actions_by_episode[episode_id].append(row)
    for row in requests:
        episode_id = str(row.get("episode_id"))
        if episode_id not in episodes_by_id:
            raise ValueError(f"Request references unknown episode {episode_id}")
        if row.get("condition") != episodes_by_id[episode_id]["condition"]:
            raise ValueError(f"Request condition mismatch for {episode_id}")
        request_id = str(row.get("request_id"))
        key = (episode_id, request_id)
        if not request_id or key in request_rows_by_key:
            raise ValueError(f"Duplicate or missing request ID for {episode_id}")
        request_rows_by_key[key] = row
        request_ids_by_episode[episode_id].add(request_id)
        requests_by_episode[episode_id] += 1
    for row in events:
        episode_id = str(row.get("episode_id"))
        if episode_id not in episodes_by_id:
            raise ValueError(f"Event references unknown episode {episode_id}")
        if row.get("condition") != episodes_by_id[episode_id]["condition"]:
            raise ValueError(f"Event condition mismatch for {episode_id}")
        events_by_episode[episode_id][str(row.get("event_type"))] += 1
        event_rows_by_episode[episode_id].append(row)

    summary_fields = (
        "action_record_count",
        "executed_action_steps",
        "discarded_action_steps",
        "stale_action_steps",
        "stale_action_duration_seconds",
        "first_stale_action_step",
        "last_stale_action_step",
        "stale_queued_actions_discarded",
        "stale_policy_results_discarded",
        "hold_steps_introduced_by_gate",
        "gate_hold_steps",
        "queue_hold_steps",
        "maximum_queue_depth",
        "action_source_observation_age_steps_mean",
        "action_source_observation_age_steps_median",
        "action_source_observation_age_steps_max",
        "discard_reason_counts",
    )
    for episode_id, episode in episodes_by_id.items():
        episode_actions = actions_by_episode.get(episode_id, [])
        request_ids = request_ids_by_episode[episode_id]
        for row in episode_actions:
            if str(row.get("request_id")) not in request_ids:
                raise ValueError(
                    f"Action references an unrecorded request for {episode_id}"
                )
        episode_events = event_rows_by_episode[episode_id]
        for row in episode_events:
            if "request_id" in row and str(row.get("request_id")) not in request_ids:
                raise ValueError(
                    f"Event references an unrecorded request for {episode_id}"
                )
        summary = summarize_action_records(
            episode_actions,
            float(episode["controller_frequency_hz"]),
        )
        mismatches = {
            field: {"episode": episode.get(field), "recomputed": summary[field]}
            for field in summary_fields
            if episode.get(field) != summary[field]
        }
        if mismatches:
            raise ValueError(f"Action summary mismatch for {episode_id}: {mismatches}")
        if int(episode["environment_steps"]) != int(summary["executed_action_steps"]):
            raise ValueError(f"Executed action count mismatch for {episode_id}")
        if requests_by_episode[episode_id] != int(episode["request_count"]):
            raise ValueError(f"Request count mismatch for {episode_id}")
        event_counts = events_by_episode[episode_id]
        if event_counts["episode_start"] != 1 or event_counts["episode_end"] != 1:
            raise ValueError(f"Episode boundary events are incomplete for {episode_id}")
        if event_counts["action_execution"] != int(summary["executed_action_steps"]):
            raise ValueError(f"Action execution events are incomplete for {episode_id}")
        request_event_count = (
            event_counts["policy_request"] + event_counts["fresh_policy_request"]
        )
        if request_event_count != requests_by_episode[episode_id]:
            raise ValueError(f"Policy request events are incomplete for {episode_id}")
        if event_counts["scene_detector_check"] != int(episode["detector_checks"]):
            raise ValueError(f"Detector check events disagree for {episode_id}")
        if event_counts["scene_gate_trigger"] != int(
            episode["scene_gate_trigger_count"]
        ):
            raise ValueError(f"Scene-gate trigger events disagree for {episode_id}")
        if event_counts["queue_clear"] != int(episode["queue_invalidation_count"]):
            raise ValueError(f"Queue-clear events disagree for {episode_id}")
        if event_counts["fresh_policy_request"] != int(episode["fresh_replan_count"]):
            raise ValueError(f"Fresh-request events disagree for {episode_id}")
        expected_false_gates = (
            event_counts["scene_gate_trigger"]
            if episode["condition"] in {ALIGNED_NO_SHIFT, GATED_NO_SHIFT}
            else 0
        )
        if int(episode["false_gate_count"]) != expected_false_gates:
            raise ValueError(f"False-gate counter disagrees for {episode_id}")

        action_events = {
            int(row["step"]): row
            for row in episode_events
            if row.get("event_type") == "action_execution"
        }
        if len(action_events) != int(summary["executed_action_steps"]):
            raise ValueError(f"Action execution steps are duplicated for {episode_id}")
        for row in episode_actions:
            if bool(row.get("discarded", row.get("record_type") == "discarded")):
                continue
            step = int(row["queue_execution_step"])
            event = action_events.get(step)
            if event is None or any(
                (
                    event.get("request_id") != row.get("request_id"),
                    event.get("source_kind") != row.get("source_kind"),
                    event.get("stale") != row.get("stale"),
                )
            ):
                raise ValueError(
                    f"Executed action/event linkage disagrees at "
                    f"{episode_id} step {step}"
                )
        result_event_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in episode_events:
            if row.get("event_type") == "policy_result_arrival":
                result_event_rows[str(row.get("request_id"))].append(row)
        outcome_events = Counter(
            str(row.get("request_id"))
            for row in episode_events
            if row.get("event_type") in {"queue_merge", "policy_result_discarded"}
        )
        for request_id in request_ids:
            request = request_rows_by_key[(episode_id, request_id)]
            result_expected = request.get("policy_result_arrival_timestamp") is not None
            expected_count = int(result_expected)
            arrivals = result_event_rows[request_id]
            expected_outcome_count = (
                int(bool(arrivals[0].get("merge"))) if len(arrivals) == 1 else 0
            )
            if (
                len(arrivals) != expected_count
                or outcome_events[request_id] != expected_outcome_count
            ):
                raise ValueError(
                    f"Policy result/event linkage disagrees for "
                    f"{episode_id} request {request_id}"
                )
    return {
        "passed": True,
        "episode_count": len(episodes),
        "action_validation": action_validation,
        "request_count": len(requests),
        "event_count": len(events),
        "episode_action_summaries_recomputed": True,
        "request_action_event_linkage_validated": True,
    }


def build_validation(
    *,
    repository_root: Path,
    config_path: Path,
    task_audit_path: Path,
    manifest_paths: list[Path],
    protocol_decision_path: Path,
    no_shift_decision_path: Path | None,
    episode_paths: list[Path],
    action_paths: list[Path],
    request_paths: list[Path],
    event_paths: list[Path],
    run_tests: bool,
) -> dict[str, Any]:
    _config = _read_json(config_path, role="config")
    task_audit = _read_json(task_audit_path, role="task audit")
    protocol = _read_json(protocol_decision_path, role="frozen protocol")
    _validate_canonical_payload(
        protocol,
        hash_field="protocol_decision_sha256",
        role="frozen protocol",
    )
    no_shift_decision = (
        None
        if no_shift_decision_path is None
        else _read_json(no_shift_decision_path, role="no-shift decision")
    )
    if no_shift_decision is not None:
        _validate_canonical_payload(
            no_shift_decision,
            hash_field="no_shift_decision_sha256",
            role="no-shift decision",
        )
        if (
            no_shift_decision["protocol_decision_sha256"]
            != protocol["protocol_decision_sha256"]
        ):
            raise ValueError("No-shift decision is bound to another protocol")

    source_sha256, source_files = implementation_source_hash(repository_root)
    config_sha256 = file_sha256(config_path)
    audit_sha256 = file_sha256(task_audit_path)
    recorded_implementation_files = {
        str(entry["path"]): str(entry["sha256"])
        for entry in protocol.get("implementation_source_files", [])
        if isinstance(entry, dict) and "path" in entry and "sha256" in entry
    }
    if len(recorded_implementation_files) != len(
        protocol.get("implementation_source_files", [])
    ):
        raise ValueError("Frozen protocol implementation source list is incomplete")
    implementation_source_drift: list[dict[str, str]] = []
    for relative_path, recorded_sha256 in recorded_implementation_files.items():
        current_sha256 = file_sha256(repository_root / relative_path)
        if current_sha256 != recorded_sha256:
            implementation_source_drift.append(
                {
                    "path": relative_path,
                    "recorded_sha256": recorded_sha256,
                    "current_sha256": current_sha256,
                }
            )
    drift_paths = {entry["path"] for entry in implementation_source_drift}
    allowed_post_calibration_drift = {
        "scripts/run_m5_g0.sh",
        "src/actionstream/m5_report.py",
        "src/actionstream/m5_validation.py",
    }
    downstream_only_patch = drift_paths == allowed_post_calibration_drift
    if drift_paths and not downstream_only_patch:
        raise ValueError(
            "Frozen experiment producer source has drifted outside the "
            "documented post-calibration validation/reporting patch: "
            f"{sorted(drift_paths)}"
        )
    source_binding_mode = (
        "exact"
        if not implementation_source_drift
        else "post_calibration_validation_and_reporting_patch"
    )
    if (
        not implementation_source_drift
        and protocol["implementation_source_sha256"] != source_sha256
    ):
        raise ValueError("Frozen protocol implementation source hash disagrees")
    if protocol["config"]["sha256"] != config_sha256:
        raise ValueError("Frozen protocol config has drifted")
    if protocol["task_audit"]["sha256"] != audit_sha256:
        raise ValueError("Frozen protocol task audit has drifted")
    recorded_protocol_sources = [
        *protocol.get("manifest_sources", []),
        *protocol.get("calibration_sources", []),
        protocol.get("config", {}),
        protocol.get("task_audit", {}),
    ]
    recorded_protocol_paths: list[Path] = []
    for entry in recorded_protocol_sources:
        if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
            raise ValueError("Frozen protocol contains an incomplete source binding")
        path = Path(str(entry["path"]))
        if not path.is_absolute():
            path = repository_root / path
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Frozen protocol source has drifted: {path}")
        recorded_protocol_paths.append(path)

    manifests, manifest_pairs = _manifest_index(manifest_paths)
    episodes = _read_jsonl(episode_paths, role="episode evidence")
    actions = _read_jsonl(action_paths, role="action evidence")
    requests = _read_jsonl(request_paths, role="request evidence")
    events = _read_jsonl(event_paths, role="event evidence")
    baseline = _validate_m4_baseline(repository_root, task_audit)
    repository_checks = (
        _run_repository_checks(repository_root)
        if run_tests
        else {"passed": False, "commands": []}
    )
    if not repository_checks["passed"]:
        raise ValueError("Repository tests must run and pass for formal validation")
    episode_validation = _validate_episode_bindings(
        rows=episodes,
        current_source_sha256=protocol["implementation_source_sha256"],
        config_sha256=config_sha256,
        task_audit_sha256=audit_sha256,
        manifests=manifests,
        manifest_pairs=manifest_pairs,
        protocol=protocol,
        no_shift_decision=no_shift_decision,
    )
    trace_validation = _validate_trace_files(
        episodes=episodes,
        actions=actions,
        requests=requests,
        events=events,
    )
    source_paths = (
        [
            config_path,
            task_audit_path,
            *manifest_paths,
            protocol_decision_path,
            *recorded_protocol_paths,
        ]
        + ([] if no_shift_decision_path is None else [no_shift_decision_path])
        + episode_paths
        + action_paths
        + request_paths
        + event_paths
    )
    source_entries: list[dict[str, str]] = []
    seen_source_paths: set[str] = set()
    for path in source_paths:
        resolved = str(path.resolve())
        if resolved in seen_source_paths:
            continue
        seen_source_paths.add(resolved)
        source_entries.append({"path": resolved, "sha256": file_sha256(path)})
    checks = {
        "m4_baseline_intact": baseline["passed"],
        "repository_tests_and_compile_passed": repository_checks["passed"],
        "frozen_source_binding_passed": True,
        "episode_schema_and_pair_contract_passed": episode_validation["passed"],
        "action_request_event_provenance_passed": trace_validation["passed"],
    }
    core = {
        "schema_version": 1,
        "milestone": "M5-G0",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "evidence_producer_source_sha256": protocol["implementation_source_sha256"],
        "validator_source_sha256": source_sha256,
        "validator_source_files": source_files,
        "source_binding": {
            "mode": source_binding_mode,
            "benchmark_runtime_and_selection_files_unchanged": (
                not drift_paths or downstream_only_patch
            ),
            "post_calibration_validation_patch": (
                "src/actionstream/m5_validation.py" in drift_paths
            ),
            "post_calibration_reporting_orchestration_patch": (
                "scripts/run_m5_g0.sh" in drift_paths
            ),
            "post_calibration_report_rendering_patch": (
                "src/actionstream/m5_report.py" in drift_paths
            ),
            "post_calibration_validation_patch_reason": (
                "The original validator compared the audit condition to an "
                "absent M4 evidence field named condition; the frozen M4 "
                "evidence records the same value under runtime_mode."
                if downstream_only_patch
                else None
            ),
            "post_calibration_reporting_patch_reason": (
                "The original no-go reporting path passed the protocol-wide "
                "hard-stop wording instead of the terminal calibration "
                "decision reason expected by canonical report recomputation, "
                "omitted the report's explicit M4-baseline flag, and the "
                "paper-style Markdown omitted the calibration result table "
                "and downstream source-drift disclosure."
                if downstream_only_patch
                else None
            ),
            "implementation_source_drift": implementation_source_drift,
        },
        "protocol_decision_sha256": protocol["protocol_decision_sha256"],
        "no_shift_decision_sha256": (
            None
            if no_shift_decision is None
            else no_shift_decision["no_shift_decision_sha256"]
        ),
        "m4_baseline_validation": baseline,
        "repository_validation": repository_checks,
        "episode_validation": episode_validation,
        "trace_validation": trace_validation,
        "sources": source_entries,
    }
    return {**core, "validation_sha256": canonical_sha256(core)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-audit", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--protocol-decision", type=Path, required=True)
    parser.add_argument("--no-shift-decision", type=Path)
    parser.add_argument("--episodes", type=Path, nargs="*", default=[])
    parser.add_argument("--actions", type=Path, nargs="*", default=[])
    parser.add_argument("--requests", type=Path, nargs="*", default=[])
    parser.add_argument("--events", type=Path, nargs="*", default=[])
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_validation(
        repository_root=Path(__file__).resolve().parents[2],
        config_path=args.config.resolve(),
        task_audit_path=args.task_audit.resolve(),
        manifest_paths=[path.resolve() for path in args.seed_manifest],
        protocol_decision_path=args.protocol_decision.resolve(),
        no_shift_decision_path=(
            None if args.no_shift_decision is None else args.no_shift_decision.resolve()
        ),
        episode_paths=[path.resolve() for path in args.episodes],
        action_paths=[path.resolve() for path in args.actions],
        request_paths=[path.resolve() for path in args.requests],
        event_paths=[path.resolve() for path in args.events],
        run_tests=args.run_tests,
    )
    if args.output.exists():
        existing = _read_json(args.output, role="existing validation artifact")
        if existing != payload:
            raise ValueError(
                f"Existing validation artifact differs from evidence: {args.output}"
            )
        print(f"{args.output} (validated existing)")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
