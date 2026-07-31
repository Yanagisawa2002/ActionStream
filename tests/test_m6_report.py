from __future__ import annotations

import json
from pathlib import Path

from actionstream.m6_conformance import canonical_sha256, file_sha256
from actionstream.m6_report import (
    FAMILY_ORDER,
    MILESTONE,
    RUNTIME_ORDER,
    SOURCE_REFERENCE_SPECS,
    validate_report_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "m6_g0"
REPORT = OUTPUT / "report"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_checked_in_m6_report_is_complete_and_scope_honest() -> None:
    report = (REPORT / "m6_g0_report.md").read_text(encoding="utf-8")
    assert "# M6-G0:" in report
    assert "M5-G0 remains **NO-GO**" in report
    assert "PoseGuard is not being revived" in report
    assert "FoundationPose is not part of M6-G0" in report
    assert "RTC was audited but not forced" in report
    assert "**NO-GO**" in report
    assert "**termination of this direction**" in report
    assert "At 1.17H, ActionStream executed no action" in report
    assert "temporal error is unavailable rather than zero" in report


def test_source_crosswalk_has_every_frozen_pinned_reference() -> None:
    source = _json(REPORT / "source_references.json")
    upstream = _json(OUTPUT / "upstream" / "pinned_upstream.json")
    rendered_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPORT / "source_semantic_crosswalk.md",
            REPORT / "portability_audit.md",
            REPORT / "m6_g0_report.md",
        )
    )
    assert source["upstream_commit"] == upstream["commit"]
    assert set(source["references"]) == set(SOURCE_REFERENCE_SPECS)
    for reference in source["references"].values():
        assert upstream["commit"] in reference["permalink"]
        assert reference["symbol"] in rendered_sources
        assert reference["start_line"] <= reference["end_line"]


def test_report_validator_reproduces_checked_in_pass() -> None:
    decision = _json(OUTPUT / "decision.json")
    validation = _json(OUTPUT / "validation.json")
    aggregate = _json(OUTPUT / "metrics" / "aggregate_metrics.json")
    source = _json(REPORT / "source_references.json")
    actual = validate_report_artifacts(
        report=(REPORT / "m6_g0_report.md").read_text(encoding="utf-8"),
        crosswalk=(REPORT / "source_semantic_crosswalk.md").read_text(encoding="utf-8"),
        portability=(REPORT / "portability_audit.md").read_text(encoding="utf-8"),
        source_payload=source,
        decision=decision,
        validation=validation,
        aggregate_rows=aggregate["rows"],
    )
    assert actual["passed"], actual["errors"]
    assert len(aggregate["rows"]) == len(RUNTIME_ORDER) * len(FAMILY_ORDER)


def test_decision_and_late_generation_metrics_are_honest() -> None:
    decision = _json(OUTPUT / "decision.json")
    core = {key: value for key, value in decision.items() if key != "decision_sha256"}
    assert decision["milestone"] == MILESTONE
    assert decision["decision_sha256"] == canonical_sha256(core)
    assert decision["classification"] == "NO-GO"
    assert (
        decision["measurements"]["family_median_temporal_error_reduction_fraction"][
            "one_twenty_percent_horizon"
        ]
        is None
    )

    rows = _json(OUTPUT / "metrics" / "aggregate_metrics.json")["rows"]
    lookup = {(row["runtime"], row["latency_family"]): row for row in rows}
    official = lookup[("lerobot_latest_only", "late_after_newer_generation")]
    aligned = lookup[("actionstream_aligned", "late_after_newer_generation")]
    assert official["late_results_accepted"] == 4
    assert official["late_results_rejected"] == 0
    assert aligned["late_results_accepted"] == 0
    assert aligned["late_results_rejected"] == 4


def test_representative_timeline_contains_full_queue_transition_contract() -> None:
    timeline = _json(REPORT / "representative_timeline.json")
    assert timeline["latency_family"] == "out_of_order"
    assert set(timeline["runtimes"]) == {
        "lerobot_latest_only",
        "actionstream_aligned",
    }
    required = {
        "request_id",
        "request_generation",
        "observation_step",
        "result_arrival_step",
        "queue_insertion_step",
        "old_queue_intended_steps",
        "new_chunk_intended_steps",
        "discarded_prefix_intended_steps",
        "new_queue_intended_steps",
    }
    for runtime in timeline["runtimes"].values():
        assert runtime["executions"]
        assert runtime["queue_transitions"]
        assert required <= set(runtime["queue_transitions"][0])


def test_artifact_manifest_hashes_every_m6_file() -> None:
    manifest = _json(OUTPUT / "artifact_manifest.json")
    core = {
        key: value
        for key, value in manifest.items()
        if key != "artifact_manifest_sha256"
    }
    assert manifest["artifact_manifest_sha256"] == canonical_sha256(core)
    paths = {
        path.relative_to(OUTPUT).as_posix(): path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    assert set(manifest["artifacts"]) == set(paths)
    for relative, path in paths.items():
        assert manifest["artifacts"][relative] == file_sha256(path)
