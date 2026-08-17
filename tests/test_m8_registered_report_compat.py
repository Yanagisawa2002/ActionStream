from pathlib import Path

import pytest

from scripts.m8_registered_report_compat import (
    audit_payloads_by_normalized_source,
    resolve_matrix_relative_freeze,
    rewrite_fixed_reproduction_paths,
)


def test_resolve_matrix_relative_freeze_accepts_canonical_parent_reference(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    matrix = root / "outputs" / "holdout" / "matrix.json"
    freeze = root / "outputs" / "protocol" / "freeze.json"
    matrix.parent.mkdir(parents=True)
    freeze.parent.mkdir(parents=True)
    matrix.write_text("{}", encoding="utf-8")
    freeze.write_text("{}", encoding="utf-8")

    assert resolve_matrix_relative_freeze(
        "../protocol/freeze.json",
        matrix_path=matrix,
        repository_root=root,
        supplied_freeze_path=freeze,
    ) == freeze.resolve()


def test_resolve_matrix_relative_freeze_rejects_unrelated_or_external_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    matrix = root / "outputs" / "holdout" / "matrix.json"
    freeze = root / "outputs" / "protocol" / "freeze.json"
    unrelated = root / "outputs" / "other" / "freeze.json"
    matrix.parent.mkdir(parents=True)
    freeze.parent.mkdir(parents=True)
    unrelated.parent.mkdir(parents=True)
    matrix.write_text("{}", encoding="utf-8")
    freeze.write_text("{}", encoding="utf-8")
    unrelated.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="does not identify"):
        resolve_matrix_relative_freeze(
            "../other/freeze.json",
            matrix_path=matrix,
            repository_root=root,
            supplied_freeze_path=freeze,
        )
    with pytest.raises(ValueError, match="escapes"):
        resolve_matrix_relative_freeze(
            "../../../outside.json",
            matrix_path=matrix,
            repository_root=root,
            supplied_freeze_path=freeze,
        )


def test_audit_payloads_normalize_archive_prefix_and_preserve_identity() -> None:
    metrics = {"task_success": True, "completion_steps": 12}
    direct = {
        "audits": [
            {
                "source": "outputs/m8_g0/holdout_v3/raw/events/episode-1.jsonl",
                "strategy": "aligned_async",
                "recomputed_metrics": metrics,
                "invariant_violation_counts": {},
            }
        ]
    }
    archived = {
        "audits": [
            {
                "source": "outputs/m8_g0/holdout_v3/complete_raw.tar.gz!outputs/m8_g0/holdout_v3/raw/events/episode-1.jsonl",
                "strategy": "aligned_async",
                "recomputed_metrics": metrics,
                "invariant_violation_counts": {},
            }
        ]
    }

    assert audit_payloads_by_normalized_source(direct) == audit_payloads_by_normalized_source(
        archived
    )


def test_audit_payloads_reject_duplicate_normalized_source() -> None:
    audit = {
        "source": "outputs/raw/episode-1.jsonl",
        "strategy": "sync_hold",
        "recomputed_metrics": {},
        "invariant_violation_counts": {},
    }
    with pytest.raises(ValueError, match="duplicate normalized"):
        audit_payloads_by_normalized_source({"audits": [audit, dict(audit)]})


def test_rewrite_fixed_reproduction_paths_is_exact_and_fail_closed(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.md"
    report.write_text("run outputs/old/matrix.json now\n", encoding="utf-8")

    records = rewrite_fixed_reproduction_paths(
        report,
        [("outputs/old/matrix.json", "outputs/v3/matrix.json")],
    )

    assert records == [
        {
            "from": "outputs/old/matrix.json",
            "to": "outputs/v3/matrix.json",
            "replacement_count": 1,
        }
    ]
    assert report.read_text(encoding="utf-8") == "run outputs/v3/matrix.json now\n"
    with pytest.raises(ValueError, match="exactly one"):
        rewrite_fixed_reproduction_paths(
            report,
            [("outputs/missing.json", "outputs/v3/missing.json")],
        )
