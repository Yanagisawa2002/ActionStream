from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from action_stream_benchmark.m8_archive import (
    ARCHIVE_EPISODE_ARTIFACT_FIELDS,
    DIRECT_EPISODE_ARTIFACT_FIELDS,
    build_archive,
    build_archive_matrix,
    read_archive_member,
    validate_archive,
)
from action_stream_benchmark.m8_cli import run as run_cli
from action_stream_benchmark.m8_faults import load_fault_trace
from action_stream_benchmark.m8_matrix import build_matrix_manifest
from action_stream_benchmark.m8_protocol import sha256_file
from action_stream_benchmark.m8_replay import recompute_metrics, validate_manifest
from action_stream_benchmark.schema import (
    canonical_sha256,
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)

from _m8_test_support import cloned_episode, scenario_payload


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _archive_fixture(
    tmp_path: Path,
    *,
    profile_1_base_latency_ms: int = 850,
) -> dict[str, Path | dict]:
    root = tmp_path / "repository"
    (root / ".git").mkdir(parents=True)
    configs = root / "configs"
    configs.mkdir()
    candidate = configs / "candidate.json"
    shutil.copyfile(REPOSITORY_ROOT / "configs" / "m8_g0.json", candidate)
    candidate_payload = read_json(candidate)
    profile_paths = {}
    for profile_id in (
        "profile_0_sanity",
        "profile_1_fixed",
        "profile_2_faults",
    ):
        profile_path = configs / f"{profile_id}.json"
        shutil.copyfile(
            REPOSITORY_ROOT
            / "ros2_ws"
            / "src"
            / "action_stream_benchmark"
            / "config"
            / f"m8_{profile_id}.json",
            profile_path,
        )
        if profile_id in {"profile_1_fixed", "profile_2_faults"}:
            profile_payload = read_json(profile_path)
            profile_payload["profile"]["base_latency_ms"] = profile_1_base_latency_ms
            write_json_atomic(profile_path, profile_payload)
        profile_paths[profile_id] = profile_path
        candidate_payload["profiles"][profile_id].update(
            {
                "profile_file": profile_path.relative_to(root).as_posix(),
                "profile_sha256": sha256_file(profile_path),
            }
        )
    write_json_atomic(candidate, candidate_payload)
    profile = profile_paths["profile_1_fixed"]
    seed = 2026081200
    seeds = configs / "seeds.json"
    write_json_atomic(
        seeds, {"schema_version": 1, "milestone": "M8-G0", "seeds": [seed]}
    )
    evidence = root / "evidence"
    raw = evidence / "raw"
    matrix_path = evidence / "matrix.json"
    matrix = build_matrix_manifest(
        repository_root=root,
        freeze_manifest_path=None,
        candidate_protocol_path=candidate,
        seed_path=seeds,
        profile_paths=[profile],
        output_root=raw,
        manifest_path=matrix_path,
        split="development",
    )
    candidate_sha256 = canonical_sha256(read_json(candidate))
    scenario_core = scenario_payload(seed)
    scenario_sha256 = canonical_sha256(scenario_core)
    scenario_file = (
        matrix_path.parent / matrix["episodes"][0]["scenario_file"]
    ).resolve()
    write_json_atomic(
        scenario_file,
        {**scenario_core, "scenario_sha256": scenario_sha256},
    )
    trace_file = (
        matrix_path.parent / matrix["episodes"][0]["fault_trace_file"]
    ).resolve()
    trace = load_fault_trace(trace_file)

    for entry in matrix["episodes"]:
        rows, summary = cloned_episode(
            strategy=entry["strategy"],
            profile_id=entry["profile_id"],
            seed=seed,
            episode_id=entry["episode_id"],
            split="development",
        )
        enriched = []
        for row in rows:
            enriched.append(row)
            if row["event_type"] != "inference_request":
                continue
            request_id = int(row["request_id"])
            fault = trace.entry(request_id - 1)
            common = {
                key: row[key]
                for key in (
                    "schema_version",
                    "milestone",
                    "evidence_class",
                    "episode_id",
                    "profile_id",
                    "seed",
                    "strategy",
                    "split",
                )
            }
            enriched.extend(
                [
                    {
                        **common,
                        "event_type": "chunk_scheduled",
                        "request_id": request_id,
                        "request_ordinal": request_id - 1,
                        "latency_ms": fault.total_delivery_delay_ms,
                        "trace_sha256": trace.sha256,
                    },
                    {
                        **common,
                        "event_type": "chunk_delivered",
                        "request_id": request_id,
                        "request_ordinal": request_id - 1,
                        "latency_ms": fault.total_delivery_delay_ms,
                        "trace_sha256": trace.sha256,
                        "duplicate": False,
                    },
                ]
            )
        for index, row in enumerate(enriched):
            row["event_index"] = index
        start = enriched[0]
        start["scenario"] = deepcopy(scenario_core)
        start["scenario_sha256"] = scenario_sha256
        start["fault_trace_sha256"] = trace.sha256
        start["protocol_sha256"] = candidate_sha256
        summary["scenario_sha256"] = scenario_sha256
        summary["fault_trace_sha256"] = trace.sha256
        summary["protocol_sha256"] = candidate_sha256
        summary["metrics"] = recompute_metrics(enriched)
        entry["scenario_sha256"] = scenario_sha256
        entry["fault_trace_sha256"] = trace.sha256
        event_path = (matrix_path.parent / entry["event_log_path"]).resolve()
        summary_path = (matrix_path.parent / entry["summary_path"]).resolve()
        write_jsonl_atomic(event_path, enriched)
        write_json_atomic(summary_path, summary)
    write_json_atomic(matrix_path, matrix)
    direct_replay = validate_manifest(matrix_path)
    assert direct_replay["passed"], direct_replay

    archive = evidence / "raw.tar.gz"
    sidecar = evidence / "raw.manifest.json"
    members = sorted(path for path in raw.rglob("*") if path.is_file())
    build_archive(
        root=root,
        members=members,
        archive_path=archive,
        manifest_path=sidecar,
    )
    derived = evidence / "matrix.archive.json"
    build_archive_matrix(
        source_matrix_path=matrix_path,
        archive_path=archive,
        archive_manifest_path=sidecar,
        output_path=derived,
    )
    return {
        "root": root,
        "raw": raw,
        "matrix": matrix_path,
        "archive": archive,
        "sidecar": sidecar,
        "derived": derived,
        "direct_replay": direct_replay,
    }


def test_replay_accepts_nondefault_profile_bound_to_candidate_bytes(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path, profile_1_base_latency_ms=900)
    assert fixture["direct_replay"]["passed"]
    matrix = read_json(fixture["matrix"])
    trace_path = (
        Path(fixture["matrix"]).parent / matrix["episodes"][0]["fault_trace_file"]
    ).resolve()
    assert load_fault_trace(trace_path).profile.base_latency_ms == 900


def test_replay_rejects_self_consistent_trace_profile_divergence(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    matrix_path = Path(fixture["matrix"])
    matrix = read_json(matrix_path)
    trace_path = (
        matrix_path.parent / matrix["episodes"][0]["fault_trace_file"]
    ).resolve()
    trace_payload = read_json(trace_path)
    old_trace_sha = trace_payload["trace_sha256"]
    trace_payload["profile"]["base_latency_ms"] = 900
    trace_core = dict(trace_payload)
    trace_core.pop("trace_sha256")
    new_trace_sha = canonical_sha256(trace_core)
    trace_payload["trace_sha256"] = new_trace_sha
    write_json_atomic(trace_path, trace_payload)

    for entry in matrix["episodes"]:
        entry["fault_trace_sha256"] = new_trace_sha
        event_path = (matrix_path.parent / entry["event_log_path"]).resolve()
        rows = read_jsonl(event_path)
        for row in rows:
            if row.get("fault_trace_sha256") == old_trace_sha:
                row["fault_trace_sha256"] = new_trace_sha
            if row.get("trace_sha256") == old_trace_sha:
                row["trace_sha256"] = new_trace_sha
        write_jsonl_atomic(event_path, rows)
        summary_path = (matrix_path.parent / entry["summary_path"]).resolve()
        summary = read_json(summary_path)
        summary["fault_trace_sha256"] = new_trace_sha
        write_json_atomic(summary_path, summary)
    write_json_atomic(matrix_path, matrix)

    audit = validate_manifest(matrix_path)
    assert audit["passed"] is False
    assert any(
        error.startswith("fault_trace_profile_payload_mismatch:")
        for error in audit["profile_errors"]
    )
    assert all(
        item["invariant_violation_counts"].get("fault_trace_profile_payload_mismatch")
        == 1
        for item in audit["audits"]
    )


def test_deterministic_archive_has_exact_member_hashes(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "a.json").write_text('{"a":1}\n', encoding="utf-8")
    (root / "b.jsonl").write_text('{"b":2}\n', encoding="utf-8")
    first = build_archive(
        root=root,
        members=["a.json", "b.jsonl"],
        archive_path=tmp_path / "first.tar.gz",
        manifest_path=tmp_path / "first.manifest.json",
    )
    second = build_archive(
        root=root,
        members=["b.jsonl", "a.json"],
        archive_path=tmp_path / "second.tar.gz",
        manifest_path=tmp_path / "second.manifest.json",
    )
    assert first["archive_sha256"] == second["archive_sha256"]
    assert (tmp_path / "first.tar.gz").read_bytes() == (
        tmp_path / "second.tar.gz"
    ).read_bytes()
    assert validate_archive(
        tmp_path / "first.tar.gz", tmp_path / "first.manifest.json"
    )["passed"]
    assert (
        read_archive_member(tmp_path / "first.tar.gz", "a.json")
        == (root / "a.json").read_bytes()
    )


def test_archive_rejects_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    try:
        build_archive(
            root=root,
            members=[outside],
            archive_path=tmp_path / "bad.tar.gz",
            manifest_path=tmp_path / "bad.json",
        )
    except ValueError as exc:
        assert "escapes root" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("archive accepted an escaping member")


def test_archive_matrix_replays_without_loose_raw_and_survives_relocation(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    derived = read_json(fixture["derived"])
    for episode in derived["episodes"]:
        assert not set(DIRECT_EPISODE_ARTIFACT_FIELDS) & set(episode)
        assert set(ARCHIVE_EPISODE_ARTIFACT_FIELDS) <= set(episode)

    archive_replay = validate_manifest(fixture["derived"])
    direct_replay = fixture["direct_replay"]
    assert archive_replay["passed"]
    assert archive_replay["fairness_errors"] == direct_replay["fairness_errors"]
    assert archive_replay["provenance_errors"] == direct_replay["provenance_errors"]
    assert [audit["recomputed_metrics"] for audit in archive_replay["audits"]] == [
        audit["recomputed_metrics"] for audit in direct_replay["audits"]
    ]
    assert [
        audit["invariant_violation_counts"] for audit in archive_replay["audits"]
    ] == [audit["invariant_violation_counts"] for audit in direct_replay["audits"]]

    shutil.rmtree(fixture["raw"])
    assert validate_manifest(fixture["derived"])["passed"]
    relocated = tmp_path / "relocated"
    shutil.copytree(fixture["root"], relocated)
    assert validate_manifest(relocated / "evidence" / "matrix.archive.json")["passed"]


def test_archive_matrix_is_non_overwriting_and_rejects_tamper(tmp_path: Path) -> None:
    fixture = _archive_fixture(tmp_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_archive_matrix(
            source_matrix_path=fixture["matrix"],
            archive_path=fixture["archive"],
            archive_manifest_path=fixture["sidecar"],
            output_path=fixture["derived"],
        )

    sidecar = Path(fixture["sidecar"])
    sidecar.write_text(sidecar.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="archive manifest SHA-256 mismatch"):
        validate_manifest(fixture["derived"])


def test_archive_matrix_cli_writes_separate_validated_matrix(tmp_path: Path) -> None:
    fixture = _archive_fixture(tmp_path)
    cli_output = Path(fixture["derived"]).with_name("matrix.archive.cli.json")
    assert (
        run_cli(
            [
                "archive-matrix",
                "--source-matrix",
                str(fixture["matrix"]),
                "--archive",
                str(fixture["archive"]),
                "--archive-manifest",
                str(fixture["sidecar"]),
                "--output",
                str(cli_output),
            ]
        )
        == 0
    )
    assert cli_output != fixture["matrix"]
    assert validate_manifest(cli_output)["passed"]


def test_archive_matrix_rejects_absolute_and_traversing_bindings(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    original = read_json(fixture["derived"])
    for field, bad_path in (
        ("archive", "C:/absolute/raw.tar.gz"),
        ("archive_manifest", "../raw.manifest.json"),
        ("source_matrix", "/absolute/matrix.json"),
    ):
        tampered = deepcopy(original)
        tampered[field] = bad_path
        write_json_atomic(fixture["derived"], tampered)
        with pytest.raises(ValueError, match="traversal-free relative path"):
            validate_manifest(fixture["derived"])


def test_archive_matrix_rejects_member_swaps_between_source_episodes(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    payload = read_json(fixture["derived"])
    payload["episodes"][0]["archive_member"] = payload["episodes"][1]["archive_member"]
    write_json_atomic(fixture["derived"], payload)
    with pytest.raises(ValueError, match="does not correspond to source artifact"):
        validate_manifest(fixture["derived"])


@pytest.mark.parametrize(
    "bad_member", ["../escape.json", "C:/absolute.json", "missing.json"]
)
def test_archive_matrix_rejects_unsafe_or_missing_member(
    tmp_path: Path,
    bad_member: str,
) -> None:
    fixture = _archive_fixture(tmp_path)
    payload = read_json(fixture["derived"])
    payload["episodes"][0]["archive_member"] = bad_member
    write_json_atomic(fixture["derived"], payload)
    with pytest.raises(ValueError, match="unsafe archive member|absent from sidecar"):
        validate_manifest(fixture["derived"])


def test_archive_matrix_rejects_archive_byte_tamper(tmp_path: Path) -> None:
    fixture = _archive_fixture(tmp_path)
    archive = Path(fixture["archive"])
    data = bytearray(archive.read_bytes())
    data[len(data) // 2] ^= 0x01
    archive.write_bytes(data)
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        validate_manifest(fixture["derived"])
