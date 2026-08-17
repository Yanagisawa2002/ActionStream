from __future__ import annotations

from pathlib import Path

import pytest

from action_stream_benchmark.m8_cli import run
from action_stream_benchmark.m8_figures import (
    generate_analysis_figures,
    generate_episode_timeline,
)
from action_stream_benchmark.schema import read_json, write_json_atomic, write_jsonl_atomic


def _analysis() -> dict:
    strategies = {}
    for index, method in enumerate(("sync_hold", "naive_async", "aligned_async")):
        strategies[method] = {
            "successes": 50 + index,
            "trials": 60,
            "success_rate": (50 + index) / 60,
            "median_penalized_recovery_latency_steps": 20 - 3 * index,
            "mean_obsolete_destination_command_steps": 8 - 2 * index,
        }
    return {
        "schema_version": 1,
        "milestone": "M8-G0",
        "evidence_class": "ros_cpp_isaac_sim",
        "episode_count": 420,
        "classification": "GO",
        "profiles": {"profile_1_fixed": {"strategies": strategies}},
    }


def _timeline_rows() -> list[dict]:
    return [
        {
            "event_index": 0,
            "event_type": "inference_request",
            "request_id": 1,
            "generation_id": 1,
            "sim_step": 80,
        },
        {
            "event_index": 1,
            "event_type": "destination_switched",
            "generation_id": 2,
            "generation_after": 2,
            "switch_step": 100,
        },
        {
            "event_index": 2,
            "event_type": "inference_request",
            "request_id": 2,
            "generation_id": 2,
            "sim_step": 100,
        },
        {
            "event_index": 3,
            "event_type": "chunk_arrived",
            "request_id": 2,
            "generation_id": 2,
            "out_of_order": False,
            "sim_step": 100,
        },
        {
            "event_index": 4,
            "event_type": "queue_updated",
            "reason": "aligned_atomic_rebuild",
            "request_id": 2,
            "generation_id": 2,
            "sim_step": 100,
        },
        {
            "event_index": 5,
            "event_type": "chunk_arrived",
            "request_id": 1,
            "generation_id": 1,
            "out_of_order": True,
            "sim_step": 80,
        },
        {
            "event_index": 6,
            "event_type": "chunk_rejected",
            "reason": "stale_generation",
            "request_id": 1,
            "generation_id": 1,
            "sim_step": 80,
        },
    ]


def test_publication_figures_are_vector_pdf_and_300dpi_png(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.json"
    write_json_atomic(analysis, _analysis())
    manifest = generate_analysis_figures(analysis, tmp_path / "figures")
    assert len(manifest["figures"]) == 3
    for figure in manifest["figures"]:
        formats = {item["format"]: item for item in figure["files"]}
        assert formats["pdf"]["vector"] is True
        assert formats["png"]["dpi"] == 300
        assert (tmp_path / "figures" / formats["pdf"]["path"]).read_bytes().startswith(b"%PDF")
        assert (tmp_path / "figures" / formats["png"]["path"]).read_bytes().startswith(b"\x89PNG")


def test_figures_refuse_non_native_or_empty_analysis(tmp_path: Path) -> None:
    payload = _analysis()
    payload["evidence_class"] = "ros_cpp_test_plant"
    path = tmp_path / "analysis.json"
    write_json_atomic(path, payload)
    with pytest.raises(ValueError, match="native-Isaac"):
        generate_analysis_figures(path, tmp_path / "figures")


def test_timeline_uses_recorded_order_and_requires_all_causal_roles(tmp_path: Path) -> None:
    event_log = tmp_path / "episode.jsonl"
    write_jsonl_atomic(event_log, _timeline_rows())
    output = tmp_path / "figures" / "representative_timeline.svg"
    timeline = generate_episode_timeline(event_log, output)

    assert timeline["coordinate"] == "recorded_event_index"
    assert timeline["event_count"] == 6
    assert list(timeline["causal_roles"]) == [
        "old_request",
        "destination_switch",
        "new_request",
        "queue_replacement",
        "out_of_order_response",
        "obsolete_action_rejection",
    ]
    assert timeline["causal_roles"]["out_of_order_response"]["event_index"] == 5
    assert "recorded event order" in output.read_text(encoding="utf-8")

    incomplete = tmp_path / "incomplete.jsonl"
    write_jsonl_atomic(incomplete, _timeline_rows()[:-1])
    with pytest.raises(ValueError, match="stale rejection"):
        generate_episode_timeline(incomplete, tmp_path / "bad.svg")


def test_cli_persists_timeline_binding_in_figure_manifest(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.json"
    event_log = tmp_path / "episode.jsonl"
    output = tmp_path / "figures"
    write_json_atomic(analysis, _analysis())
    write_jsonl_atomic(event_log, _timeline_rows())

    assert (
        run(
            [
                "figures",
                "--analysis",
                str(analysis),
                "--output-dir",
                str(output),
                "--timeline-log",
                str(event_log),
            ]
        )
        == 0
    )
    stored = read_json(output / "figure_manifest.json")
    assert stored["timeline"]["coordinate"] == "recorded_event_index"
    timeline_hash = stored["timeline"]["sha256"]
    assert len(timeline_hash) == 64
    assert (output / "representative_timeline.svg").is_file()
