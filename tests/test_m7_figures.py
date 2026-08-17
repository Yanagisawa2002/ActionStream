from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "action_stream_policy"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "action_stream_benchmark"))

from action_stream_benchmark import figures  # noqa: E402
from action_stream_benchmark.schema import (  # noqa: E402
    write_json_atomic,
    write_jsonl_atomic,
)


STRATEGIES = ("sync_hold", "naive_async", "aligned_async")


def _analysis() -> dict[str, Any]:
    profiles = {}
    for profile_index, profile_id in enumerate(("profile_a", "profile_b")):
        strategies = {}
        for strategy_index, strategy in enumerate(STRATEGIES):
            successes = 6 + profile_index + strategy_index
            strategies[strategy] = {
                "successes": successes,
                "trials": 10,
                "success_rate": successes / 10,
                "median_wall_clock_seconds": 5.0 - 0.4 * strategy_index,
                "mean_total_hold_seconds": 2.0 - 0.7 * strategy_index,
            }
        profiles[profile_id] = {"strategies": strategies}
    return {"profiles": profiles}


def _events(*, include_rejection: bool = True) -> list[dict[str, Any]]:
    event_types = [
        "inference_request",
        "chunk_arrived",
        "queue_updated",
        "command_executed",
    ]
    if include_rejection:
        event_types.insert(2, "chunk_rejected")
    return [
        {
            "event_type": event_type,
            "event_index": index,
            "wall_time_ns": 9_000_000_000_000 + index * 50_000_000,
            "hold": False,
        }
        for index, event_type in enumerate(event_types)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generate_figures_writes_deterministic_png_pdf_pairs(
    tmp_path: Path,
) -> None:
    analysis_path = tmp_path / "analysis.json"
    event_log = tmp_path / "events.jsonl"
    write_json_atomic(analysis_path, _analysis())
    write_jsonl_atomic(event_log, _events())

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_paths = figures.generate_figures(
        analysis_path=analysis_path,
        example_event_log=event_log,
        output_dir=first,
    )
    second_paths = figures.generate_figures(
        analysis_path=analysis_path,
        example_event_log=event_log,
        output_dir=second,
    )

    assert len(first_paths) == len(second_paths) == 6
    assert {Path(path).suffix for path in first_paths} == {".png", ".pdf"}
    for first_name, second_name in zip(first_paths, second_paths, strict=True):
        first_path = Path(first_name)
        second_path = Path(second_name)
        assert first_path.is_file() and first_path.stat().st_size > 0
        assert second_path.is_file() and second_path.stat().st_size > 0
        assert _sha256(first_path) == _sha256(second_path)
        if first_path.suffix == ".png":
            with Image.open(first_path) as image:
                dpi = image.info["dpi"]
                assert abs(dpi[0] - 300.0) < 0.1
                assert abs(dpi[1] - 300.0) < 0.1
        else:
            pdf = first_path.read_bytes()
            assert pdf.startswith(b"%PDF-")
            assert b"/Subtype /Image" not in pdf


def test_timeline_labels_present_and_absent_categories_honestly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_labels: list[list[str]] = []

    def capture(fig, _path: Path) -> tuple[Path, Path]:
        captured_labels.append([label.get_text() for label in fig.axes[0].get_yticklabels()])
        return tmp_path / "unused.png", tmp_path / "unused.pdf"

    monkeypatch.setattr(figures, "_save", capture)
    complete = tmp_path / "complete.jsonl"
    without_rejection = tmp_path / "without-rejection.jsonl"
    write_jsonl_atomic(complete, _events())
    write_jsonl_atomic(without_rejection, _events(include_rejection=False))

    figures.plot_timeline(complete, tmp_path / "complete.png")
    figures.plot_timeline(without_rejection, tmp_path / "incomplete.png")

    assert captured_labels[0] == [
        "Request",
        "Response",
        "Reject",
        "Queue update",
        "Execute",
    ]
    assert captured_labels[1] == [
        "Request",
        "Response",
        "Reject (none)",
        "Queue update",
        "Execute",
    ]
