from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from actionstream.m4_capture import (
    _annotated_frame,
    _decode_video,
    _encode_video,
    _extract_agentview_frame,
    _validate_pair,
)


def test_extract_agentview_frame_handles_vector_batch_and_render_flip() -> None:
    source = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    source[0, 0, 0] = [1, 2, 3]
    source[0, 1, 2] = [4, 5, 6]
    frame = _extract_agentview_frame({"pixels": {"image": source}})
    assert frame.shape == (2, 3, 3)
    assert frame[1, 2].tolist() == [1, 2, 3]
    assert frame[0, 0].tolist() == [4, 5, 6]


def test_extract_agentview_frame_rejects_missing_pixels() -> None:
    with pytest.raises(ValueError, match="pixels"):
        _extract_agentview_frame({})


def test_annotated_frame_has_fixed_video_shape() -> None:
    record = {
        "task_id": 0,
        "seed": 144,
        "initial_state_index": 4,
        "injected_delay_ms": 950,
        "success": True,
        "environment_steps": 150,
    }
    frame = _annotated_frame(
        np.zeros((256, 256, 3), dtype=np.uint8),
        mode="async_aligned",
        step_index=10,
        record=record,
        queue_before=4,
        queue_after=3,
        hold=False,
    )
    assert frame.shape == (720, 640, 3)
    assert frame.dtype == np.uint8
    assert frame.any()


def test_encode_video_writes_decodable_h264(tmp_path: Path) -> None:
    output = tmp_path / "capture.mp4"
    frames = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.full((32, 32, 3), 127, dtype=np.uint8),
    ]
    record = {
        "task_id": 0,
        "seed": 144,
        "initial_state_index": 4,
        "injected_delay_ms": 950,
        "success": False,
        "environment_steps": 1,
    }
    video_info = _encode_video(
        output,
        frames,
        fps=20,
        mode="async_naive",
        record=record,
        telemetry={
            "queue_depth_before_action": np.asarray([0], dtype=np.int32),
            "queue_depth_after_action": np.asarray([0], dtype=np.int32),
            "queue_hold_mask": np.asarray([True], dtype=np.bool_),
        },
    )
    decoded, rate = _decode_video(output)
    assert video_info["frame_count"] == 2
    assert len(decoded) == 2
    assert rate == 20.0
    assert decoded[0].shape == (720, 640, 3)


def test_validate_pair_requires_same_condition_and_order() -> None:
    shared = {
        "task_id": 0,
        "episode_index": 2,
        "initial_state_index": 4,
        "base_seed": 142,
        "episode_seed": 144,
        "injected_delay_ms": 950,
        "episode_length": 800,
        "replan_interval_steps": 10,
        "model_id": "lerobot/xvla-libero",
        "model_revision": "revision",
    }
    runtime = {"torch": "test", "gpu_name": "test"}
    before = {
        "parameters": shared | {"mode": "async_naive"},
        "runtime_identity": runtime,
    }
    after = {
        "parameters": shared | {"mode": "async_aligned"},
        "runtime_identity": runtime,
    }
    _validate_pair(before, after)
    after["parameters"]["episode_seed"] = 145
    with pytest.raises(ValueError, match="episode_seed"):
        _validate_pair(before, after)
