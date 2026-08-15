from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from actionstream.current_baselines import DelayTrace, load_protocol
from actionstream.runtime import (
    InferencePayload,
    InferenceRequest,
    LatestRequestWorker,
)


ROOT = Path(__file__).resolve().parents[1]


def test_current_protocol_freezes_models_delays_and_rtc_boundary() -> None:
    protocol = load_protocol(ROOT / "configs" / "current_lerobot_baselines.json")
    assert protocol.raw["source"]["commit"] == "6adf51511b7625090eade8d82d9f61a1846ebe56"
    assert set(protocol.models) == {"xvla", "pi05", "smolvla"}
    assert protocol.models["xvla"].rtc_expected is False
    assert protocol.models["pi05"].rtc_expected is True
    assert protocol.models["pi05"].control_mode == "relative"
    smol = next(model for model in protocol.raw["models"] if model["key"] == "smolvla")
    assert smol["dependencies"] == [
        {
            "model_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            "revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        }
    ]
    assert set(protocol.delays) == {
        "fixed_0000",
        "fixed_0250",
        "fixed_0500",
        "fixed_0950",
        "jitter_0500_pm0250",
    }


def test_no_grad_allows_current_lerobot_rtc_to_reenable_autograd() -> None:
    # Current RTC computes a guidance correction inside torch.enable_grad().
    # This is the exact semantic distinction from torch.inference_mode(), which
    # cannot be overridden by the upstream RTC block.
    with torch.no_grad():
        with torch.enable_grad():
            value = torch.tensor(2.0, requires_grad=True)
            output = value.square()
            gradient = torch.autograd.grad(output, value)[0]
    assert gradient.item() == pytest.approx(4.0)


def test_seeded_jitter_trace_is_bounded_deterministic_and_nonrepeating() -> None:
    definition = {
        "key": "jitter",
        "kind": "seeded_uniform",
        "center_milliseconds": 500,
        "half_width_milliseconds": 250,
        "seed": 17,
        "trace_length": 4,
    }
    first = DelayTrace.from_mapping(definition)
    second = DelayTrace.from_mapping(json.loads(json.dumps(definition)))
    assert first.milliseconds == second.milliseconds
    assert first.trace_sha256 == second.trace_sha256
    assert all(250 <= delay <= 750 for delay in first.milliseconds)
    with pytest.raises(IndexError, match="exhausted"):
        first.seconds_at(4)


def test_fixed_delay_repeats_without_exhaustion() -> None:
    trace = DelayTrace.from_mapping(
        {"key": "fixed", "kind": "fixed", "milliseconds": 950}
    )
    assert trace.seconds_at(0) == pytest.approx(0.95)
    assert trace.seconds_at(10000) == pytest.approx(0.95)


def test_worker_records_per_request_delay_trace_metadata() -> None:
    def infer(_request: InferenceRequest) -> InferencePayload:
        return InferencePayload(
            actions=np.ones((2, 7), dtype=np.float32),
            model_inference_latency_seconds=0.001,
        )

    observed: list[tuple[int, int]] = []

    def delay(request: InferenceRequest, ordinal: int) -> float:
        observed.append((request.observation_control_step, ordinal))
        return 0.001 * (ordinal + 1)

    worker = LatestRequestWorker(infer, delivery_delay_seconds=delay)
    try:
        worker.reset_episode("episode")
        worker.submit(InferenceRequest({}, "task", "episode", 3, time.monotonic()))
        assert worker.wait_for_result(timeout=2)
        result = worker.drain_results()[0]
        assert observed == [(3, 0)]
        assert result.metadata["delay_trace_index"] == 0
        assert result.metadata["injected_delivery_delay_seconds"] == pytest.approx(0.001)
    finally:
        worker.close()


def test_protocol_rejects_duplicate_keys(tmp_path: Path) -> None:
    source = json.loads(
        (ROOT / "configs" / "current_lerobot_baselines.json").read_text(encoding="utf-8")
    )
    source["models"].append(dict(source["models"][0]))
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="must be unique"):
        load_protocol(path)
