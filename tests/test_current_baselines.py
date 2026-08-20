from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from actionstream.current_baselines import (
    CurrentInferenceOutput,
    CurrentLeRobotBackend,
    DelayTrace,
    _base_record,
    load_protocol,
    run_actionstream_backend_episode,
    run_gpu_warmup,
)
from actionstream.lerobot_backend import StepOutput
from actionstream.lerobot_inference import ActionStreamInferenceConfig
from actionstream.runtime import (
    InferencePayload,
    InferenceRequest,
    LatestRequestWorker,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_backend_gpu_v1_freezes_candidate_and_disjoint_canary_holdout() -> None:
    expected_tasks = {
        "object": ("libero_object", 5),
        "spatial": ("libero_spatial", 7),
        "goal": ("libero_goal", 2),
    }
    candidate_hashes = {
        "current_baselines_sha256": _sha256(
            ROOT / "src" / "actionstream" / "current_baselines.py"
        ),
        "lerobot_inference_sha256": _sha256(
            ROOT / "src" / "actionstream" / "lerobot_inference.py"
        ),
    }

    holdout_seeded_trace_hashes: set[str] = set()
    canary_seeded_trace_hashes: set[str] = set()
    for family, (suite, task_id) in expected_tasks.items():
        canary = load_protocol(
            ROOT / "configs" / f"actionstream_backend_gpu_xvla_canary_{family}_v1.json"
        )
        holdout = load_protocol(
            ROOT / "configs" / f"actionstream_backend_gpu_xvla_holdout_{family}_v1.json"
        )
        assert canary.raw["candidate"] | candidate_hashes == canary.raw["candidate"]
        assert holdout.raw["candidate"] | candidate_hashes == holdout.raw["candidate"]
        assert canary.raw["environment"]["suite"] == suite
        assert holdout.raw["environment"]["task_ids"] == [task_id]
        assert canary.raw["environment"]["initial_state_indices"] == [37]
        assert holdout.raw["environment"]["initial_state_indices"] == [40, 41, 42, 43, 44]
        assert canary.raw["gpu_warmup"]["initial_state_index"] == 36
        assert holdout.raw["gpu_warmup"]["initial_state_index"] == 38
        assert canary.raw["environment"]["base_seed"] != holdout.raw["environment"]["base_seed"]
        canary_seeded_trace_hashes.update(
            item.trace_sha256
            for item in canary.delays.values()
            if item.definition["kind"].startswith("seeded_")
        )
        holdout_seeded_trace_hashes.update(
            item.trace_sha256
            for item in holdout.delays.values()
            if item.definition["kind"].startswith("seeded_")
        )

    assert canary_seeded_trace_hashes.isdisjoint(holdout_seeded_trace_hashes)
    assert len(canary_seeded_trace_hashes) == 3
    assert len(holdout_seeded_trace_hashes) == 6

    for family, (suite, task_id) in expected_tasks.items():
        holdout = load_protocol(
            ROOT
            / "configs"
            / f"actionstream_backend_gpu_smolvla_rtc_holdout_{family}_v1.json"
        )
        assert holdout.raw["candidate"] | candidate_hashes == holdout.raw["candidate"]
        assert holdout.raw["environment"]["suite"] == suite
        assert holdout.raw["environment"]["task_ids"] == [task_id]
        assert holdout.raw["runtimes"] == [
            "sync_hold",
            "lerobot_latest_only",
            "lerobot_rtc",
        ]


def test_xvla_task_seed_expansion_is_disjoint_and_paired() -> None:
    original = load_protocol(ROOT / "configs" / "current_lerobot_baselines.json")
    expansion = load_protocol(
        ROOT / "configs" / "xvla_task_seed_expansion_20260816.json"
    )

    assert expansion.raw["experiment_id"] == "xvla_task_seed_expansion_20260816"
    assert set(expansion.models) == {"xvla"}
    assert expansion.raw["environment"]["task_ids"] == [0, 1, 2]
    assert expansion.raw["environment"]["episode_length"] == 280
    assert expansion.raw["compact_matrix"] == {
        "episodes_per_task": 10,
        "paired": True,
        "reuse_delay_trace_across_runtimes": True,
    }
    assert expansion.raw["runtimes"] == [
        "lerobot_latest_only",
        "actionstream_aligned",
    ]
    assert set(expansion.delays) == {
        "fixed_0000",
        "fixed_0950",
        "jitter_0500_pm0250",
    }
    assert expansion.raw["environment"]["base_seed"] > original.raw["environment"][
        "base_seed"
    ]
    assert set(expansion.raw["environment"]["initial_state_indices"]).isdisjoint(
        original.raw["environment"]["initial_state_indices"]
    )


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


def test_disabling_rtc_clears_upstream_processor_references() -> None:
    backend = CurrentLeRobotBackend.__new__(CurrentLeRobotBackend)
    backend.spec = SimpleNamespace(key="fake")
    backend.policy = SimpleNamespace(
        supports_rtc=lambda: True,
        config=SimpleNamespace(rtc_config=object()),
        rtc_processor=object(),
        model=SimpleNamespace(rtc_processor=object()),
    )

    backend.configure_rtc(enabled=False)

    assert backend.policy.config.rtc_config is None
    assert backend.policy.rtc_processor is None
    assert backend.policy.model.rtc_processor is None


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


def test_seeded_burst_trace_is_deterministic_and_contains_outages() -> None:
    definition = {
        "key": "burst",
        "kind": "seeded_burst",
        "base_center_milliseconds": 350,
        "base_half_width_milliseconds": 150,
        "burst_probability": 0.25,
        "burst_min_milliseconds": 1500,
        "burst_max_milliseconds": 2400,
        "seed": 2026081899,
        "trace_length": 128,
    }
    first = DelayTrace.from_mapping(definition)
    second = DelayTrace.from_mapping(dict(definition))
    assert first.milliseconds == second.milliseconds
    assert first.trace_sha256 == second.trace_sha256
    assert any(200 <= item <= 500 for item in first.milliseconds)
    assert any(1500 <= item <= 2400 for item in first.milliseconds)


def test_scripted_burst_trace_hits_only_frozen_request_ordinals() -> None:
    trace = DelayTrace.from_mapping(
        {
            "key": "phase_burst",
            "kind": "scripted_burst",
            "base_milliseconds": 350,
            "burst_milliseconds": 1700,
            "burst_ordinals": [5, 13],
            "trace_length": 16,
        }
    )

    assert trace.milliseconds[5] == 1700
    assert trace.milliseconds[13] == 1700
    assert all(
        value == (1700 if index in {5, 13} else 350)
        for index, value in enumerate(trace.milliseconds)
    )
    with pytest.raises(IndexError, match="exhausted"):
        trace.seconds_at(16)


def test_scripted_override_supports_a_frozen_fresh_return_pulse() -> None:
    trace = DelayTrace.from_mapping(
        {
            "key": "closed_phase_fresh",
            "kind": "scripted_override",
            "base_milliseconds": 350,
            "overrides": [{"ordinal": 8, "milliseconds": 0}],
            "trace_length": 12,
        }
    )

    assert trace.milliseconds[8] == 0
    assert all(
        value == (0 if index == 8 else 350)
        for index, value in enumerate(trace.milliseconds)
    )


def test_scripted_override_rejects_duplicate_ordinals() -> None:
    with pytest.raises(ValueError, match="Invalid scripted override"):
        DelayTrace.from_mapping(
            {
                "key": "duplicate",
                "kind": "scripted_override",
                "base_milliseconds": 350,
                "overrides": [
                    {"ordinal": 2, "milliseconds": 0},
                    {"ordinal": 2, "milliseconds": 50},
                ],
                "trace_length": 12,
            }
        )


def test_adaptive_protocol_freezes_selector_tasks_states_and_network_traces() -> None:
    canary = load_protocol(
        ROOT / "configs" / "actionstream_adaptive_v1_canary_object.json"
    )
    holdout = load_protocol(
        ROOT / "configs" / "actionstream_adaptive_v1_holdout_object.json"
    )

    assert canary.adaptive_selector is not None
    assert canary.adaptive_selector.selector_id == "actionstream_adaptive_v1"
    assert canary.raw["environment"]["task_ids"] == [3]
    assert canary.raw["environment"]["initial_state_indices"] == [13]
    assert holdout.raw["environment"]["task_ids"] == [4]
    assert holdout.raw["environment"]["initial_state_indices"] == [20, 21, 22, 23, 24]
    assert set(holdout.raw["runtimes"]) == {
        "lerobot_latest_only",
        "actionstream_aligned",
        "actionstream_adaptive",
    }
    assert len(holdout.delays) == 6
    assert set(canary.raw["environment"]["initial_state_indices"]).isdisjoint(
        holdout.raw["environment"]["initial_state_indices"]
    )


def test_phase_stable_v3_canary_is_disjoint_and_uses_scripted_phase_outages() -> None:
    paths = {
        "libero_object": (
            ROOT / "configs" / "actionstream_adaptive_phase_stable_v3_canary_object.json",
            5,
        ),
        "libero_spatial": (
            ROOT / "configs" / "actionstream_adaptive_phase_stable_v3_canary_spatial.json",
            7,
        ),
        "libero_goal": (
            ROOT / "configs" / "actionstream_adaptive_phase_stable_v3_canary_goal.json",
            2,
        ),
    }

    for suite, (path, task_id) in paths.items():
        protocol = load_protocol(path)
        assert protocol.raw["environment"]["suite"] == suite
        assert protocol.raw["environment"]["task_ids"] == [task_id]
        assert protocol.raw["environment"]["initial_state_indices"] == [30]
        assert protocol.adaptive_selector is not None
        assert (
            protocol.adaptive_selector.selector_id
            == "actionstream_adaptive_phase_stable_v3"
        )
        trace = protocol.delays["phase_outage_requests_05_13"]
        assert trace.milliseconds[5] == 1700
        assert trace.milliseconds[13] == 1700
        assert trace.milliseconds[4] == 350


def test_phase_stable_v3_coverage_probe_uses_new_state_and_frozen_pulses() -> None:
    expected_pulse = {"object": 10, "spatial": 8, "goal": 8}
    for suite, ordinal in expected_pulse.items():
        protocol = load_protocol(
            ROOT
            / "configs"
            / f"actionstream_adaptive_phase_stable_v3_coverage_{suite}.json"
        )
        assert protocol.raw["environment"]["initial_state_indices"] == [31]
        assert protocol.raw["runtimes"] == ["actionstream_adaptive"]
        outage = protocol.delays["early_outage_requests_02_05"]
        assert outage.milliseconds[2] == 1700
        assert outage.milliseconds[5] == 1700
        pulse_key = next(
            key for key in protocol.delays if key.startswith("closed_phase_fresh")
        )
        pulse = protocol.delays[pulse_key]
        assert pulse.milliseconds[ordinal] == 0
        assert pulse.milliseconds[ordinal - 1] == 350


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


def test_episode_record_uses_selected_protocol_experiment_id() -> None:
    backend = SimpleNamespace(
        suite="libero_object",
        spec=SimpleNamespace(
            key="xvla",
            model_id="model",
            revision="revision",
            control_mode="absolute",
        )
    )
    record = _base_record(
        experiment_id="adaptive-canary",
        run_id="run",
        backend=backend,
        runtime="actionstream_adaptive",
        profile=DelayTrace.from_mapping(
            {"key": "fixed_0000", "kind": "fixed", "milliseconds": 0}
        ),
        task_id=3,
        episode_index=0,
        initial_state_index=13,
        seed=17,
    )
    assert record["experiment_id"] == "adaptive-canary"
    assert record["suite"] == "libero_object"


class _BackendBenchmarkHarness:
    def __init__(self) -> None:
        self.spec = SimpleNamespace(
            key="fake",
            model_id="fake/model",
            revision="f" * 40,
            control_mode="relative",
            chunk_size=3,
            request_interval_steps=1,
        )
        self.suite = "libero_goal"
        self.task_ids = [2]
        self.policy = SimpleNamespace(config=SimpleNamespace(device="cpu"))
        self.preprocessor = object()
        self.postprocessor = object()
        self.episode_length = 8
        self.peak_cuda_memory_mib = 0.0
        self.steps = 0
        self.inferences = 0

    def reset_episode(self, **_kwargs):
        self.steps = 0
        self.inferences = 0
        return {"state": np.zeros((1, 4), dtype=np.float32)}, {}, "do the task"

    def controller_frequency_hz(self, _task_id: int) -> float:
        return 100.0

    def infer_action_chunk(self, _observation, _instruction) -> CurrentInferenceOutput:
        self.inferences += 1
        actions = np.full((3, 7), self.inferences, dtype=np.float32)
        return CurrentInferenceOutput(
            actions=actions,
            raw_actions=actions.copy(),
            model_latency_seconds=0.001,
            raw_shape=(1, 3, 7),
            raw_dtype="torch.float32",
        )

    def step(self, _task_id: int, _action: np.ndarray) -> StepOutput:
        self.steps += 1
        return StepOutput(
            observation={"state": np.full((1, 4), self.steps, dtype=np.float32)},
            reward=float(self.steps >= 5),
            terminated=False,
            truncated=False,
            success=self.steps >= 5,
            info={},
        )


def test_gpu_warmup_is_non_scored_and_records_latency() -> None:
    backend = _BackendBenchmarkHarness()

    record = run_gpu_warmup(
        backend,
        {
            "task_id": 2,
            "initial_state_index": 38,
            "seed": 2026082038,
            "inference_calls": 2,
        },
    )

    assert record["status"] == "completed_non_scored_warmup"
    assert record["initial_state_index"] == 38
    assert record["inference_calls"] == 2
    assert record["inference_latency_seconds"] == [0.001, 0.001]
    assert backend.steps == 0


def test_formal_backend_episode_records_disconnect_recovery_and_queue_telemetry(
    tmp_path: Path,
) -> None:
    backend = _BackendBenchmarkHarness()
    profile = DelayTrace.from_mapping(
        {
            "key": "disconnect_recovery",
            "kind": "fixed",
            "milliseconds": 0,
            "disconnect_ordinals": [1],
        }
    )

    record = run_actionstream_backend_episode(
        backend,
        experiment_id="backend-formal-test",
        runtime="actionstream_backend_guarded",
        engine_config=ActionStreamInferenceConfig(
            inference_timeout_s=1.0,
            bounded_hold_steps=1,
            retry_backoff_s=0.0,
            max_consecutive_failures=3,
            join_timeout_s=1.0,
            latest_only_fallback=True,
        ),
        profile=profile,
        task_id=2,
        episode_index=0,
        initial_state_index=40,
        seed=2026082100,
        run_id="test-run",
        trace_path=tmp_path / "trace.json",
    )

    assert record["status"] == "completed"
    assert record["success"] is True
    assert record["suite"] == "libero_goal"
    assert record["disconnects"] == 1
    assert record["recoveries"] == 1
    assert record["inference_errors"] == 1
    assert record["queue_depth_p50_steps"] is not None
    assert record["environment_steps_per_second"] > 0
    assert Path(record["trace_path"]).is_file()
