from __future__ import annotations

from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from actionstream.adaptive_runtime import (
    AdaptiveSelector,
    load_selector_config,
    rotation_geodesic_radians,
)
from actionstream.current_baselines import ModelSpec, _AdaptiveActionQueue, _AdaptiveQueue
from actionstream.runtime import InferenceResult


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "configs" / "actionstream_adaptive_v1_selector.json"
SELECTOR_SHA256 = "e5cd59bbe760a5d7c9b6e4b1addbb039af31f4962d622af8973944e2125006d5"


def _chunk() -> np.ndarray:
    actions = np.zeros((30, 7), dtype=np.float32)
    actions[:, 0] = np.linspace(-0.10, -0.071, 30)
    actions[:, 2] = 0.10
    actions[:, 6] = 1.0
    return actions


def _selector() -> AdaptiveSelector:
    config = load_selector_config(SELECTOR_PATH, expected_sha256=SELECTOR_SHA256)
    return AdaptiveSelector(config, chunk_size=30, control_mode="absolute")


@pytest.mark.parametrize(
    ("age", "regime", "mode", "index"),
    [
        (3, "fresh", "full_chunk", 0),
        (14, "moderate", "age_aligned", 14),
        (23, "severe_but_bounded", "full_chunk", 0),
        (29, "outage", "safe_hold", None),
    ],
)
def test_frozen_age_regimes_select_expected_execution(
    age: int,
    regime: str,
    mode: str,
    index: int | None,
) -> None:
    decision = _selector().decide(
        _chunk(),
        observation_control_step=10,
        current_control_step=10 + age,
        queue_depth_steps=12,
        last_action=np.asarray([-0.10, 0.0, 0.10, 0.0, 0.0, 0.0, 1.0]),
    )

    assert decision.regime == regime
    assert decision.execution_mode == mode
    assert decision.selected_action_index == index


def test_risk_gate_uses_safe_alternate_before_hold() -> None:
    actions = _chunk()
    actions[0, 0] = 0.80
    decision = _selector().decide(
        actions,
        observation_control_step=0,
        current_control_step=3,
        queue_depth_steps=5,
        last_action=np.asarray([-0.10, 0.0, 0.10, 0.0, 0.0, 0.0, 1.0]),
    )

    assert decision.regime == "fresh"
    assert decision.execution_mode == "age_aligned"
    assert decision.safe_alternate_override
    assert decision.preferred_action_index == 0
    assert decision.selected_action_index == 3
    assert decision.full_chunk_risk.reasons == ("outside_workspace", "position_disagreement")
    assert decision.aligned_risk.safe


def test_risk_gate_holds_when_both_candidates_are_unsafe() -> None:
    actions = _chunk()
    actions[:, 0] = 0.80
    decision = _selector().decide(
        actions,
        observation_control_step=0,
        current_control_step=14,
        queue_depth_steps=5,
        last_action=np.asarray([-0.10, 0.0, 0.10, 0.0, 0.0, 0.0, 1.0]),
    )

    assert decision.execution_mode == "safe_hold"
    assert decision.reason == "moderate_both_candidates_failed_risk_gate"
    assert not decision.full_chunk_risk.safe
    assert not decision.aligned_risk.safe


def test_selector_rejects_relative_action_contract() -> None:
    config = load_selector_config(SELECTOR_PATH, expected_sha256=SELECTOR_SHA256)
    with pytest.raises(ValueError, match="absolute actions"):
        AdaptiveSelector(config, chunk_size=30, control_mode="relative")


def test_selector_protocol_hash_is_mandatory() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_selector_config(SELECTOR_PATH, expected_sha256="0" * 64)


def test_rotation_disagreement_uses_so3_geodesic() -> None:
    assert rotation_geodesic_radians(
        np.zeros(3), np.asarray([0.0, 0.0, np.pi / 2.0])
    ) == pytest.approx(np.pi / 2.0)


def test_adaptive_safe_hold_discards_pending_but_retains_last_action() -> None:
    actions = _chunk()[:4]
    result = InferenceResult(
        actions=actions,
        episode_id="episode",
        observation_control_step=0,
        request_timestamp=1.0,
        start_timestamp=2.0,
        end_timestamp=3.0,
        delivery_timestamp=4.0,
        model_inference_latency_seconds=1.0,
    )
    queue = _AdaptiveActionQueue()
    queue.reset_episode("episode")
    queue.replace(result, current_control_step=0, mode="async_naive")
    first, held = queue.next_action()
    assert not held
    assert queue.discard_pending_for_hold() == 3
    repeated, held_repeat = queue.next_action()
    assert held_repeat
    np.testing.assert_array_equal(repeated, first)


class _TimedAction:
    def __init__(self, action: torch.Tensor, timestep: int) -> None:
        self._action = action
        self._timestep = timestep

    def get_action(self) -> torch.Tensor:
        return self._action

    def get_timestep(self) -> int:
        return self._timestep


class _FakePolicyServer:
    @staticmethod
    def _time_action_chunk(
        _server: object,
        _capture_timestamp: float,
        actions: list[torch.Tensor],
        observation_timestep: int,
    ) -> list[_TimedAction]:
        return [
            _TimedAction(action, observation_timestep + index)
            for index, action in enumerate(actions)
        ]


class _FakeRobotClient:
    @staticmethod
    def _aggregate_action_queues(
        client: object,
        timed_actions: list[_TimedAction],
        _aggregate: object,
    ) -> None:
        with client.action_queue_lock:
            client.action_queue.queue.clear()
            for item in timed_actions:
                if item.get_timestep() > client.latest_action:
                    client.action_queue.queue.append(item)

    @staticmethod
    def control_loop_action(client: object, *, verbose: bool) -> None:
        del verbose
        with client.action_queue_lock:
            item = client.action_queue.queue.popleft()
        with client.latest_action_lock:
            client.latest_action = item.get_timestep()

    @staticmethod
    def _ready_to_send_observation(client: object) -> bool:
        return client.action_queue.qsize() <= client.action_chunk_size // 2


def _fake_bindings() -> SimpleNamespace:
    return SimpleNamespace(
        PolicyServer=_FakePolicyServer,
        RobotClient=_FakeRobotClient,
        aggregate_functions={"latest_only": object()},
        torch=torch,
    )


def _result(actions: np.ndarray, observation_step: int) -> InferenceResult:
    return InferenceResult(
        actions=actions,
        episode_id="episode",
        observation_control_step=observation_step,
        request_timestamp=1.0,
        start_timestamp=2.0,
        end_timestamp=3.0,
        delivery_timestamp=4.0,
        model_inference_latency_seconds=1.0,
    )


def test_adaptive_queue_switches_between_official_latest_only_and_aligned() -> None:
    spec = ModelSpec(
        key="xvla",
        model_id="model",
        revision="revision",
        control_mode="absolute",
        chunk_size=30,
        request_interval_steps=10,
        rtc_expected=False,
        rtc_execution_horizon=10,
        required=True,
        rename_map={},
    )
    selector = load_selector_config(SELECTOR_PATH, expected_sha256=SELECTOR_SHA256)
    queue = _AdaptiveQueue(
        bindings=_fake_bindings(),
        period=0.05,
        spec=spec,
        selector_config=selector,
    )
    queue.reset_episode("episode")

    fresh = queue.merge(_result(_chunk(), 0), control_step=0)
    assert fresh["execution_backend"] == "official_lerobot_latest_only"
    assert fresh["queue_after"] == 30
    queue.pop()

    moderate = queue.merge(_result(_chunk(), 0), control_step=14)
    assert moderate["execution_backend"] == "actionstream_age_aligned"
    assert moderate["dropped_prefix_steps"] == 14
    queue.pop()

    severe = queue.merge(_result(_chunk(), 0), control_step=23)
    assert severe["execution_backend"] == "official_lerobot_latest_only"
    assert severe["queue_after"] == 7
    assert severe["queued_timesteps"][0] == 23
