from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from actionstream.adaptive_runtime import (
    AdaptiveSelector,
    load_selector_config,
    robot_eef_reference_action,
    rotation_geodesic_radians,
    rotation_matrix_to_axis_angle,
)
from actionstream.current_baselines import (
    ModelSpec,
    _AdaptiveActionQueue,
    _AdaptiveQueue,
    load_protocol,
)
from actionstream.runtime import InferenceResult


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "configs" / "actionstream_adaptive_v1_selector.json"
SELECTOR_SHA256 = "e5cd59bbe760a5d7c9b6e4b1addbb039af31f4962d622af8973944e2125006d5"
SELECTOR_V2_PATH = ROOT / "configs" / "actionstream_adaptive_v2_selector.json"
SELECTOR_V2_SHA256 = "8679606e12b85accc26bd1ad1a1d710bade6682c738cebc031a4196d4a367572"
SELECTOR_V3_PATH = (
    ROOT / "configs" / "actionstream_adaptive_phase_stable_v3_selector.json"
)
SELECTOR_V3_SHA256 = "c824762ed9b05d37369181812795313c870b3bf8f5714d4c4ccd98e221021c98"
SELECTOR_V4_PATH = ROOT / "configs" / "actionstream_adaptive_queue_slack_v4_selector.json"
SELECTOR_V4_SHA256 = "7e437cbaeb3d274e098d8a068fafa482697148933eb5431c28c48cd51632a021"
SELECTOR_V5_PATH = (
    ROOT / "configs" / "actionstream_adaptive_budgeted_release_v5_selector.json"
)
SELECTOR_V5_SHA256 = "51b3dbd120a0a39183a21f68e10d801151d255946c9aa7b8821f022ccf425a1b"
V5_CANARY_PROTOCOLS = (
    ROOT / "configs" / "actionstream_adaptive_budgeted_release_v5_canary_object.json",
    ROOT / "configs" / "actionstream_adaptive_budgeted_release_v5_canary_spatial.json",
    ROOT / "configs" / "actionstream_adaptive_budgeted_release_v5_canary_goal.json",
)
V5_DEV_PROTOCOL = (
    ROOT / "configs" / "actionstream_adaptive_budgeted_release_v5_dev_probe.json"
)


def _chunk() -> np.ndarray:
    actions = np.zeros((30, 7), dtype=np.float32)
    actions[:, 0] = np.linspace(-0.10, -0.071, 30)
    actions[:, 2] = 0.10
    actions[:, 6] = 1.0
    return actions


def _selector() -> AdaptiveSelector:
    config = load_selector_config(SELECTOR_PATH, expected_sha256=SELECTOR_SHA256)
    return AdaptiveSelector(config, chunk_size=30, control_mode="absolute")


def _selector_v2() -> AdaptiveSelector:
    config = load_selector_config(SELECTOR_V2_PATH, expected_sha256=SELECTOR_V2_SHA256)
    return AdaptiveSelector(config, chunk_size=30, control_mode="absolute")


def _selector_v3() -> AdaptiveSelector:
    config = load_selector_config(SELECTOR_V3_PATH, expected_sha256=SELECTOR_V3_SHA256)
    return AdaptiveSelector(config, chunk_size=30, control_mode="absolute")


def _selector_v4() -> AdaptiveSelector:
    config = load_selector_config(SELECTOR_V4_PATH, expected_sha256=SELECTOR_V4_SHA256)
    return AdaptiveSelector(config, chunk_size=30, control_mode="absolute")


def _selector_v5() -> AdaptiveSelector:
    config = load_selector_config(SELECTOR_V5_PATH, expected_sha256=SELECTOR_V5_SHA256)
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


def test_rotation_matrix_round_trips_near_pi_axis_angle() -> None:
    axis_angle = np.asarray([-2.1949704, -2.2026410, 0.0454896])
    angle = float(np.linalg.norm(axis_angle))
    axis = axis_angle / angle
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    matrix = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    recovered = rotation_matrix_to_axis_angle(matrix)
    assert rotation_geodesic_radians(axis_angle, recovered) < 1e-8


def test_v2_uses_frame_local_robot_eef_reference_for_first_chunk() -> None:
    actions = np.tile(
        np.asarray([-0.1919872, 0.0023193, 1.1832070, -2.1949704, -2.2026410, 0.0454896, -1.0]),
        (30, 1),
    ).astype(np.float32)
    observation = {
        "robot_state": [
            {
                "eef": {
                    "pos": np.asarray([[-0.2060072, -0.0073079, 1.1762330]]),
                    "mat": np.asarray(
                        [
                            [
                                [0.0006019, 0.9983447, -0.0575107],
                                [0.9999997, -0.0006262, -0.0004040],
                                [-0.0004393, -0.0575104, -0.9983448],
                            ]
                        ]
                    ),
                }
            }
        ]
    }
    reference = robot_eef_reference_action(observation)
    decision = _selector_v2().decide(
        actions,
        observation_control_step=0,
        current_control_step=0,
        queue_depth_steps=0,
        last_action=None,
        safety_reference_action=reference,
    )

    assert reference[2] == pytest.approx(1.1762330)
    assert decision.execution_mode == "full_chunk"
    assert decision.full_chunk_risk.safe
    assert decision.risk_reference_source == "request_observation_robot_eef"
    assert decision.full_chunk_risk.position_disagreement_m < 0.02


def test_v2_aborts_first_chunk_without_robot_reference() -> None:
    decision = _selector_v2().decide(
        _chunk(),
        observation_control_step=0,
        current_control_step=0,
        queue_depth_steps=0,
        last_action=None,
    )
    assert decision.execution_mode == "safe_hold"
    assert decision.full_chunk_risk.reasons == ("missing_reference_action",)


def test_v3_routes_severe_bounded_age_to_aligned_execution() -> None:
    decision = _selector_v3().decide(
        _chunk(),
        observation_control_step=0,
        current_control_step=23,
        queue_depth_steps=12,
        last_action=np.asarray([-0.10, 0.0, 0.10, 0.0, 0.0, 0.0, -1.0]),
    )

    assert decision.regime == "severe_but_bounded"
    assert decision.execution_mode == "age_aligned"
    assert decision.selected_action_index == 23


def test_v3_latest_only_risk_gates_first_nonstale_command() -> None:
    actions = _chunk()
    actions[0, 0] = 0.80
    decision = _selector_v3().decide(
        actions,
        observation_control_step=0,
        current_control_step=3,
        queue_depth_steps=12,
        last_action=np.asarray([-0.10, 0.0, 0.10, 0.0, 0.0, 0.0, -1.0]),
    )

    assert decision.execution_mode == "full_chunk"
    assert decision.selected_action_index == 3
    assert decision.full_chunk_risk.safe


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


def _v3_result(actions: np.ndarray, observation_step: int) -> InferenceResult:
    safety_reference = np.asarray(actions[0], dtype=np.float32).copy()
    safety_reference[6] = 0.0
    return InferenceResult(
        actions=actions,
        episode_id="episode",
        observation_control_step=observation_step,
        request_timestamp=1.0,
        start_timestamp=2.0,
        end_timestamp=3.0,
        delivery_timestamp=4.0,
        model_inference_latency_seconds=1.0,
        metadata={"safety_reference_action": safety_reference.tolist()},
    )


def _model_spec() -> ModelSpec:
    return ModelSpec(
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


def _v3_queue() -> _AdaptiveQueue:
    selector = load_selector_config(
        SELECTOR_V3_PATH, expected_sha256=SELECTOR_V3_SHA256
    )
    queue = _AdaptiveQueue(
        bindings=_fake_bindings(),
        period=0.05,
        spec=_model_spec(),
        selector_config=selector,
    )
    queue.reset_episode("episode")
    return queue


def _v4_queue() -> _AdaptiveQueue:
    selector = load_selector_config(
        SELECTOR_V4_PATH, expected_sha256=SELECTOR_V4_SHA256
    )
    queue = _AdaptiveQueue(
        bindings=_fake_bindings(),
        period=0.05,
        spec=_model_spec(),
        selector_config=selector,
    )
    queue.reset_episode("episode")
    return queue


def _v5_queue() -> _AdaptiveQueue:
    selector = load_selector_config(
        SELECTOR_V5_PATH, expected_sha256=SELECTOR_V5_SHA256
    )
    queue = _AdaptiveQueue(
        bindings=_fake_bindings(),
        period=0.05,
        spec=_model_spec(),
        selector_config=selector,
    )
    queue.reset_episode("episode")
    return queue


def test_adaptive_queue_switches_between_official_latest_only_and_aligned() -> None:
    selector = load_selector_config(SELECTOR_PATH, expected_sha256=SELECTOR_SHA256)
    queue = _AdaptiveQueue(
        bindings=_fake_bindings(),
        period=0.05,
        spec=_model_spec(),
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


def test_v3_rejected_outage_preserves_nonempty_active_queue() -> None:
    queue = _v3_queue()
    fresh = queue.merge(_v3_result(_chunk(), 0), control_step=0)
    assert fresh["execution_backend"] == "official_lerobot_latest_only"
    queue.pop()

    outage = queue.merge(_v3_result(_chunk(), 0), control_step=29)

    assert outage["reason"] == "adaptive_rejected_chunk_preserved_validated_queue"
    assert outage["execution_backend"] == "official_lerobot_latest_only"
    assert outage["queue_before"] == 29
    assert outage["queue_after"] == 29
    assert outage["guard_discarded_pending_steps"] == 0
    _, held = queue.pop()
    assert not held
    assert queue.summary()["adaptive_pending_actions_discarded_by_guard"] == 0


def test_v3_closed_gripper_phase_lock_rejects_backend_switch() -> None:
    queue = _v3_queue()
    closed = _chunk()
    closed[:, 6] = 1.0
    queue.merge(_v3_result(closed, 0), control_step=0)
    queue.pop()

    moderate = queue.merge(_v3_result(closed, 0), control_step=14)

    assert moderate["reason"] == "adaptive_phase_lock_preserved_validated_queue"
    assert moderate["adaptive_phase_lock_active"]
    assert moderate["execution_backend"] == "official_lerobot_latest_only"
    assert queue.summary()["adaptive_phase_lock_rejections"] == 1


def test_v3_release_confirmation_unlocks_after_three_open_commands() -> None:
    queue = _v3_queue()
    phase_actions = _chunk()
    phase_actions[0, 6] = 1.0
    phase_actions[1:, 6] = -1.0
    queue.merge(_v3_result(phase_actions, 0), control_step=0)
    queue.pop()
    queue.pop()
    queue.pop()
    queue.pop()

    moderate = queue.merge(_v3_result(phase_actions, 0), control_step=14)

    assert moderate["execution_backend"] == "actionstream_age_aligned"
    assert not moderate["adaptive_phase_lock_active"]


def test_v3_minimum_residency_suppresses_pregrasp_oscillation() -> None:
    queue = _v3_queue()
    open_actions = _chunk()
    open_actions[:, 6] = -1.0
    queue.merge(_v3_result(open_actions, 0), control_step=0)
    queue.pop()

    early = queue.merge(_v3_result(open_actions, 0), control_step=6)
    assert early["reason"] == "adaptive_residency_preserved_validated_queue"
    assert early["execution_backend"] == "official_lerobot_latest_only"

    settled = queue.merge(_v3_result(open_actions, 0), control_step=14)
    assert settled["execution_backend"] == "actionstream_age_aligned"


def test_v4_online_service_estimate_triggers_slack_prefetch() -> None:
    queue = _v4_queue()
    queue.note_request(0)
    queue.merge(_v3_result(_chunk(), 0), control_step=0)

    for _ in range(3):
        queue.pop()

    assert queue.queue_depth() == 27
    assert queue.should_request(3)
    queue.note_request(3)
    summary = queue.summary()
    assert summary["adaptive_queue_slack_scheduler_enabled"]
    assert summary["adaptive_service_time_ewma_steps"] == pytest.approx(28.0)
    assert summary["adaptive_last_prefetch_threshold_steps"] == 29
    assert summary["adaptive_last_request_interval_steps"] == 3
    assert summary["adaptive_prefetch_requests"] == 1
    assert summary["adaptive_outstanding_requests_at_end"] == 1


def test_v4_reserve_holds_five_commands_for_unresolved_request() -> None:
    queue = _v4_queue()
    queue.note_request(0)
    queue.merge(_v3_result(_chunk(), 0), control_step=0)
    for _ in range(25):
        _, held = queue.pop()
        assert not held

    assert queue.queue_depth() == 5
    assert queue.should_request(25)
    queue.note_request(25)
    held_action, held = queue.pop()

    assert held
    assert np.isfinite(held_action).all()
    assert queue.queue_depth() == 5
    assert queue.action_trace_state()["adaptive_queue_reserve_active"]
    summary = queue.summary()
    assert summary["adaptive_reserve_activations"] == 1
    assert summary["adaptive_reserve_hold_steps"] == 1


def test_v4_outage_reject_preserves_active_recovery_reserve() -> None:
    queue = _v4_queue()
    queue.note_request(0)
    queue.merge(_v3_result(_chunk(), 0), control_step=0)
    for _ in range(25):
        queue.pop()
    queue.note_request(25)
    queue.pop()

    outage = queue.merge(_v3_result(_chunk(), 0), control_step=29)

    assert outage["reason"] == "adaptive_rejected_chunk_preserved_validated_queue"
    assert outage["queue_before"] == 5
    assert outage["queue_after"] == 5
    assert queue.summary()["adaptive_pending_actions_discarded_by_guard"] == 0


def _prime_v5_reserve(queue: _AdaptiveQueue) -> np.ndarray:
    queue.note_request(0)
    queue.merge(_v3_result(_chunk(), 0), control_step=0)
    for _ in range(25):
        _, held = queue.pop()
        assert not held
    pose = np.asarray([-0.10, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0])
    queue.note_control_observation(pose)
    queue.note_request(25)
    return pose


def test_v5_selector_freezes_bounded_spending_contract() -> None:
    config = load_selector_config(
        SELECTOR_V5_PATH, expected_sha256=SELECTOR_V5_SHA256
    )

    assert config.schema_version == 5
    assert config.queue_reserve_steps == 5
    assert config.reserve_spend_budget_steps == 4
    assert config.minimum_protected_reserve_steps == 1
    assert config.maximum_release_wait_steps == 4
    assert config.progress_window_steps == 3
    assert _selector_v5().config.selector_id.endswith("budgeted_release_v5")


def test_v5_canary_state_and_traces_are_disjoint_from_dev_probe() -> None:
    canaries = [load_protocol(path) for path in V5_CANARY_PROTOCOLS]
    development = load_protocol(V5_DEV_PROTOCOL)

    assert {
        tuple(protocol.raw["environment"]["initial_state_indices"])
        for protocol in canaries
    } == {(33,)}
    assert tuple(development.raw["environment"]["initial_state_indices"]) == (34, 35)
    canary_hashes = {
        trace.trace_sha256
        for protocol in canaries
        for trace in protocol.delays.values()
    }
    development_hashes = {
        trace.trace_sha256 for trace in development.delays.values()
    }
    assert len(canary_hashes) == 7
    assert len(development_hashes) == 2
    assert canary_hashes.isdisjoint(development_hashes)


def test_v5_deadline_releases_one_reserve_command_while_robot_progresses() -> None:
    queue = _v5_queue()
    pose = _prime_v5_reserve(queue)

    for index in range(4):
        _, held = queue.pop()
        assert held
        moving = pose.copy()
        moving[0] += 0.003 * (index + 1)
        queue.note_control_observation(moving)

    _, released_as_hold = queue.pop()

    assert not released_as_hold
    assert queue.queue_depth() == 4
    summary = queue.summary()
    assert summary["adaptive_reserve_spend_steps"] == 1
    assert summary["adaptive_reserve_deadline_spend_steps"] == 1
    assert summary["adaptive_reserve_progress_spend_steps"] == 0
    assert summary["adaptive_last_reserve_release_reason"] == "service_deadline"


def test_v5_progress_stall_releases_before_service_deadline() -> None:
    queue = _v5_queue()
    pose = _prime_v5_reserve(queue)

    for _ in range(3):
        _, held = queue.pop()
        assert held
        queue.note_control_observation(pose)

    _, released_as_hold = queue.pop()

    assert not released_as_hold
    assert queue.queue_depth() == 4
    summary = queue.summary()
    assert summary["adaptive_reserve_progress_spend_steps"] == 1
    assert summary["adaptive_reserve_deadline_spend_steps"] == 0
    assert summary["adaptive_last_progress_translation_m"] == pytest.approx(0.0)
    assert summary["adaptive_last_progress_rotation_radians"] == pytest.approx(0.0)


def test_v5_spend_budget_never_consumes_protected_final_command() -> None:
    queue = _v5_queue()
    pose = _prime_v5_reserve(queue)

    for _ in range(40):
        queue.pop()
        queue.note_control_observation(pose)
        if queue.summary()["adaptive_reserve_spend_steps"] == 4:
            break

    assert queue.queue_depth() == 1
    spent_before = queue.summary()["adaptive_reserve_spend_steps"]
    for _ in range(6):
        _, held = queue.pop()
        assert held
        queue.note_control_observation(pose)

    summary = queue.summary()
    assert summary["adaptive_reserve_spend_steps"] == spent_before == 4
    assert summary["adaptive_reserve_budget_exhaustions"] == 1
    assert summary["adaptive_reserve_protected_floor_hold_steps"] > 0
    assert queue.queue_depth() == 1


def test_v5_requires_finite_measured_eef_progress_observation() -> None:
    queue = _v5_queue()

    with pytest.raises(ValueError, match="finite measured EEF pose"):
        queue.note_control_observation(np.asarray([np.nan] * 7))
