import pytest

from action_stream_isaac.request_driver import (
    RequestDriverConfig,
    should_request_observation,
    success_from_task_state,
)


def test_request_schedule_and_frozen_success_index() -> None:
    config = RequestDriverConfig("episode", 1, request_interval_steps=10)
    config.validate()
    assert [step for step in range(31) if should_request_observation(step, 10)] == [
        0,
        10,
        20,
        30,
    ]
    state = [0.0] * 12
    assert not success_from_task_state(state)
    state[9] = 1.0
    assert success_from_task_state(state)


def test_request_driver_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="episode_id"):
        RequestDriverConfig("", 0).validate()
    with pytest.raises(ValueError, match="request_interval_steps"):
        RequestDriverConfig("episode", 0, request_interval_steps=30).validate()
    with pytest.raises(ValueError, match="12 finite"):
        success_from_task_state([0.0] * 11)
