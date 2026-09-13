"""Lifecycle regressions: deterministic scheduling, no GPU or remote calls.

Assertions cover reset at the final queue commit and old-generation failure.
The merge wrapper only pauses a worker at a real unlocked scheduling boundary;
it does not change the production merge or reset behavior.
"""

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from actionstream.lerobot_inference import (
    ActionStreamInferenceConfig,
    ActionStreamInferenceEngine,
)


class Resettable:
    config = SimpleNamespace(use_amp=False)

    def reset(self):
        pass


def make_engine(infer, **config):
    return ActionStreamInferenceEngine(
        policy=Resettable(),
        preprocessor=Resettable(),
        postprocessor=Resettable(),
        hw_features={},
        task="audit",
        device="cpu",
        robot_type="mock",
        config=ActionStreamInferenceConfig(**config),
        infer_chunk=infer,
    )


def wait_until(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("audit synchronization timed out")


@pytest.mark.parametrize("pipelined", [False, True])
@pytest.mark.parametrize("transition", ["reset", "stop"])
def test_reset_after_acceptance_check_must_not_repopulate_queue(
    pipelined, transition, tmp_path
):
    before_merge = threading.Event()
    continue_merge = threading.Event()
    merge_finished = threading.Event()
    engine = make_engine(
        lambda obs, task: torch.full((1, 4, 2), 37.0),
        delivery_scheduler_enabled=pipelined,
        telemetry_jsonl_path=str(tmp_path / "events.jsonl"),
    )
    original_merge = engine._merge_chunk

    def paused_merge(*args, **kwargs):
        before_merge.set()
        assert continue_merge.wait(3)
        try:
            return original_merge(*args, **kwargs)
        finally:
            merge_finished.set()

    engine._merge_chunk = paused_merge
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"episode": "old"})
        assert before_merge.wait(3)
        stopper = None
        if transition == "reset":
            engine.reset()
        else:
            stopper = threading.Thread(target=engine.stop)
            stopper.start()
            wait_until(engine._shutdown.is_set)
        assert engine.telemetry.queue_depth == 0
        continue_merge.set()
        assert merge_finished.wait(3)
        if stopper is not None:
            stopper.join(3)
            assert not stopper.is_alive()
        assert engine.telemetry.queue_depth == 0
        leaked_action = engine.get_action(None)
        assert leaked_action is None, (
            "An old-episode action was delivered after reset without a new observation: "
            f"{leaked_action.tolist()}"
        )
    finally:
        continue_merge.set()
        engine.stop()


def test_old_generation_failure_must_not_fail_reset_episode():
    entered = threading.Event()
    release = threading.Event()

    def failing_old_request(obs, task):
        if obs["episode"] == "new":
            return torch.full((1, 4, 2), 73.0)
        entered.set()
        assert release.wait(3)
        raise ConnectionError("audit old-generation disconnect")

    engine = make_engine(
        failing_old_request, max_consecutive_failures=1, retry_backoff_s=0
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"episode": "old"})
        assert entered.wait(3)
        engine.reset()
        engine.notify_observation({"episode": "new"})
        release.set()
        wait_until(lambda: engine.failed or engine.telemetry.chunks_accepted > 0)
        assert not engine.failed, "A pre-reset failure poisoned the new episode"
        torch.testing.assert_close(engine.get_action(None), torch.tensor([73.0, 73.0]))
    finally:
        release.set()
        engine.stop()
