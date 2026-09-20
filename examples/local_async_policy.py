"""CPU-only ActionStream lifecycle demo: alignment plus reset rejection."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import torch

from actionstream.lerobot_inference import (
    ActionStreamInferenceConfig,
    ActionStreamInferenceEngine,
)


class Resettable:
    def reset(self) -> None:
        pass


def wait_for(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise RuntimeError("demo timed out")


def main() -> None:
    stale_entered = threading.Event()
    stale_release = threading.Event()
    reset_entered = threading.Event()
    reset_release = threading.Event()

    def infer(observation: dict, _task: str) -> torch.Tensor:
        phase = observation.get("phase")
        if phase == "stale":
            stale_entered.set()
            if not stale_release.wait(2):
                raise RuntimeError("stale demo was not released")
        if phase == "reset":
            reset_entered.set()
            if not reset_release.wait(2):
                raise RuntimeError("reset demo was not released")
        base = float(observation["value"])
        return torch.arange(base, base + 4, dtype=torch.float32).reshape(1, 4, 1)

    policy = Resettable()
    policy.config = SimpleNamespace(use_amp=False)
    engine = ActionStreamInferenceEngine(
        policy=policy,
        preprocessor=Resettable(),
        postprocessor=Resettable(),
        hw_features={},
        task="cpu-demo",
        device="cpu",
        robot_type="mock",
        config=ActionStreamInferenceConfig(
            bounded_hold_steps=1,
            retry_backoff_s=0.0,
        ),
        infer_chunk=infer,
        reset_provider=lambda: None,
    )

    engine.reset()
    engine.start()
    engine.resume()
    try:
        # The first request starts at controller step 0. Two newer observations
        # arrive while inference is blocked, so two expired actions are discarded.
        engine.notify_observation({"value": 10, "phase": "stale"})
        if not stale_entered.wait(2):
            raise RuntimeError("inference did not start")
        engine.notify_observation({"value": 20})
        engine.notify_observation({"value": 30})
        engine.pause()
        stale_release.set()

        wait_for(lambda: engine.telemetry.chunks_accepted == 1)
        aligned = engine.get_action(None)
        if aligned is None or aligned.item() != 12:
            raise RuntimeError(f"expected age-aligned action 12, got {aligned}")
        print("alignment: dropped 2 expired actions ->", aligned.item())

        # A reset advances the generation while the old request is still running.
        # When that call returns, its result cannot repopulate the new episode.
        engine.reset()
        engine.resume()
        engine.notify_observation({"value": 40, "phase": "reset"})
        if not reset_entered.wait(2):
            raise RuntimeError("reset-race inference did not start")
        engine.reset()
        engine.resume()
        engine.notify_observation({"value": 50})
        reset_release.set()

        wait_for(lambda: engine.telemetry.chunks_rejected_reset == 1)
        wait_for(lambda: engine.telemetry.chunks_accepted == 1)
        current = engine.get_action(None)
        if current is None or current.item() != 50:
            raise RuntimeError(f"expected new-generation action 50, got {current}")
        print("reset: rejected old generation ->", current.item())
    finally:
        stale_release.set()
        reset_release.set()
        engine.stop()


if __name__ == "__main__":
    main()
