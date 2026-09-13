"""CPU-only revision check on one control thread; no robot or policy assets."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import torch

from actionstream.lerobot_inference import (
    ActionStreamInferenceConfig,
    ActionStreamInferenceEngine,
)


def main() -> None:
    delivered = threading.Event()
    executed: list[list[float]] = []

    def observer(update):
        if update["status"] == "delivered":
            delivered.set()

    provider = SimpleNamespace(reset=lambda: None)
    engine = ActionStreamInferenceEngine(
        policy=provider,
        preprocessor=provider,
        postprocessor=provider,
        hw_features={},
        task="first goal",
        device="cpu",
        robot_type="mock",
        config=ActionStreamInferenceConfig(delivery_scheduler_enabled=True),
        infer_chunk=lambda obs, task: torch.full((1, 4, 2), float(obs["value"])),
        delivery_observer=observer,
    )

    def dispatch(packet):
        # Lifecycle changes and this validation/write pair share this thread.
        if not engine.is_action_current(packet):
            return False
        executed.append(packet.action.tolist())  # Mock actuator only.
        return True

    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"value": 1})
        if not delivered.wait(3):
            raise RuntimeError("First CPU response did not arrive")
        old = engine.get_action_with_revision(None)
        assert old is not None and dispatch(old)

        engine.set_task("second goal")
        engine.reset()
        rejected = not dispatch(old)
        assert rejected
        delivered.clear()
        engine.notify_observation({"value": 2})
        if not delivered.wait(3):
            raise RuntimeError("Second CPU response did not arrive")
        new = engine.get_action_with_revision(None)
        assert new is not None and dispatch(new)
        assert new.epoch != old.epoch and new.task_revision != old.task_revision
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "old_packet_rejected_after_reset": rejected,
                    "old_epoch": old.epoch,
                    "new_epoch": new.epoch,
                    "mock_actions": executed,
                },
                indent=2,
            )
        )
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
