"""Real spawn/IPC lifecycle tests with a CPU owner, without learned models."""

import os
from types import SimpleNamespace
import time

import numpy as np
import pytest

from actionstream.llm_vla.async_process import ProcessInferenceMixin


def owner(connection, assets, seed, batch):
    from actionstream.lerobot_backend import InferenceOutput

    ordinal = 0
    connection.send(("ready", os.getpid()))
    while True:
        command, payload = connection.recv()
        if command == "close":
            break
        if command == "reset":
            connection.send(("reset", None))
        if command == "infer":
            public, instruction = payload
            assert set(public) == {"pixels", "robot_state"}
            ordinal += 1
            actions = np.zeros((30, 7), np.float32)
            actions[:, 0] = ordinal
            connection.send(
                ("result", InferenceOutput(actions, 0.0, (1, 30, 20), "torch.float32"))
            )
    connection.close()


class Recorder:
    def close(self):
        self.closed = True


class Port(ProcessInferenceMixin, Recorder):
    inference_process_target = staticmethod(owner)
    backend = SimpleNamespace(model_id="/unused/xvla")
    seed = 5
    batch_postprocessing = True


def test_spawned_owner_survives_reset_without_replaying_warmup_rng():
    port = Port()
    observation = SimpleNamespace(native_input=lambda: dict(pixels={}, robot_state={}))
    try:
        port.reset_inference()
        pid = port.inference_owner_pid
        assert pid != os.getpid()
        first, receipt = port.infer(observation, "instruction")
        port.reset_inference()
        second, receipt2 = port.infer(observation, "instruction")
        assert receipt["inference_pid"] == receipt2["inference_pid"] == pid
        assert np.all(first[:, 0] == 1) and np.all(second[:, 0] == 2)
    finally:
        port.close()
    assert port.closed and not port.inference_process.is_alive()


def test_unresponsive_owner_receive_is_bounded_and_cleanup_still_reaps_process():
    port = Port()
    try:
        port.reset_inference()
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            port._receive("unsolicited", timeout=0.02)
        assert time.monotonic() - started < 1
    finally:
        port.close()
    assert not port.inference_process.is_alive()
