from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch

from actionstream.inference_transport import (
    InferenceCancelled,
    InferenceDeadlineExceeded,
    InferenceTransportError,
)
from actionstream.rpc_matrix import run_matrix
from actionstream.rpc_transport import (
    RpcFaultProfile,
    RpcInferenceServer,
    TcpInferenceTransport,
)


def test_rpc_round_trip_preserves_nested_tensor_and_array_payloads():
    observed = {}

    def infer(observation, task):
        observed["observation"] = observation
        observed["task"] = task
        return observation["tensor"].sum().reshape(1)

    with RpcInferenceServer(infer) as server:
        transport = TcpInferenceTransport(server.host, server.port)
        transport.reset()
        result = transport.infer(
            {
                "tensor": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                "array": np.arange(4, dtype=np.int16).reshape(2, 2),
                "nested": {"tuple": (b"x", 2)},
            },
            "pick object",
            timeout_s=1.0,
        )

    assert result.tolist() == [15.0]
    assert torch.equal(
        observed["observation"]["tensor"],
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
    )
    assert np.array_equal(
        observed["observation"]["array"],
        np.arange(4, dtype=np.int16).reshape(2, 2),
    )
    assert observed["observation"]["nested"]["tuple"] == (b"x", 2)
    assert observed["task"] == "pick object"


def test_rpc_timeout_closes_connection_and_next_request_reconnects():
    faults = RpcFaultProfile(stall_every_n=2, stall_s=0.1)

    with RpcInferenceServer(
        lambda observation, task: torch.tensor([1.0]),
        fault_profile=faults,
    ) as server:
        transport = TcpInferenceTransport(server.host, server.port)
        transport.reset()
        assert transport.infer({}, "task", timeout_s=0.5).item() == 1.0
        with pytest.raises(InferenceDeadlineExceeded):
            transport.infer({}, "task", timeout_s=0.02)
        assert transport.infer({}, "task", timeout_s=0.5).item() == 1.0
        telemetry = transport.telemetry()

    assert telemetry.requests_started == 3
    assert telemetry.requests_completed == 2
    assert telemetry.deadlines == 1
    assert telemetry.reconnects >= 2


def test_rpc_cancel_invalidates_inflight_generation():
    def infer(observation, task):
        time.sleep(0.15)
        return torch.tensor([1.0])

    with RpcInferenceServer(infer) as server:
        transport = TcpInferenceTransport(server.host, server.port)
        transport.reset()
        result = []

        def request():
            try:
                transport.infer({}, "task", timeout_s=1.0)
            except BaseException as exc:
                result.append(exc)

        thread = threading.Thread(target=request)
        thread.start()
        time.sleep(0.03)
        assert transport.cancel() is True
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert len(result) == 1
        assert isinstance(result[0], InferenceCancelled)
        assert transport.telemetry().cancellations == 1


def test_rpc_reset_calls_remote_worker_reset_when_idle():
    state = {"resets": 0}

    def reset():
        state["resets"] += 1

    with RpcInferenceServer(
        lambda observation, task: torch.tensor([1.0]), reset=reset
    ) as server:
        transport = TcpInferenceTransport(server.host, server.port)
        transport.reset()
        assert transport.infer({}, "task", timeout_s=0.5).item() == 1.0

    assert state["resets"] == 1


def test_rpc_server_error_is_not_misclassified_as_network_disconnect():
    faults = RpcFaultProfile(server_error_every_n=1)
    with RpcInferenceServer(
        lambda observation, task: torch.tensor([1.0]), fault_profile=faults
    ) as server:
        transport = TcpInferenceTransport(server.host, server.port)
        transport.reset()
        with pytest.raises(InferenceTransportError, match="Remote inference failed"):
            transport.infer({}, "task", timeout_s=0.5)
        telemetry = transport.telemetry()

    assert telemetry.server_errors == 1
    assert telemetry.transport_errors == 0


def test_frozen_loopback_fault_matrix_has_expected_failure_taxonomy():
    result = run_matrix(
        {
            "schema_version": 1,
            "cases": [
                {
                    "name": "healthy",
                    "requests": 4,
                    "deadline_ms": 200,
                    "expected_counts": {
                        "success": 4,
                        "deadline": 0,
                        "transport_error": 0,
                        "server_error": 0,
                    },
                },
                {
                    "name": "deadline",
                    "requests": 4,
                    "deadline_ms": 20,
                    "stall_every_n": 2,
                    "stall_ms": 60,
                    "expected_counts": {"success": 2, "deadline": 2},
                },
                {
                    "name": "disconnect",
                    "requests": 4,
                    "deadline_ms": 100,
                    "disconnect_before_infer_every_n": 2,
                    "expected_counts": {"success": 2, "transport_error": 2},
                },
                {
                    "name": "server-error",
                    "requests": 4,
                    "deadline_ms": 100,
                    "server_error_every_n": 2,
                    "expected_counts": {"success": 2, "server_error": 2},
                },
            ],
        }
    )
    assert result["pass"] is True
