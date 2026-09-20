"""CPU event-gated lifecycle tests; no policy weights, CUDA or GPU outcomes."""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest
import torch

from actionstream.inference_transport import (
    InferenceCancelled,
    InferenceDeadlineExceeded,
)
from actionstream.rpc_transport import (
    RpcFaultProfile,
    RpcInferenceServer,
    TcpInferenceTransport,
)


def wait_until(predicate):
    limit = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < limit, "condition did not become true"
        time.sleep(0.001)


def launch(fn):
    result = []

    def run():
        try:
            result.append(fn())
        except BaseException as exc:
            result.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result


def finish(thread, result):
    thread.join(3)
    assert not thread.is_alive()
    assert len(result) == 1
    return result[0]


class GatedWorker:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.threads = []
        self.active = 0
        self.maximum_active = 0

    def __call__(self, observation, task):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.calls.append(task)
        self.threads.append(threading.get_ident())
        try:
            if task == "A":
                self.entered.set()
                assert self.release.wait(5), "test failed to release worker"
            return torch.tensor([float(observation.get("value", 1))])
        finally:
            self.active -= 1


def test_deadline_reconnect_does_not_replace_executor_or_accept_old_result(tmp_path):
    worker = GatedWorker()
    with RpcInferenceServer(
        worker, telemetry_jsonl_path=tmp_path / "server.jsonl"
    ) as server:
        executor = server._executor
        transport = TcpInferenceTransport(server.host, server.port)
        transport.reset()
        first = launch(lambda: transport.infer({"value": 11}, "A", timeout_s=0.15))
        try:
            assert worker.entered.wait(2)
            assert isinstance(finish(*first), InferenceDeadlineExceeded)
            # A is still executing even though its client generation is closed.
            assert worker.active == 1
            second = launch(lambda: transport.infer({"value": 22}, "B", timeout_s=2))
            wait_until(lambda: server._jobs.qsize() == 1)
            assert server._executor is executor
            assert worker.calls == ["A"]
            worker.release.set()
            assert finish(*second).item() == 22
            assert transport.telemetry().requests_completed == 1
            assert transport.telemetry().cancelled_invalidated_requests == 1
            assert transport.telemetry().reconnections == 1
            assert server.telemetry()["invalidated_results"] == 1
        finally:
            worker.release.set()
            transport.close()
    assert worker.calls == ["A", "B"]
    assert len(set(worker.threads)) == 1
    assert worker.maximum_active == 1
    assert server.telemetry()["executor_starts"] == 1
    assert not executor.is_alive()
    events = [
        json.loads(x) for x in (tmp_path / "server.jsonl").read_text().splitlines()
    ]
    assert any(x["event"] == "invalidated_result" for x in events)


@pytest.mark.parametrize("invalidate", ["reset", "close", "cancel", "deadline"])
def test_obsolete_queued_request_never_enters_compute(invalidate):
    worker = GatedWorker()
    with RpcInferenceServer(worker) as server:
        first_client = TcpInferenceTransport(server.host, server.port)
        first_client.reset()
        queued_client = TcpInferenceTransport(server.host, server.port)
        queued_client.reset()
        first = launch(lambda: first_client.infer({}, "A", timeout_s=3))
        try:
            assert worker.entered.wait(2)
            queued = launch(
                lambda: queued_client.infer(
                    {}, "obsolete", timeout_s=0.15 if invalidate == "deadline" else 3
                )
            )
            wait_until(lambda: server._jobs.qsize() == 1)
            resetting = None
            if invalidate == "reset":
                resetting = launch(queued_client.reset)
            elif invalidate != "deadline":
                getattr(queued_client, invalidate)()
            assert isinstance(
                finish(*queued),
                InferenceDeadlineExceeded
                if invalidate == "deadline"
                else InferenceCancelled,
            )
            worker.release.set()
            assert finish(*first).item() == 1
            if resetting is not None:
                assert finish(*resetting) is True
            assert (
                queued_client.infer({"value": 3}, "new-generation", timeout_s=2).item()
                == 3
            )
            assert worker.calls == ["A", "new-generation"]
            assert server.telemetry()["dropped_obsolete_before_compute"] == 1
        finally:
            worker.release.set()
            first_client.close()
            queued_client.close()


def test_raw_disconnect_before_queued_compute_is_dropped():
    from actionstream.rpc_transport import _send_frame

    worker = GatedWorker()
    with RpcInferenceServer(worker) as server:
        client = TcpInferenceTransport(server.host, server.port)
        client.reset()
        first = launch(lambda: client.infer({}, "A", timeout_s=3))
        try:
            assert worker.entered.wait(2)
            with socket.create_connection((server.host, server.port)) as raw:
                _send_frame(
                    raw,
                    dict(
                        version=1,
                        kind="infer",
                        request_id="queued",
                        observation={},
                        task="obsolete",
                    ),
                )
                wait_until(lambda: server._jobs.qsize() == 1)
                raw.shutdown(socket.SHUT_RDWR)
            worker.release.set()
            assert finish(*first).item() == 1
            wait_until(
                lambda: server.telemetry()["dropped_obsolete_before_compute"] == 1
            )
            assert worker.calls == ["A"]
        finally:
            worker.release.set()
            client.close()


def test_shutdown_waits_for_inflight_and_drops_queued_work():
    worker = GatedWorker()
    server = RpcInferenceServer(worker)
    acceptor = server.start_in_thread()
    one = TcpInferenceTransport(server.host, server.port)
    one.reset()
    two = TcpInferenceTransport(server.host, server.port)
    two.reset()
    first = launch(lambda: one.infer({}, "A", timeout_s=3))
    try:
        assert worker.entered.wait(2)
        second = launch(lambda: two.infer({}, "queued", timeout_s=3))
        wait_until(lambda: server._jobs.qsize() == 1)
        shutdown = launch(server.close)
        wait_until(server._stop.is_set)
        assert shutdown[0].is_alive()
        assert worker.active == 1
        worker.release.set()
        assert finish(*shutdown) is None
        assert not server._executor.is_alive()
        assert isinstance(finish(*first), Exception)
        assert isinstance(finish(*second), Exception)
        assert worker.calls == ["A"]
        assert server.telemetry()["dropped_obsolete_before_compute"] == 1
    finally:
        worker.release.set()
        one.close()
        two.close()
        server.close()
        acceptor.join(2)
    assert not acceptor.is_alive()
    assert not server._threads


def test_reset_is_serialized_on_the_same_executor():
    worker = GatedWorker()
    reset_threads = []

    def reset():
        assert worker.active == 0
        reset_threads.append(threading.get_ident())

    with RpcInferenceServer(worker, reset=reset) as server:
        client = TcpInferenceTransport(server.host, server.port)
        client.reset()
        controller = TcpInferenceTransport(server.host, server.port)
        reset_threads.clear()  # Initial client reset precedes the ordering under test.
        first = launch(lambda: client.infer({}, "A", timeout_s=3))
        try:
            assert worker.entered.wait(2)
            resetting = launch(controller.reset)
            wait_until(lambda: server._jobs.qsize() == 1)
            assert not reset_threads
            worker.release.set()
            assert finish(*first).item() == 1
            assert finish(*resetting) is False
            assert reset_threads == worker.threads
        finally:
            worker.release.set()
            client.close()
            controller.close()


def test_startup_steady_reconnect_budgets_and_control_does_not_consume_startup(
    tmp_path,
):
    with RpcInferenceServer(
        lambda obs, task: torch.ones(1),
        fault_profile=RpcFaultProfile(response_delay_s=0.12),
    ) as server:
        transport = TcpInferenceTransport(
            server.host,
            server.port,
            startup_inference_timeout_s=0.8,
            steady_inference_timeout_s=0.03,
            telemetry_jsonl_path=tmp_path / "client.jsonl",
        )
        transport.reset()  # Opens socket, but is not an inference request.
        assert transport.next_inference_timeout_s(0.01) == 0.8
        assert transport.infer({}, "first", timeout_s=0.01).item() == 1
        first = transport.telemetry().latest_response_telemetry
        assert transport.next_inference_timeout_s(0.01) == 0.03
        with pytest.raises(InferenceDeadlineExceeded):
            transport.infer({}, "second", timeout_s=5)
        assert transport.infer({}, "reconnected", timeout_s=0.01).item() == 1
        third = transport.telemetry().latest_response_telemetry
        assert first["connection_id"] != third["connection_id"]
        assert (
            first["connection_request_ordinal"]
            == third["connection_request_ordinal"]
            == 1
        )
        assert first["global_request_ordinal"] == 1
        assert third["global_request_ordinal"] == 3
        assert transport.telemetry().steady_deadlines == 1
        assert transport.telemetry().startup_deadlines == 0
        assert transport.telemetry().deadlines == 1
        transport.close()
    rows = [json.loads(x) for x in (tmp_path / "client.jsonl").read_text().splitlines()]
    assert [x["deadline_class"] for x in rows] == ["startup", "steady", "startup"]
    assert [x["deadline_s"] for x in rows] == [0.8, 0.03, 0.8]


def test_startup_deadline_is_counted_separately():
    with RpcInferenceServer(
        lambda obs, task: torch.ones(1),
        fault_profile=RpcFaultProfile(response_delay_s=0.15),
    ) as server:
        client = TcpInferenceTransport(
            server.host,
            server.port,
            startup_inference_timeout_s=0.03,
            steady_inference_timeout_s=0.8,
        )
        client.reset()
        for _ in range(2):
            with pytest.raises(InferenceDeadlineExceeded):
                client.infer({}, "task", timeout_s=5)
        telemetry = client.telemetry()
        assert telemetry.deadlines == telemetry.startup_deadlines == 2
        assert telemetry.steady_deadlines == 0
        assert telemetry.reconnects == telemetry.connections_established == 2
        assert telemetry.reconnections == 1
        client.close()


def test_timing_separates_queue_compute_service_and_delivery():
    worker = GatedWorker()
    with RpcInferenceServer(
        worker, fault_profile=RpcFaultProfile(response_delay_s=0.06)
    ) as server:
        one = TcpInferenceTransport(server.host, server.port)
        one.reset()
        two = TcpInferenceTransport(server.host, server.port)
        two.reset()
        first = launch(lambda: one.infer({}, "A", timeout_s=3))
        try:
            assert worker.entered.wait(2)
            second = launch(lambda: two.infer({}, "B", timeout_s=3))
            wait_until(lambda: server._jobs.qsize() == 1)
            time.sleep(
                0.06
            )  # Deliberate queue residence, not thread-order synchronization.
            worker.release.set()
            assert finish(*first).item() == finish(*second).item() == 1
            row = two.telemetry().latest_response_telemetry
            assert row["server_queue_wait_s"] >= 0.05
            assert row["server_service_s"] >= row["server_compute_s"] >= 0
            assert row["server_inference_s"] >= (
                row["server_queue_wait_s"] + row["server_compute_s"]
            )
            assert row["server_inference_s"] <= (
                row["server_queue_wait_s"] + row["server_service_s"]
            )
            assert (
                two.telemetry().latest_roundtrip_latency_ms / 1000
                >= row["server_inference_s"] + 0.05
            )
            assert "server_lock_wait_s" not in row
        finally:
            worker.release.set()
            one.close()
            two.close()


def test_cancel_during_connect_cannot_publish_socket_into_new_generation(monkeypatch):
    with RpcInferenceServer(lambda obs, task: torch.ones(1)) as server:
        entered = threading.Event()
        release = threading.Event()
        create_connection = socket.create_connection

        def blocked_connect(*args, **kwargs):
            connection = create_connection(*args, **kwargs)
            entered.set()
            assert release.wait(3)
            return connection

        client = TcpInferenceTransport(server.host, server.port)
        client.reset()
        client.cancel()  # Exercise a new socket within the confirmed episode.
        monkeypatch.setattr(socket, "create_connection", blocked_connect)
        pending = launch(lambda: client.infer({}, "old", timeout_s=2))
        try:
            assert entered.wait(2)
            client.cancel()
            release.set()
            assert isinstance(finish(*pending), InferenceCancelled)
            assert client._socket is None
            assert server.telemetry()["inference_calls"] == 0
            assert client.infer({}, "new", timeout_s=2).item() == 1
        finally:
            release.set()
            client.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"startup_inference_timeout_s": 1},
        {"startup_inference_timeout_s": -1, "steady_inference_timeout_s": 1},
        {"startup_inference_timeout_s": 1, "steady_inference_timeout_s": float("nan")},
    ],
)
def test_invalid_deadline_configuration(kwargs):
    with pytest.raises(ValueError):
        TcpInferenceTransport("127.0.0.1", 50051, **kwargs)


def test_engine_honors_transport_startup_budget_and_records_it(tmp_path):
    from types import SimpleNamespace
    from actionstream.lerobot_inference import (
        ActionStreamInferenceConfig,
        ActionStreamInferenceEngine,
    )

    stub = SimpleNamespace(reset=lambda: None)
    with RpcInferenceServer(
        lambda obs, task: torch.ones(30, 7),
        fault_profile=RpcFaultProfile(response_delay_s=0.1),
    ) as server:
        client = TcpInferenceTransport(
            server.host,
            server.port,
            startup_inference_timeout_s=0.8,
            steady_inference_timeout_s=0.3,
        )
        client.reset()
        engine = ActionStreamInferenceEngine(
            policy=stub,
            preprocessor=stub,
            postprocessor=stub,
            hw_features={},
            task="cpu",
            device="cpu",
            robot_type="mock",
            config=ActionStreamInferenceConfig(
                inference_timeout_s=0.02,
                telemetry_jsonl_path=str(tmp_path / "engine.jsonl"),
            ),
            infer_chunk=lambda obs, task: None,
            reset_provider=lambda: None,
            transport=client,
        )
        engine.start()
        engine.resume()
        try:
            engine.notify_observation({"value": 1})
            wait_until(lambda: engine.telemetry.chunks_accepted == 1)
            assert engine.telemetry.inference_timeouts == 0
        finally:
            engine.stop()
    rows = [json.loads(x) for x in (tmp_path / "engine.jsonl").read_text().splitlines()]
    started = [row for row in rows if row["event"] == "inference_started"]
    assert started[0]["deadline_s"] == 0.8


def test_error_episode_persists_final_rpc_counters_without_libero_or_gpu(
    tmp_path, monkeypatch
):
    import actionstream.libero_rpc_client as episode

    class FakeEnvironment:
        def __init__(self, **kwargs):
            pass

        def reset(self, **kwargs):
            return {}, {}, "cpu", 20.0

        def step(self, action):
            raise AssertionError("deadline test must not apply an action")

        def close(self):
            pass

    monkeypatch.setattr(episode, "LiberoEnvironmentClient", FakeEnvironment)
    with RpcInferenceServer(
        lambda obs, task: torch.ones(30, 7),
        fault_profile=RpcFaultProfile(response_delay_s=0.5),
    ) as server:
        args = episode.build_parser().parse_args(
            [
                "--suite",
                "libero_object",
                "--task-id",
                "5",
                "--initial-state-index",
                "25",
                "--seed",
                "2026091725",
                "--host",
                server.host,
                "--port",
                str(server.port),
                "--output",
                str(tmp_path),
                "--inference-timeout-s",
                "0.03",
                "--action-wait-timeout-s",
                "0.15",
            ]
        )
        with pytest.raises(TimeoutError, match="No remote action"):
            episode.run_episode(args)
    receipt = json.loads((tmp_path / "episode_receipt.json").read_text())
    assert receipt["status"] == "ERROR"
    assert receipt["success"] is None
    assert receipt["control_steps"] == 0
    assert receipt["rpc_telemetry"]["deadlines"] >= 1
    assert receipt["engine_telemetry"]
    assert receipt["rpc_telemetry_sha256"]
    assert receipt["engine_telemetry_sha256"]


def test_streaming_response_cannot_extend_absolute_deadline():
    from actionstream.rpc_transport import _recv_exact

    first, second = socket.socketpair()
    stop = threading.Event()

    def drip():
        while not stop.wait(0.02):
            try:
                second.sendall(b"x")
            except OSError:
                return

    thread = threading.Thread(target=drip)
    thread.start()
    try:
        deadline = time.monotonic() + 0.1
        with pytest.raises(TimeoutError):
            _recv_exact(first, 100, deadline)
        assert time.monotonic() < deadline + 0.3
    finally:
        stop.set()
        first.close()
        second.close()
        thread.join(2)


def test_periodic_pre_inference_disconnect_restores_startup_budget(tmp_path):
    from actionstream.inference_transport import InferenceTransportError

    with RpcInferenceServer(
        lambda obs, task: torch.ones(1),
        fault_profile=RpcFaultProfile(disconnect_before_infer_every_n=2),
    ) as server:
        client = TcpInferenceTransport(
            server.host,
            server.port,
            startup_inference_timeout_s=15,
            steady_inference_timeout_s=5,
            telemetry_jsonl_path=tmp_path / "rpc.jsonl",
        )
        client.reset()
        assert client.infer({}, "first", timeout_s=5).item() == 1
        with pytest.raises(InferenceTransportError):
            client.infer({}, "fault", timeout_s=5)
        assert client.infer({}, "recovery", timeout_s=5).item() == 1
        assert client.telemetry().transport_errors == 1
        assert client.telemetry().deadlines == 0
        assert (
            client.telemetry().latest_response_telemetry["connection_request_ordinal"]
            == 1
        )
        client.close()
    rows = [json.loads(x) for x in (tmp_path / "rpc.jsonl").read_text().splitlines()]
    assert [row["deadline_s"] for row in rows] == [15, 5, 15]


def test_worker_exception_does_not_replace_or_stop_executor():
    from actionstream.inference_transport import InferenceTransportError

    threads = []

    def worker(obs, task):
        threads.append(threading.get_ident())
        if task == "error":
            raise RuntimeError("synthetic worker failure")
        return torch.ones(1)

    with RpcInferenceServer(worker) as server:
        client = TcpInferenceTransport(server.host, server.port)
        client.reset()
        with pytest.raises(InferenceTransportError, match="synthetic worker failure"):
            client.infer({}, "error", timeout_s=1)
        assert client.infer({}, "recovered", timeout_s=1).item() == 1
        row = client.telemetry()
        assert row.server_errors == 1
        assert row.transport_errors == 0
        assert row.latest_response_telemetry["connection_request_ordinal"] == 2
        assert row.reconnects == 1
        client.close()
    assert len(set(threads)) == 1
