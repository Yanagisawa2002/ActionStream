"""CPU/socket regressions for acknowledged reset and explicit listener exposure."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time

import pytest
import torch

from actionstream import rpc_server
from actionstream.inference_transport import InferenceCancelled
from actionstream.rpc_transport import (
    RemoteResetError,
    RpcInferenceServer,
    TcpInferenceTransport,
    _recv_frame,
    _send_frame,
)


def launch(call):
    result = []

    def run():
        try:
            result.append(call())
        except BaseException as exc:
            result.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result


def finish(pending):
    thread, result = pending
    thread.join(3)
    assert not thread.is_alive()
    assert len(result) == 1
    return result[0]


def wait_until(predicate):
    limit = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < limit
        time.sleep(0.001)


class StatefulWorker:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.reset_entered = threading.Event()
        self.reset_release = threading.Event()
        self.reset_release.set()
        self.fail_reset = False
        self.value = 0
        self.active = 0
        self.maximum_active = 0
        self.events = []
        self.threads = []

    def __call__(self, obs, task):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.threads.append(threading.get_ident())
        try:
            if task == "old":
                self.events.append("infer_enter")
                self.entered.set()
                assert self.release.wait(5)
                self.events.append("infer_exit")
            else:
                self.events.append("next_infer")
            self.value += 1
            return torch.tensor([self.value])
        finally:
            self.active -= 1

    def reset(self):
        assert self.active == 0
        self.threads.append(threading.get_ident())
        self.reset_entered.set()
        assert self.reset_release.wait(5)
        if self.fail_reset:
            raise RuntimeError("state reset failed")
        self.value = 0
        self.events.append("reset")


def test_cli_default_is_loopback():
    args = rpc_server.build_parser().parse_args(["--factory", "fake:worker"])
    assert args.host == "127.0.0.1"
    assert args.allow_unauthenticated_remote is False


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.1", "example.invalid"])
def test_remote_listener_requires_ack_before_factory_or_socket(
    monkeypatch, capsys, host
):
    monkeypatch.setattr(
        sys, "argv", ["rpc", "--factory", "fake:worker", "--host", host]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("must reject bind before constructing worker or listener")

    monkeypatch.setattr(rpc_server, "load_worker", forbidden)
    monkeypatch.setattr(rpc_server, "RpcInferenceServer", forbidden)
    with pytest.raises(SystemExit) as failure:
        rpc_server.main()
    assert failure.value.code == 2
    assert "--allow-unauthenticated-remote" in capsys.readouterr().err


@pytest.mark.parametrize("remote", [None, "0.0.0.0", "::"])
@pytest.mark.parametrize("callback", [False, True])
def test_cli_receipt_states_exposure_and_reset_capability(
    monkeypatch, tmp_path, remote, callback
):
    path = tmp_path / "ready.json"
    argv = ["rpc", "--factory", "fake:worker", "--port", "0", "--ready-file", str(path)]
    if remote:
        argv += ["--host", remote, "--allow-unauthenticated-remote"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(rpc_server.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        rpc_server,
        "load_worker",
        lambda spec: (
            lambda obs, task: torch.ones(1),
            (lambda: None) if callback else None,
        ),
    )
    monkeypatch.setattr(RpcInferenceServer, "serve_forever", lambda self: None)
    assert rpc_server.main() == 0
    receipt = json.loads(path.read_text())
    assert receipt["host"] == (remote or "127.0.0.1")
    assert receipt["authentication"] == "none"
    assert receipt["listener_exposure"] == (
        "unauthenticated_remote" if remote else "loopback"
    )
    assert receipt["reset_capability"] == ("callback" if callback else "stateless_noop")


def test_new_client_requires_confirmed_reset_and_idle_reset_allows_inference():
    worker = StatefulWorker()
    worker.value = 41  # A new client must not infer against inherited policy state.
    with RpcInferenceServer(worker, reset=worker.reset) as server:
        client = TcpInferenceTransport(server.host, server.port)
        try:
            with pytest.raises(RemoteResetError):
                client.infer({}, "next", timeout_s=1)
            assert server.telemetry()["inference_calls"] == 0
            assert client.reset() is False
            assert client.infer({}, "next", timeout_s=1).item() == 1
            assert not client.telemetry().reset_required
            assert client.telemetry().reset_successes == 1
        finally:
            client.close()


def test_callback_error_is_wire_error_and_client_fails_closed_then_recovers(tmp_path):
    worker = StatefulWorker()
    worker.fail_reset = True
    with RpcInferenceServer(
        worker, reset=worker.reset, telemetry_jsonl_path=tmp_path / "server.jsonl"
    ) as server:
        with socket.create_connection((server.host, server.port)) as raw:
            raw.settimeout(2)
            _send_frame(raw, dict(version=1, kind="reset", request_id="reset-error"))
            response, _ = _recv_frame(raw)
            assert response["kind"] == "error"
            assert response["error_type"] == "RuntimeError"
        client = TcpInferenceTransport(server.host, server.port)
        try:
            with pytest.raises(RemoteResetError, match="state reset failed"):
                client.reset()
            assert client.telemetry().reset_failures == 1
            assert client.telemetry().reset_successes == 0
            with pytest.raises(RemoteResetError):
                client.infer({}, "must-not-run", timeout_s=1)
            worker.fail_reset = False
            client.reset()
            assert client.infer({}, "next", timeout_s=1).item() == 1
            assert not client.telemetry().reset_required
            assert server.telemetry()["reset_failures"] == 2
            assert server.telemetry()["reset_successes"] == 1
        finally:
            client.close()
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "server.jsonl").read_text().splitlines()
    ]
    assert events.count("reset_failed") == 2
    assert events.count("reset_succeeded") == 1


def test_active_reset_orders_stateful_execution_and_gates_inference_until_ack():
    worker = StatefulWorker()
    with RpcInferenceServer(worker, reset=worker.reset) as server:
        client = TcpInferenceTransport(
            server.host, server.port, control_timeout_s=0.01, reset_timeout_s=2
        )
        client.reset()
        worker.events.clear()
        worker.reset_entered.clear()
        worker.reset_release.clear()
        old = launch(lambda: client.infer({}, "old", timeout_s=3))
        try:
            assert worker.entered.wait(2)
            resetting = launch(client.reset)
            wait_until(lambda: server._jobs.qsize() == 1)
            assert isinstance(finish(old), InferenceCancelled)
            assert resetting[0].is_alive()
            with pytest.raises(RemoteResetError):
                client.infer({}, "too-early", timeout_s=1)
            worker.release.set()
            assert worker.reset_entered.wait(2)
            # Callback entered but has not completed; an ACK cannot exist yet.
            with pytest.raises(RemoteResetError):
                client.infer({}, "still-too-early", timeout_s=1)
            worker.reset_release.set()
            assert finish(resetting) is True
            assert client.infer({}, "next", timeout_s=1).item() == 1
            assert worker.events == ["infer_enter", "infer_exit", "reset", "next_infer"]
            assert worker.maximum_active == 1 and len(set(worker.threads)) == 1
            assert server.telemetry()["executor_starts"] == 1
            assert server.telemetry()["invalidated_results"] == 1
        finally:
            worker.release.set()
            worker.reset_release.set()
            client.close()
    assert not server._executor.is_alive()


def test_active_reset_timeout_reconnect_and_recreation_cannot_bypass_reset():
    worker = StatefulWorker()
    with RpcInferenceServer(worker, reset=worker.reset) as server:
        client = TcpInferenceTransport(server.host, server.port, reset_timeout_s=0.15)
        client.reset()
        old = launch(lambda: client.infer({}, "old", timeout_s=3))
        try:
            assert worker.entered.wait(2)
            with pytest.raises(RemoteResetError) as failed:
                client.reset()
            assert isinstance(failed.value.__cause__, TimeoutError)
            assert worker.active == 1  # No compute preemption was performed.
            assert isinstance(finish(old), InferenceCancelled)
            assert client.telemetry().reset_timeouts == 1
            assert client.telemetry().reset_required
            # Deliberately reopen only the socket: this must not clear poison.
            connections = client.telemetry().connections_established
            with client._request_lock:
                client._ensure_socket(1)
            assert client.telemetry().connections_established == connections + 1
            with pytest.raises(RemoteResetError):
                client.infer({}, "not-reset", timeout_s=1)
            replacement = TcpInferenceTransport(server.host, server.port)
            try:
                with pytest.raises(RemoteResetError):
                    replacement.infer({}, "replacement-not-reset", timeout_s=1)
            finally:
                replacement.close()
            worker.release.set()
            client.reset()
            assert not client.telemetry().reset_required
            assert client.infer({}, "next", timeout_s=1).item() == 1
            assert worker.maximum_active == 1
        finally:
            worker.release.set()
            client.close()


@pytest.mark.parametrize("failure", ["eof", "timeout", "bad_ack", "cancel"])
def test_failed_or_invalid_idle_reset_never_allows_inference(monkeypatch, failure):
    with RpcInferenceServer(lambda obs, task: torch.ones(1)) as server:
        client = TcpInferenceTransport(server.host, server.port, reset_timeout_s=0.1)
        client.reset()

        def fail_control(*args, **kwargs):
            if failure == "eof":
                raise EOFError("closed")
            if failure == "timeout":
                raise socket.timeout("late")
            if failure == "cancel":
                client.cancel()
                raise InferenceCancelled("cancelled")
            raise ValueError("malformed ack")

        monkeypatch.setattr(client, "_control", fail_control)
        with pytest.raises(RemoteResetError):
            client.reset()
        assert client.telemetry().reset_required
        with pytest.raises(RemoteResetError):
            client.infer({}, "unsafe", timeout_s=1)
        client.close()


def test_reset_budget_includes_waiting_for_client_io_lock():
    with RpcInferenceServer(lambda obs, task: torch.ones(1)) as server:
        client = TcpInferenceTransport(server.host, server.port, reset_timeout_s=0.05)
        client.reset()
        with client._request_lock:
            started = time.monotonic()
            with pytest.raises(RemoteResetError):
                client.reset()
            assert time.monotonic() - started < 1
        assert client.telemetry().reset_timeouts == 1
        client.reset()
        assert client.infer({}, "next", timeout_s=1).item() == 1
        client.close()


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_reset_budget_must_be_finite_positive(budget):
    with pytest.raises(ValueError, match="reset_timeout_s"):
        TcpInferenceTransport("127.0.0.1", 50051, reset_timeout_s=budget)


@pytest.mark.parametrize(
    "reply", ["eof", "wrong_id", "wrong_version", "not_ack", "error", "timeout"]
)
def test_reset_requires_valid_wire_ack(reply):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2)
    release = threading.Event()

    def peer():
        connection, _ = listener.accept()
        with connection:
            request, _ = _recv_frame(connection)
            if reply == "eof":
                return
            if reply == "timeout":
                release.wait(2)
                return
            _send_frame(
                connection,
                dict(
                    version=9 if reply == "wrong_version" else 1,
                    request_id="wrong"
                    if reply == "wrong_id"
                    else request["request_id"],
                    kind="error"
                    if reply == "error"
                    else "ok"
                    if reply == "not_ack"
                    else "ack",
                    error_type="ResetFailure",
                    error="state not cleared",
                ),
            )

    pending = launch(peer)
    client = TcpInferenceTransport(*listener.getsockname(), reset_timeout_s=0.15)
    try:
        with pytest.raises(RemoteResetError):
            client.reset()
        assert client.telemetry().reset_required
        with pytest.raises(RemoteResetError):
            client.infer({}, "unsafe", timeout_s=1)
    finally:
        release.set()
        client.close()
        listener.close()
        assert finish(pending) is None


def test_cancel_after_ack_before_reset_commit_cannot_clear_required_state(monkeypatch):
    with RpcInferenceServer(lambda obs, task: torch.ones(1)) as server:
        client = TcpInferenceTransport(server.host, server.port)
        ack = threading.Event()
        release = threading.Event()
        control = client._control

        def held_ack(*args, **kwargs):
            control(*args, **kwargs)
            ack.set()
            assert release.wait(3)

        monkeypatch.setattr(client, "_control", held_ack)
        pending = launch(client.reset)
        try:
            assert ack.wait(2)
            client.cancel()
            release.set()
            assert isinstance(finish(pending), RemoteResetError)
            assert client.telemetry().reset_required
            with pytest.raises(RemoteResetError):
                client.infer({}, "unsafe", timeout_s=1)
            monkeypatch.setattr(client, "_control", control)
            client.reset()
            assert client.infer({}, "safe", timeout_s=1).item() == 1
        finally:
            release.set()
            client.close()


def test_engine_reset_propagates_remote_failure_and_blocks_new_episode():
    from types import SimpleNamespace
    from actionstream.lerobot_inference import (
        ActionStreamInferenceConfig,
        ActionStreamInferenceEngine,
    )

    worker = StatefulWorker()
    stub = SimpleNamespace(reset=lambda: None)
    with RpcInferenceServer(worker, reset=worker.reset) as server:
        client = TcpInferenceTransport(server.host, server.port)
        engine = ActionStreamInferenceEngine(
            policy=stub,
            preprocessor=stub,
            postprocessor=stub,
            hw_features={},
            task="cpu",
            device="cpu",
            robot_type="mock",
            config=ActionStreamInferenceConfig(),
            infer_chunk=lambda obs, task: None,
            reset_provider=lambda: None,
            transport=client,
        )
        engine.start()
        try:
            engine.reset()
            worker.fail_reset = True
            with pytest.raises(RemoteResetError):
                engine.reset()
            assert engine.telemetry.failed
            assert client.telemetry().reset_required
            engine.resume()
            engine.notify_observation({})
            assert engine.get_action(None) is None
            assert server.telemetry()["inference_calls"] == 0
        finally:
            engine.stop()
