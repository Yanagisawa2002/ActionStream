"""Persistent ActionStream TCP inference client.

Connection ownership, request admission, reset poisoning, cancellation, and
client-side deadlines live here. Wire framing is isolated in rpc_protocol.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import torch

from actionstream.inference_transport import (
    InferenceCancelled,
    InferenceDeadlineExceeded,
    InferenceTransportError,
)
from actionstream.rpc_protocol import (
    _PROTOCOL_VERSION,
    _recv_frame,
    _send_frame,
)
from actionstream.rpc_telemetry import RpcTransportTelemetry


class RemoteResetError(InferenceTransportError):
    """Remote state is unconfirmed; an explicit successful reset is required."""


class TcpInferenceTransport:
    """Persistent TCP client; call reset() successfully before first inference."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float = 3.0,
        control_timeout_s: float = 3.0,
        reset_timeout_s: float = 20.0,
        startup_inference_timeout_s: float | None = None,
        steady_inference_timeout_s: float | None = None,
        telemetry_jsonl_path: str | Path | None = None,
    ) -> None:
        if not host:
            raise ValueError("host is required")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        for name, value in (
            ("connect_timeout_s", connect_timeout_s),
            ("control_timeout_s", control_timeout_s),
            ("reset_timeout_s", reset_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self._host = host
        self._port = port
        self._connect_timeout_s = float(connect_timeout_s)
        self._control_timeout_s = float(control_timeout_s)
        self._reset_timeout_s = float(reset_timeout_s)
        if (startup_inference_timeout_s is None) != (
            steady_inference_timeout_s is None
        ):
            raise ValueError(
                "startup and steady inference budgets must be set together"
            )
        for value in (startup_inference_timeout_s, steady_inference_timeout_s):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("inference budgets must be finite and positive")
        self._startup_timeout_s = startup_inference_timeout_s
        self._steady_timeout_s = steady_inference_timeout_s
        self._telemetry_path = (
            Path(telemetry_jsonl_path) if telemetry_jsonl_path else None
        )
        self._connection_requests = 0
        self._startup_deadlines = 0
        self._steady_deadlines = 0
        self._invalidated_requests = 0
        self._latest_response_telemetry: dict[str, Any] = {}
        self._request_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._resolution: Future | None = None
        self._generation = 0
        # A new socket/client is not evidence of clean persistent worker state.
        self._reset_required = True
        self._reset_requests = 0
        self._reset_successes = 0
        self._reset_failures = 0
        self._reset_timeouts = 0
        self._inflight = False
        self._requests_started = 0
        self._requests_completed = 0
        self._reconnects = 0
        self._cancellations = 0
        self._deadlines = 0
        self._transport_errors = 0
        self._server_errors = 0
        self._bytes_sent = 0
        self._bytes_received = 0
        self._latest_connect_latency_s: float | None = None
        self._latest_roundtrip_latency_s: float | None = None
        self._latest_server_inference_s: float | None = None

    @property
    def deadline_enforced(self) -> bool:
        return True

    @property
    def process_restarts(self) -> int:
        # Compatibility with ActionStreamTelemetry's existing transport counter.
        # For TCP, the analogous lifecycle event is a connection restart.
        with self._state_lock:
            return self._reconnects

    @property
    def cancellations(self) -> int:
        with self._state_lock:
            return self._cancellations

    @property
    def latest_startup_latency_s(self) -> float | None:
        with self._state_lock:
            return self._latest_connect_latency_s

    @property
    def latest_request_latency_s(self) -> float | None:
        with self._state_lock:
            return self._latest_roundtrip_latency_s

    def telemetry(self) -> RpcTransportTelemetry:
        with self._state_lock:
            return RpcTransportTelemetry(
                requests_started=self._requests_started,
                requests_completed=self._requests_completed,
                reconnects=self._reconnects,
                cancellations=self._cancellations,
                deadlines=self._deadlines,
                transport_errors=self._transport_errors,
                server_errors=self._server_errors,
                startup_deadlines=self._startup_deadlines,
                steady_deadlines=self._steady_deadlines,
                cancelled_invalidated_requests=self._invalidated_requests,
                connections_established=self._reconnects,
                reconnections=max(0, self._reconnects - 1),
                latest_response_telemetry=dict(self._latest_response_telemetry),
                reset_requests=self._reset_requests,
                reset_successes=self._reset_successes,
                reset_failures=self._reset_failures,
                reset_timeouts=self._reset_timeouts,
                reset_required=self._reset_required,
                bytes_sent=self._bytes_sent,
                bytes_received=self._bytes_received,
                latest_connect_latency_ms=(
                    None
                    if self._latest_connect_latency_s is None
                    else 1000 * self._latest_connect_latency_s
                ),
                latest_roundtrip_latency_ms=(
                    None
                    if self._latest_roundtrip_latency_s is None
                    else 1000 * self._latest_roundtrip_latency_s
                ),
                latest_server_inference_ms=(
                    None
                    if self._latest_server_inference_s is None
                    else 1000 * self._latest_server_inference_s
                ),
            )

    def _close_socket_locked(self) -> None:
        connection = self._socket
        self._socket = None
        if connection is None:
            return
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            connection.close()
        except OSError:
            pass

    def _ensure_socket(
        self, timeout_s: float, generation: int | None = None
    ) -> socket.socket:
        with self._state_lock:
            if generation is None:
                generation = self._generation
            if generation != self._generation:
                raise InferenceCancelled(
                    "TCP connect belongs to invalidated generation"
                )
            if self._socket is not None:
                return self._socket
        started = time.monotonic()
        connection = self._connect(started + min(self._connect_timeout_s, timeout_s))
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        elapsed = time.monotonic() - started
        with self._state_lock:
            if generation != self._generation:
                connection.close()
                raise InferenceCancelled("TCP connect completed after invalidation")
            if self._socket is not None:
                connection.close()
                return self._socket
            self._socket = connection
            self._connection_requests = 0
            self._reconnects += 1
            self._latest_connect_latency_s = elapsed
            return connection

    def _connect(self, deadline: float) -> socket.socket:
        # OS name resolution cannot be cancelled. Keep at most one outstanding
        # lookup per client, including across timeouts, without holding up reset.
        with self._state_lock:
            if self._resolution is None:
                self._resolution = Future()
                pending = self._resolution

                def resolve() -> None:
                    try:
                        pending.set_result(
                            socket.getaddrinfo(
                                self._host, self._port, type=socket.SOCK_STREAM
                            )
                        )
                    except BaseException as exc:
                        pending.set_exception(exc)

                threading.Thread(
                    target=resolve, name="ActionStreamRpcResolver", daemon=True
                ).start()
            pending = self._resolution
        try:
            addresses = pending.result(timeout=self._remaining(deadline))
        finally:
            with self._state_lock:
                if pending.done() and self._resolution is pending:
                    self._resolution = None
        last_error: OSError = OSError("RPC hostname resolved to no stream addresses")
        for family, kind, protocol, _, address in addresses:
            timeout = self._remaining(deadline)
            connection = socket.socket(family, kind, protocol)
            try:
                connection.settimeout(timeout)
                connection.connect(address)
                self._remaining(deadline)
                return connection
            except OSError as exc:
                last_error = exc
                connection.close()
            except BaseException:
                connection.close()
                raise
        raise last_error

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise InferenceDeadlineExceeded("TCP inference deadline exceeded")
        return remaining

    def infer(
        self,
        observation: Mapping[str, Any],
        task: str,
        *,
        timeout_s: float,
    ) -> torch.Tensor:
        return self._infer_request(observation, task, timeout_s=timeout_s)

    def next_inference_timeout_s(self, fallback: float) -> float:
        """Budget preview for engine telemetry; request admission selects atomically."""
        with self._state_lock:
            startup = self._socket is None or self._connection_requests == 0
            budget = self._startup_timeout_s if startup else self._steady_timeout_s
            return fallback if budget is None else budget

    def infer_cancellable(
        self,
        observation: Mapping[str, Any],
        task: str,
        *,
        timeout_s: float,
        cancellation_event: threading.Event,
    ) -> torch.Tensor:
        return self._infer_request(
            observation,
            task,
            timeout_s=timeout_s,
            cancellation_event=cancellation_event,
        )

    def _infer_request(
        self,
        observation: Mapping[str, Any],
        task: str,
        *,
        timeout_s: float,
        cancellation_event: threading.Event | None = None,
    ) -> torch.Tensor:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        with self._state_lock:
            if self._reset_required:
                raise RemoteResetError(
                    "TCP inference requires a successful remote reset"
                )
            generation = self._generation
        request_id = uuid.uuid4().hex
        with self._request_lock:
            with self._state_lock:
                if self._reset_required:
                    raise RemoteResetError(
                        "TCP inference requires a successful remote reset"
                    )
                if generation != self._generation:
                    raise InferenceCancelled("TCP request queued before invalidation")
                self._inflight = True
                self._requests_started += 1
                startup = self._socket is None or self._connection_requests == 0
                budget = self._startup_timeout_s if startup else self._steady_timeout_s
                timeout_s = timeout_s if budget is None else budget
            deadline = time.monotonic() + timeout_s
            started = time.monotonic()
            started_utc_ns = time.time_ns()
            outcome = "cancelled"
            response_telemetry: dict[str, Any] = {}
            remote_error = False
            try:
                if cancellation_event is not None and cancellation_event.is_set():
                    raise InferenceCancelled("TCP request cancelled before connect")
                connection = self._ensure_socket(self._remaining(deadline), generation)
                with self._state_lock:
                    self._connection_requests += 1
                connection.settimeout(self._remaining(deadline))
                sent = _send_frame(
                    connection,
                    {
                        "version": _PROTOCOL_VERSION,
                        "kind": "infer",
                        "request_id": request_id,
                        "task": task,
                        "observation": dict(observation),
                    },
                    deadline,
                )
                with self._state_lock:
                    self._bytes_sent += sent
                if cancellation_event is not None and cancellation_event.is_set():
                    raise InferenceCancelled("TCP request cancelled after send")
                connection.settimeout(self._remaining(deadline))
                response, received = _recv_frame(connection, deadline)
                self._remaining(deadline)
                response_telemetry = {
                    key: response[key]
                    for key in (
                        "server_inference_s",
                        "server_queue_wait_s",
                        "server_compute_s",
                        "server_service_s",
                        "connection_id",
                        "connection_request_ordinal",
                        "global_request_ordinal",
                    )
                    if key in response
                }
                with self._state_lock:
                    self._bytes_received += received
                    if generation != self._generation:
                        raise InferenceCancelled("TCP response generation invalidated")
                if response.get("version") != _PROTOCOL_VERSION:
                    raise InferenceTransportError("RPC protocol version mismatch")
                if response.get("request_id") != request_id:
                    raise InferenceTransportError("RPC response id mismatch")
                kind = response.get("kind")
                if kind == "error":
                    remote_error = True
                    with self._state_lock:
                        self._server_errors += 1
                    raise InferenceTransportError(
                        f"Remote inference failed: {response.get('error_type')}: "
                        f"{response.get('error')}"
                    )
                if kind != "ok":
                    raise InferenceTransportError(
                        f"Unknown RPC response kind: {kind!r}"
                    )
                actions = response.get("actions")
                if not isinstance(actions, torch.Tensor):
                    raise InferenceTransportError(
                        f"Remote actions must be torch.Tensor, got {type(actions)!r}"
                    )
                with self._state_lock:
                    if generation != self._generation:
                        raise InferenceCancelled(
                            "TCP response belongs to invalidated generation"
                        )
                    if cancellation_event is not None and cancellation_event.is_set():
                        raise InferenceCancelled(
                            "TCP response arrived after cancellation"
                        )
                    self._requests_completed += 1
                    self._latest_response_telemetry = response_telemetry
                    self._latest_roundtrip_latency_s = time.monotonic() - started
                    server_s = response.get("server_inference_s")
                    self._latest_server_inference_s = (
                        float(server_s) if server_s is not None else None
                    )
                outcome = "success"
                return actions
            except (socket.timeout, TimeoutError) as exc:
                with self._state_lock:
                    if generation != self._generation:
                        raise InferenceCancelled("TCP request invalidated") from exc
                    self._deadlines += 1
                    if startup:
                        self._startup_deadlines += 1
                    else:
                        self._steady_deadlines += 1
                    self._generation += 1
                    self._close_socket_locked()
                outcome = "deadline"
                raise InferenceDeadlineExceeded(
                    f"TCP inference exceeded {timeout_s:.3f}s deadline"
                ) from exc
            except InferenceCancelled:
                with self._state_lock:
                    self._close_socket_locked()
                raise
            except InferenceTransportError:
                outcome = "server_error" if remote_error else "transport_error"
                with self._state_lock:
                    if not remote_error:
                        self._transport_errors += 1
                        self._generation += 1
                        self._close_socket_locked()
                raise
            except (EOFError, OSError, ValueError, json.JSONDecodeError) as exc:
                with self._state_lock:
                    invalidated = generation != self._generation
                    self._close_socket_locked()
                    if not invalidated:
                        self._transport_errors += 1
                        self._generation += 1
                if invalidated:
                    raise InferenceCancelled("TCP request invalidated") from exc
                outcome = "transport_error"
                raise InferenceTransportError("TCP inference transport failed") from exc
            finally:
                with self._state_lock:
                    self._inflight = False
                    if outcome in ("cancelled", "deadline"):
                        self._invalidated_requests += 1
                if self._telemetry_path is not None:
                    self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                    with self._telemetry_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "request_id": request_id,
                                    "started_utc_unix_ns": started_utc_ns,
                                    "finished_utc_unix_ns": time.time_ns(),
                                    "generation": generation,
                                    "deadline_class": "startup"
                                    if startup
                                    else "steady",
                                    "deadline_s": timeout_s,
                                    "outcome": outcome,
                                    "roundtrip_s": time.monotonic() - started,
                                    **response_telemetry,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )

    def _control(
        self, kind: str, *, deadline: float | None = None, generation: int | None = None
    ) -> None:
        if deadline is None:
            deadline = time.monotonic() + self._control_timeout_s
        if generation is None:
            with self._state_lock:
                generation = self._generation
        if not self._request_lock.acquire(timeout=self._remaining(deadline)):
            raise InferenceDeadlineExceeded(
                f"TCP {kind} deadline waiting for client I/O"
            )
        try:
            connection = self._ensure_socket(self._remaining(deadline), generation)
            request_id = uuid.uuid4().hex
            connection.settimeout(self._remaining(deadline))
            sent = _send_frame(
                connection,
                {
                    "version": _PROTOCOL_VERSION,
                    "kind": kind,
                    "request_id": request_id,
                },
                deadline,
            )
            with self._state_lock:
                self._bytes_sent += sent
            connection.settimeout(self._remaining(deadline))
            response, received = _recv_frame(connection, deadline)
            self._remaining(deadline)
            with self._state_lock:
                self._bytes_received += received
                if generation != self._generation:
                    raise InferenceCancelled(f"TCP {kind} acknowledgement invalidated")
            if (
                response.get("version") != _PROTOCOL_VERSION
                or response.get("request_id") != request_id
            ):
                raise InferenceTransportError(f"Invalid RPC {kind} acknowledgement")
            if response.get("kind") == "error":
                raise InferenceTransportError(
                    f"Remote {kind} failed: {response.get('error_type')}: {response.get('error')}"
                )
            if response.get("kind") != "ack":
                raise InferenceTransportError(f"Invalid RPC {kind} acknowledgement")
        finally:
            self._request_lock.release()

    def cancel(self) -> bool:
        with self._state_lock:
            was_inflight = self._inflight
            self._generation += 1
            if was_inflight:
                self._cancellations += 1
            self._close_socket_locked()
        return was_inflight

    def reset(self) -> bool:
        """Invalidate immediately, then wait for executor reset completion and ACK.

        The absolute reset budget includes local I/O-lock wait, connect, server
        queue/callback and ACK. Timeout does not preempt remote compute or reset.
        Failure poisons inference until a later explicit reset succeeds.
        """
        deadline = time.monotonic() + self._reset_timeout_s
        with self._state_lock:
            self._reset_required = True
            self._reset_requests += 1
            cancelled = self._inflight
            self._generation += 1
            generation = self._generation
            if cancelled:
                self._cancellations += 1
            self._close_socket_locked()
        try:
            self._control("reset", deadline=deadline, generation=generation)
            with self._state_lock:
                if generation != self._generation:
                    raise InferenceCancelled("TCP reset superseded by cancellation")
                self._reset_required = False
                self._reset_successes += 1
            return cancelled
        except BaseException as exc:
            with self._state_lock:
                self._reset_failures += 1
                if isinstance(exc, TimeoutError):
                    self._reset_timeouts += 1
                # An older reset must not invalidate a newer acknowledged reset.
                if generation == self._generation:
                    self._reset_required = True
                    self._generation += 1
                    self._close_socket_locked()
            if not isinstance(exc, Exception):
                raise
            raise RemoteResetError(
                f"Remote reset was not acknowledged; inference remains blocked: {exc}"
            ) from exc

    def close(self) -> None:
        with self._state_lock:
            self._generation += 1
            self._close_socket_locked()

