"""Length-prefixed TCP inference transport and deterministic fault-injection server.

The client implements ActionStream's ``InferenceTransport`` protocol without
requiring changes to the core inference engine. The wire protocol is deliberately
small and versioned: UTF-8 JSON frames carry recursively encoded tensors/arrays.

This transport provides a real socket boundary and client-side hard I/O deadlines.
Closing the socket invalidates late responses, but it cannot preempt a CUDA kernel
already running on a remote server. Deployments that require server-side compute
preemption still need process isolation or another cooperative cancellation layer.
"""

from __future__ import annotations

import base64
import importlib
import json
import math
import queue
import random
import select
import socket
import struct
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from actionstream.inference_transport import (
    InferenceCancelled,
    InferenceDeadlineExceeded,
    InferenceTransportError,
)

_PROTOCOL_VERSION = 1
_HEADER = struct.Struct("!I")
_MAX_FRAME_BYTES = 256 * 1024 * 1024
_TENSOR_TAG = "__actionstream_tensor__"
_NDARRAY_TAG = "__actionstream_ndarray__"
_BYTES_TAG = "__actionstream_bytes__"

_TORCH_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


def _encode_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            _TENSOR_TAG: True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            _NDARRAY_TAG: True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, bytes | bytearray | memoryview):
        return {
            _BYTES_TAG: True,
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("RPC mappings require string keys")
            result[key] = _encode_value(child)
        return result
    if isinstance(value, tuple):
        return {"__actionstream_tuple__": [_encode_value(child) for child in value]}
    if isinstance(value, list):
        return [_encode_value(child) for child in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported RPC value type: {type(value)!r}")


def _decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_value(child) for child in value]
    if not isinstance(value, dict):
        return value
    if value.get(_TENSOR_TAG) is True:
        dtype_name = value["dtype"]
        try:
            dtype = _TORCH_DTYPES[dtype_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported torch dtype on wire: {dtype_name}") from exc
        raw = base64.b64decode(value["data"], validate=True)
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
        return tensor.reshape(tuple(int(item) for item in value["shape"]))
    if value.get(_NDARRAY_TAG) is True:
        dtype = np.dtype(value["dtype"])
        raw = base64.b64decode(value["data"], validate=True)
        array = np.frombuffer(raw, dtype=dtype).copy()
        return array.reshape(tuple(int(item) for item in value["shape"]))
    if value.get(_BYTES_TAG) is True:
        return base64.b64decode(value["data"], validate=True)
    if "__actionstream_tuple__" in value:
        return tuple(_decode_value(child) for child in value["__actionstream_tuple__"])
    return {key: _decode_value(child) for key, child in value.items()}


def _encode_frame(payload: Mapping[str, Any]) -> bytes:
    body = json.dumps(
        _encode_value(dict(payload)),
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(body) > _MAX_FRAME_BYTES:
        raise ValueError(f"RPC frame exceeds {_MAX_FRAME_BYTES} bytes")
    return _HEADER.pack(len(body)) + body


def _recv_exact(
    connection: socket.socket, size: int, deadline: float | None = None
) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        if deadline is not None:
            connection.settimeout(TcpInferenceTransport._remaining(deadline))
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError("RPC peer closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_frame(
    connection: socket.socket, deadline: float | None = None
) -> tuple[dict[str, Any], int]:
    header = _recv_exact(connection, _HEADER.size, deadline)
    (length,) = _HEADER.unpack(header)
    if length <= 0 or length > _MAX_FRAME_BYTES:
        raise ValueError(f"Invalid RPC frame length: {length}")
    body = _recv_exact(connection, length, deadline)
    decoded = json.loads(body.decode("utf-8"))
    value = _decode_value(decoded)
    if not isinstance(value, dict):
        raise ValueError("RPC top-level frame must be a mapping")
    return value, _HEADER.size + length


def _send_frame(
    connection: socket.socket, payload: Mapping[str, Any], deadline: float | None = None
) -> int:
    frame = _encode_frame(payload)
    if deadline is not None:
        connection.settimeout(TcpInferenceTransport._remaining(deadline))
    connection.sendall(frame)
    return len(frame)


@dataclass(frozen=True)
class RpcFaultProfile:
    """Deterministic request-indexed server faults for transport validation."""

    response_delay_s: float = 0.0
    response_jitter_s: float = 0.0
    stall_every_n: int = 0
    stall_s: float = 0.0
    disconnect_before_infer_every_n: int = 0
    drop_response_every_n: int = 0
    server_error_every_n: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("response_delay_s", "response_jitter_s", "stall_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "stall_every_n",
            "disconnect_before_infer_every_n",
            "drop_response_every_n",
            "server_error_every_n",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @staticmethod
    def _matches(ordinal: int, every_n: int) -> bool:
        return every_n > 0 and ordinal % every_n == 0

    def delay_for(self, ordinal: int) -> float:
        rng = random.Random(self.seed + ordinal * 1_000_003)
        jitter = (
            rng.uniform(-self.response_jitter_s, self.response_jitter_s)
            if self.response_jitter_s
            else 0.0
        )
        return max(0.0, self.response_delay_s + jitter)


@dataclass(frozen=True)
class RpcTransportTelemetry:
    requests_started: int
    requests_completed: int
    reconnects: int
    cancellations: int
    deadlines: int
    transport_errors: int
    server_errors: int
    bytes_sent: int
    bytes_received: int
    latest_connect_latency_ms: float | None
    latest_roundtrip_latency_ms: float | None
    latest_server_inference_ms: float | None
    startup_deadlines: int
    steady_deadlines: int
    cancelled_invalidated_requests: int
    connections_established: int
    reconnections: int
    latest_response_telemetry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TcpInferenceTransport:
    """Persistent TCP client implementing ActionStream's transport contract."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float = 3.0,
        control_timeout_s: float = 3.0,
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
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self._host = host
        self._port = port
        self._connect_timeout_s = float(connect_timeout_s)
        self._control_timeout_s = float(control_timeout_s)
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
        self._generation = 0
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
        connection = socket.create_connection(
            (self._host, self._port),
            timeout=min(self._connect_timeout_s, timeout_s),
        )
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
        request_id = uuid.uuid4().hex
        with self._request_lock:
            with self._state_lock:
                generation = self._generation
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

    def _control(self, kind: str) -> None:
        deadline = time.monotonic() + self._control_timeout_s
        with self._request_lock:
            connection = self._ensure_socket(self._remaining(deadline))
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
            if (
                response.get("version") != _PROTOCOL_VERSION
                or response.get("request_id") != request_id
                or response.get("kind") != "ack"
            ):
                raise InferenceTransportError(f"Invalid RPC {kind} acknowledgement")

    def cancel(self) -> bool:
        with self._state_lock:
            was_inflight = self._inflight
            self._generation += 1
            if was_inflight:
                self._cancellations += 1
            self._close_socket_locked()
        return was_inflight

    def reset(self) -> bool:
        # If an inference is active, invalidate it immediately. The remote worker
        # may still finish that CUDA call; its response cannot cross this generation.
        cancelled = self.cancel()
        if not cancelled:
            try:
                self._control("reset")
            except (
                InferenceTransportError,
                InferenceDeadlineExceeded,
                InferenceCancelled,
                OSError,
                EOFError,
            ):
                with self._state_lock:
                    self._close_socket_locked()
                # A fresh connection is still a clean client-side generation.
        return cancelled

    def close(self) -> None:
        with self._state_lock:
            self._generation += 1
            self._close_socket_locked()


@dataclass
class _ExecutorRequest:
    request: dict[str, Any]
    connection: socket.socket
    metadata: dict[str, Any]
    enqueued_at: float = field(default_factory=time.perf_counter)
    obsolete: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None


class RpcInferenceServer:
    """Connection handlers route work to one persistent, serialized executor.

    Only the executor calls infer/reset. Closing a socket invalidates its pending
    work, but cannot interrupt an already running worker call. close() waits for
    that call to return before joining the executor; it has no CUDA preemption.
    """

    def __init__(
        self,
        infer: Callable[[Mapping[str, Any], str], torch.Tensor],
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        reset: Callable[[], None] | None = None,
        fault_profile: RpcFaultProfile | None = None,
        backlog: int = 16,
        telemetry_jsonl_path: str | Path | None = None,
    ) -> None:
        if not callable(infer):
            raise TypeError("infer must be callable")
        self._infer = infer
        self._reset = reset
        self._faults = fault_profile or RpcFaultProfile()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(backlog)
        self._listener.settimeout(0.1)
        self.host, self.port = self._listener.getsockname()[:2]
        self._stop = threading.Event()
        self._ordinal_lock = threading.Lock()
        self._next_ordinal = 1
        self._jobs: queue.Queue[_ExecutorRequest | None] = queue.Queue()
        self._threads: set[threading.Thread] = set()
        self._connections: set[socket.socket] = set()
        self._threads_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self._telemetry_path = (
            Path(telemetry_jsonl_path) if telemetry_jsonl_path else None
        )
        self._counters = dict(
            executor_starts=1,
            inference_calls=0,
            reset_calls=0,
            dropped_obsolete_before_compute=0,
            invalidated_results=0,
        )
        self._executor = threading.Thread(
            target=self._executor_loop, name="ActionStreamRpcExecutor", daemon=True
        )
        self._executor.start()

    def telemetry(self) -> dict[str, int]:
        with self._telemetry_lock:
            return dict(self._counters)

    def _count(self, name: str) -> None:
        with self._telemetry_lock:
            self._counters[name] += 1

    def _record(self, event: str, job: _ExecutorRequest, **fields: Any) -> None:
        if self._telemetry_path is not None:
            with self._telemetry_lock:
                self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                with self._telemetry_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "event": event,
                                "utc_unix_ns": time.time_ns(),
                                "request_id": job.request["request_id"],
                                "kind": job.request["kind"],
                                **job.metadata,
                                **fields,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

    @staticmethod
    def _disconnected(connection: socket.socket) -> bool:
        # The protocol allows one outstanding request per connection. EOF, reset,
        # or unexpected pipelining all make this request path unusable. No read
        # consumes a future response, and executor threads never perform writes.
        try:
            readable, _, _ = select.select([connection], [], [], 0)
            return bool(readable)
        except (OSError, ValueError):
            return True

    def _obsolete(self, job: _ExecutorRequest) -> bool:
        return (
            self._stop.is_set()
            or job.obsolete.is_set()
            or self._disconnected(job.connection)
        )

    def _executor_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                if self._obsolete(job):
                    self._count("dropped_obsolete_before_compute")
                    self._record("dropped_obsolete_before_compute", job)
                    continue
                started = time.perf_counter()
                compute_s = 0.0
                response = {
                    "version": _PROTOCOL_VERSION,
                    "request_id": job.request["request_id"],
                    **job.metadata,
                }
                try:
                    if job.request["kind"] == "reset":
                        if self._reset is not None:
                            self._count("reset_calls")
                            self._reset()
                        response["kind"] = "ack"
                        inference_finished = time.perf_counter()
                    else:
                        self._count("inference_calls")
                        compute_started = time.perf_counter()
                        try:
                            actions = self._infer(
                                job.request["observation"], job.request["task"]
                            )
                        finally:
                            compute_s = time.perf_counter() - compute_started
                        if not isinstance(actions, torch.Tensor):
                            raise TypeError("RPC worker must return torch.Tensor")
                        inference_finished = time.perf_counter()
                        response.update(kind="ok", actions=actions.detach().cpu())
                except BaseException as exc:
                    inference_finished = time.perf_counter()
                    response.update(
                        kind="error", error_type=type(exc).__name__, error=str(exc)
                    )
                finished = time.perf_counter()
                timing = dict(
                    server_queue_wait_s=started - job.enqueued_at,
                    server_compute_s=compute_s,
                    server_service_s=finished - started,
                    # Legacy composite wait + worker/validation interval; CPU
                    # materialization and response delivery remain outside it.
                    server_inference_s=inference_finished - job.enqueued_at,
                )
                response.update(timing)
                if self._obsolete(job):
                    self._count("invalidated_results")
                    self._record("invalidated_result", job, **timing)
                else:
                    job.response = response
                    self._record(
                        "execution_finished",
                        job,
                        **timing,
                        result_kind=response["kind"],
                    )
            finally:
                job.done.set()

    def _execute(self, connection, request, metadata):
        job = _ExecutorRequest(request, connection, metadata)
        self._record("enqueued", job)
        job.enqueued_at = time.perf_counter()
        self._jobs.put(job)
        try:
            while not job.done.wait(0.01):
                if self._obsolete(job):
                    return None
            return None if self._obsolete(job) else job.response
        finally:
            job.obsolete.set()

    def _client_loop(self, connection: socket.socket) -> None:
        connection_id = uuid.uuid4().hex
        connection_ordinal = 0
        try:
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while not self._stop.is_set():
                    request, _ = _recv_frame(connection)
                    if request.get("version") != _PROTOCOL_VERSION:
                        return
                    request_id = request.get("request_id")
                    if not isinstance(request_id, str) or not request_id:
                        return
                    kind = request.get("kind")
                    metadata = {"connection_id": connection_id}
                    if kind == "close":
                        _send_frame(
                            connection,
                            dict(
                                version=_PROTOCOL_VERSION,
                                kind="ack",
                                request_id=request_id,
                            ),
                        )
                        return
                    if kind == "reset":
                        response = self._execute(connection, request, metadata)
                        if response is None:
                            return
                        _send_frame(connection, response)
                        continue
                    if (
                        kind != "infer"
                        or not isinstance(request.get("observation"), dict)
                        or not isinstance(request.get("task"), str)
                    ):
                        return
                    connection_ordinal += 1
                    with self._ordinal_lock:
                        ordinal = self._next_ordinal
                        self._next_ordinal += 1
                    metadata.update(
                        connection_request_ordinal=connection_ordinal,
                        global_request_ordinal=ordinal,
                        request_ordinal=ordinal,
                    )
                    faults = self._faults
                    if faults._matches(ordinal, faults.disconnect_before_infer_every_n):
                        return
                    # Preserve existing injection boundaries: stall before enqueue;
                    # delivery delay after execution, outside all service timers.
                    if faults._matches(ordinal, faults.stall_every_n):
                        if self._stop.wait(faults.stall_s):
                            return
                    if faults._matches(ordinal, faults.server_error_every_n):
                        _send_frame(
                            connection,
                            dict(
                                version=_PROTOCOL_VERSION,
                                kind="error",
                                request_id=request_id,
                                **metadata,
                                error_type="InjectedServerError",
                                error="deterministic fault injection",
                            ),
                        )
                        continue
                    response = self._execute(connection, request, metadata)
                    if response is None:
                        return
                    if response["kind"] == "ok":
                        if self._stop.wait(faults.delay_for(ordinal)):
                            return
                        if faults._matches(ordinal, faults.drop_response_every_n):
                            return
                    _send_frame(connection, response)
        except (OSError, EOFError, ValueError, json.JSONDecodeError):
            return
        finally:
            with self._threads_lock:
                self._threads.discard(threading.current_thread())
                self._connections.discard(connection)

    def serve_forever(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    connection, _ = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    return
                thread = threading.Thread(
                    target=self._client_loop,
                    args=(connection,),
                    daemon=True,
                    name="ActionStreamRpcClient",
                )
                with self._threads_lock:
                    if self._stop.is_set():
                        connection.close()
                        return
                    self._threads.add(thread)
                    self._connections.add(connection)
                    thread.start()
        finally:
            self._listener.close()

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(
            target=self.serve_forever, daemon=True, name="ActionStreamRpcServer"
        )
        thread.start()
        return thread

    def close(self) -> None:
        self._stop.set()
        self._listener.close()
        with self._threads_lock:
            threads = list(self._threads)
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        # Stop producers before the sentinel; queued jobs are drained as obsolete.
        for thread in threads:
            thread.join()
        if self._executor.is_alive():
            self._jobs.put(None)
            self._executor.join()

    def __enter__(self) -> "RpcInferenceServer":
        self.start_in_thread()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def load_worker(factory_spec: str) -> tuple[Callable, Callable[[], None] | None]:
    """Load ``module:factory``; factory returns a callable or callable object."""
    module_name, separator, attribute = factory_spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("RPC worker factory must use 'module:callable' syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"RPC worker factory is not callable: {factory_spec}")
    worker = factory()
    if not callable(worker):
        raise TypeError("RPC worker factory must return a callable")
    reset = getattr(worker, "reset", None)
    if reset is not None and not callable(reset):
        raise TypeError("RPC worker reset attribute must be callable")
    return worker, reset
