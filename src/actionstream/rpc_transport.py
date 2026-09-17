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
import random
import socket
import struct
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
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


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError("RPC peer closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_frame(connection: socket.socket) -> tuple[dict[str, Any], int]:
    header = _recv_exact(connection, _HEADER.size)
    (length,) = _HEADER.unpack(header)
    if length <= 0 or length > _MAX_FRAME_BYTES:
        raise ValueError(f"Invalid RPC frame length: {length}")
    body = _recv_exact(connection, length)
    decoded = json.loads(body.decode("utf-8"))
    value = _decode_value(decoded)
    if not isinstance(value, dict):
        raise ValueError("RPC top-level frame must be a mapping")
    return value, _HEADER.size + length


def _send_frame(connection: socket.socket, payload: Mapping[str, Any]) -> int:
    frame = _encode_frame(payload)
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

    def to_dict(self) -> dict[str, int | float | None]:
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

    def _ensure_socket(self, timeout_s: float) -> socket.socket:
        with self._state_lock:
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
            if self._socket is not None:
                connection.close()
                return self._socket
            self._socket = connection
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
        deadline = time.monotonic() + timeout_s
        with self._request_lock:
            with self._state_lock:
                generation = self._generation
                self._inflight = True
                self._requests_started += 1
            started = time.monotonic()
            try:
                if cancellation_event is not None and cancellation_event.is_set():
                    raise InferenceCancelled("TCP request cancelled before connect")
                connection = self._ensure_socket(self._remaining(deadline))
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
                )
                with self._state_lock:
                    self._bytes_sent += sent
                if cancellation_event is not None and cancellation_event.is_set():
                    raise InferenceCancelled("TCP request cancelled after send")
                connection.settimeout(self._remaining(deadline))
                response, received = _recv_frame(connection)
                with self._state_lock:
                    self._bytes_received += received
                if response.get("version") != _PROTOCOL_VERSION:
                    raise InferenceTransportError("RPC protocol version mismatch")
                if response.get("request_id") != request_id:
                    raise InferenceTransportError("RPC response id mismatch")
                kind = response.get("kind")
                if kind == "error":
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
                    self._latest_roundtrip_latency_s = time.monotonic() - started
                    server_s = response.get("server_inference_s")
                    self._latest_server_inference_s = (
                        float(server_s) if server_s is not None else None
                    )
                return actions
            except (socket.timeout, TimeoutError) as exc:
                with self._state_lock:
                    if generation != self._generation:
                        raise InferenceCancelled("TCP request invalidated") from exc
                    self._deadlines += 1
                    self._close_socket_locked()
                raise InferenceDeadlineExceeded(
                    f"TCP inference exceeded {timeout_s:.3f}s deadline"
                ) from exc
            except InferenceDeadlineExceeded:
                with self._state_lock:
                    self._deadlines += 1
                    self._close_socket_locked()
                raise
            except InferenceCancelled:
                with self._state_lock:
                    self._close_socket_locked()
                raise
            except (EOFError, OSError, ValueError, json.JSONDecodeError) as exc:
                with self._state_lock:
                    invalidated = generation != self._generation
                    self._close_socket_locked()
                    if not invalidated:
                        self._transport_errors += 1
                if invalidated:
                    raise InferenceCancelled("TCP request invalidated") from exc
                raise InferenceTransportError("TCP inference transport failed") from exc
            finally:
                with self._state_lock:
                    self._inflight = False

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
            )
            with self._state_lock:
                self._bytes_sent += sent
            connection.settimeout(self._remaining(deadline))
            response, received = _recv_frame(connection)
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
            except (InferenceTransportError, InferenceDeadlineExceeded, OSError):
                with self._state_lock:
                    self._close_socket_locked()
                # A fresh connection is still a clean client-side generation.
        return cancelled

    def close(self) -> None:
        with self._state_lock:
            self._generation += 1
            self._close_socket_locked()


class RpcInferenceServer:
    """Small threaded TCP server around a trusted inference callable."""

    def __init__(
        self,
        infer: Callable[[Mapping[str, Any], str], torch.Tensor],
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        reset: Callable[[], None] | None = None,
        fault_profile: RpcFaultProfile | None = None,
        backlog: int = 16,
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
        self._infer_lock = threading.Lock()
        self._threads: set[threading.Thread] = set()
        self._connections: set[socket.socket] = set()
        self._threads_lock = threading.Lock()

    def _claim_ordinal(self) -> int:
        with self._ordinal_lock:
            ordinal = self._next_ordinal
            self._next_ordinal += 1
            return ordinal

    def _client_loop(self, connection: socket.socket) -> None:
        try:
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while not self._stop.is_set():
                    try:
                        request, _ = _recv_frame(connection)
                    except EOFError:
                        return
                    if request.get("version") != _PROTOCOL_VERSION:
                        return
                    request_id = request.get("request_id")
                    kind = request.get("kind")
                    if kind == "reset":
                        if self._reset is not None:
                            with self._infer_lock:
                                self._reset()
                        _send_frame(
                            connection,
                            {
                                "version": _PROTOCOL_VERSION,
                                "kind": "ack",
                                "request_id": request_id,
                            },
                        )
                        continue
                    if kind == "close":
                        _send_frame(
                            connection,
                            {
                                "version": _PROTOCOL_VERSION,
                                "kind": "ack",
                                "request_id": request_id,
                            },
                        )
                        return
                    if kind != "infer":
                        return
                    ordinal = self._claim_ordinal()
                    faults = self._faults
                    if faults._matches(ordinal, faults.disconnect_before_infer_every_n):
                        return
                    if faults._matches(ordinal, faults.stall_every_n):
                        time.sleep(faults.stall_s)
                    if faults._matches(ordinal, faults.server_error_every_n):
                        _send_frame(
                            connection,
                            {
                                "version": _PROTOCOL_VERSION,
                                "kind": "error",
                                "request_id": request_id,
                                "request_ordinal": ordinal,
                                "error_type": "InjectedServerError",
                                "error": "deterministic fault injection",
                            },
                        )
                        continue
                    started = time.perf_counter()
                    try:
                        with self._infer_lock:
                            actions = self._infer(
                                request["observation"], str(request["task"])
                            )
                        if not isinstance(actions, torch.Tensor):
                            raise TypeError(
                                f"RPC worker must return torch.Tensor, got {type(actions)!r}"
                            )
                    except BaseException as exc:
                        _send_frame(
                            connection,
                            {
                                "version": _PROTOCOL_VERSION,
                                "kind": "error",
                                "request_id": request_id,
                                "request_ordinal": ordinal,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        )
                        continue
                    inference_s = time.perf_counter() - started
                    delay = faults.delay_for(ordinal)
                    if delay:
                        time.sleep(delay)
                    if faults._matches(ordinal, faults.drop_response_every_n):
                        return
                    _send_frame(
                        connection,
                        {
                            "version": _PROTOCOL_VERSION,
                            "kind": "ok",
                            "request_id": request_id,
                            "request_ordinal": ordinal,
                            "server_inference_s": inference_s,
                            "actions": actions.detach().cpu(),
                        },
                    )
        except (OSError, EOFError, ValueError, json.JSONDecodeError):
            return
        finally:
            current = threading.current_thread()
            with self._threads_lock:
                self._threads.discard(current)
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
                    self._threads.add(thread)
                    self._connections.add(connection)
                thread.start()
        finally:
            try:
                self._listener.close()
            except OSError:
                pass

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(
            target=self.serve_forever,
            daemon=True,
            name="ActionStreamRpcServer",
        )
        thread.start()
        return thread

    def close(self) -> None:
        self._stop.set()
        try:
            self._listener.close()
        except OSError:
            pass
        with self._threads_lock:
            threads = list(self._threads)
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass
        for thread in threads:
            thread.join(timeout=1.0)

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
