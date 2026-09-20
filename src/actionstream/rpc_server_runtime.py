"""Socket listener and fault-injection runtime for ActionStream RPC."""

from __future__ import annotations

import importlib
import json
import math
import random
import socket
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from actionstream.rpc_executor import RpcExecutor
from actionstream.rpc_protocol import _PROTOCOL_VERSION, _recv_frame, _send_frame


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



class RpcInferenceServer:
    """Route socket connections into one persistent serialized executor."""

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
        if reset is not None and not callable(reset):
            raise TypeError("reset must be callable or None (stateless no-op)")
        self._faults = fault_profile or RpcFaultProfile()
        self.reset_capability = "callback" if reset is not None else "stateless_noop"
        self._listener = socket.socket(
            socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM
        )
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(backlog)
        self._listener.settimeout(0.1)
        self.host, self.port = self._listener.getsockname()[:2]
        self._stop = threading.Event()
        self._ordinal_lock = threading.Lock()
        self._next_ordinal = 1
        self._threads: set[threading.Thread] = set()
        self._connections: set[socket.socket] = set()
        self._threads_lock = threading.Lock()
        self._executor_runtime = RpcExecutor(
            infer,
            reset=reset,
            stop_event=self._stop,
            telemetry_jsonl_path=telemetry_jsonl_path,
        )
        # Compatibility aliases retained for existing diagnostics/tests.
        self._jobs = self._executor_runtime.jobs
        self._executor = self._executor_runtime.thread

    def telemetry(self) -> dict[str, int]:
        return self._executor_runtime.telemetry()

    def _execute(
        self,
        connection: socket.socket,
        request: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        return self._executor_runtime.execute(connection, request, metadata)

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
                    if faults._matches(
                        ordinal, faults.disconnect_before_infer_every_n
                    ):
                        return
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
                        if faults._matches(
                            ordinal, faults.drop_response_every_n
                        ):
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
            target=self.serve_forever,
            daemon=True,
            name="ActionStreamRpcServer",
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
        for thread in threads:
            thread.join()
        self._executor_runtime.close()

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
