"""Inference transports with explicit deadline and cancellation semantics.

The in-process transport preserves the normal LeRobot local-policy path, but a
Python thread cannot safely cancel a blocked CUDA or network call.  Deployments
that need a hard deadline use :class:`ProcessInferenceTransport`: a dedicated
child owns the remote client, and timeout/reset/stop terminate that child before
the engine continues.  The next request starts a fresh child from the pinned
factory entry point.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import torch


class InferenceDeadlineExceeded(TimeoutError):
    """A transport request exceeded an enforced deadline."""


class InferenceCancelled(RuntimeError):
    """A transport request was cancelled by reset or shutdown."""


class InferenceTransportError(RuntimeError):
    """The isolated transport failed outside the caller process."""


class InferenceTransport(Protocol):
    """Minimal transport contract consumed by the inference engine."""

    @property
    def deadline_enforced(self) -> bool: ...

    @property
    def process_restarts(self) -> int: ...

    @property
    def cancellations(self) -> int: ...

    @property
    def latest_startup_latency_s(self) -> float | None: ...

    @property
    def latest_request_latency_s(self) -> float | None: ...

    def infer(
        self,
        observation: Mapping[str, Any],
        task: str,
        *,
        timeout_s: float,
    ) -> torch.Tensor: ...

    def cancel(self) -> bool: ...

    def reset(self) -> bool: ...

    def close(self) -> None: ...


class DirectInferenceTransport:
    """Run inference in the engine thread with an explicitly advisory deadline.

    This path is appropriate for local CUDA inference where the policy already
    lives in the LeRobot process.  A slow call is reported after it returns, but
    cannot be preempted.  Telemetry exposes ``deadline_enforced=False`` so the
    mode cannot be mistaken for the process-isolated contract.
    """

    def __init__(self, infer: Callable[[Mapping[str, Any], str], torch.Tensor]) -> None:
        self._infer = infer
        self._latest_request_latency_s: float | None = None
        self._state_lock = threading.Lock()
        self._generation = 0

    @property
    def deadline_enforced(self) -> bool:
        return False

    @property
    def process_restarts(self) -> int:
        return 0

    @property
    def cancellations(self) -> int:
        return 0

    @property
    def latest_startup_latency_s(self) -> float | None:
        return 0.0

    @property
    def latest_request_latency_s(self) -> float | None:
        with self._state_lock:
            return self._latest_request_latency_s

    def infer(
        self,
        observation: Mapping[str, Any],
        task: str,
        *,
        timeout_s: float,
    ) -> torch.Tensor:
        return self._infer_request(observation, task, timeout_s=timeout_s)

    def _infer_request(
        self,
        observation: Mapping[str, Any],
        task: str,
        *,
        timeout_s: float,
        cancellation_event: threading.Event | None = None,
    ) -> torch.Tensor:
        with self._state_lock:
            if cancellation_event is not None and cancellation_event.is_set():
                raise InferenceCancelled("Direct request invalidated before entry")
            generation = self._generation
        started = time.perf_counter()
        result = self._infer(observation, task)
        elapsed = time.perf_counter() - started
        with self._state_lock:
            if generation == self._generation and not (
                cancellation_event is not None and cancellation_event.is_set()
            ):
                self._latest_request_latency_s = elapsed
        if cancellation_event is not None and cancellation_event.is_set():
            raise InferenceCancelled("Direct request invalidated during inference")
        if elapsed > timeout_s:
            raise InferenceDeadlineExceeded(
                f"Direct inference returned after advisory deadline: "
                f"{elapsed:.3f}s > {timeout_s:.3f}s"
            )
        return result

    def infer_cancellable(
        self,
        observation: Mapping[str, Any],
        task: str,
        *,
        timeout_s: float,
        cancellation_event: threading.Event,
    ) -> torch.Tensor:
        """Discard invalidated calls; running Python/CUDA remains advisory."""
        return self._infer_request(
            observation,
            task,
            timeout_s=timeout_s,
            cancellation_event=cancellation_event,
        )

    def cancel(self) -> bool:
        return self.reset()

    def reset(self) -> bool:
        with self._state_lock:
            self._generation += 1
            self._latest_request_latency_s = None
        return False

    def close(self) -> None:
        self.reset()


def _load_factory(
    factory_spec: str,
) -> Callable[[], Callable[[Mapping[str, Any], str], Any]]:
    module_name, separator, attribute = factory_spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("process transport factory must use 'module:callable' syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"Process transport factory is not callable: {factory_spec}")
    return factory


def _process_transport_main(connection: Any, factory_spec: str) -> None:
    """Own one transport client and answer request messages until closed."""

    try:
        infer = _load_factory(factory_spec)()
        if not callable(infer):
            raise TypeError(
                f"Process transport factory did not return a callable: {factory_spec}"
            )
        connection.send(("ready", None, None))
        while True:
            message = connection.recv()
            kind = message[0]
            if kind == "close":
                return
            if kind != "infer":
                raise ValueError(f"Unknown process transport message: {kind!r}")
            _, request_id, observation, task = message
            try:
                result = infer(observation, task)
                if not isinstance(result, torch.Tensor):
                    raise TypeError(
                        f"Process transport must return torch.Tensor, got {type(result)!r}"
                    )
                connection.send(("ok", request_id, result.detach().cpu()))
            except BaseException as exc:
                connection.send(
                    (
                        "error",
                        request_id,
                        "".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )
                        ),
                    )
                )
    except BaseException as exc:
        try:
            connection.send(
                (
                    "fatal",
                    None,
                    "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                )
            )
        except BaseException:
            pass
    finally:
        connection.close()


class ProcessInferenceTransport:
    """Persistent, restartable child-process transport with hard deadlines."""

    def __init__(
        self,
        factory_spec: str,
        *,
        start_method: str = "spawn",
        startup_timeout_s: float = 30.0,
        terminate_timeout_s: float = 1.0,
    ) -> None:
        _load_factory(factory_spec)
        if startup_timeout_s <= 0 or terminate_timeout_s <= 0:
            raise ValueError("Process transport timeouts must be positive")
        self._factory_spec = factory_spec
        self._context = mp.get_context(start_method)
        self._startup_timeout_s = float(startup_timeout_s)
        self._terminate_timeout_s = float(terminate_timeout_s)
        self._request_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._process: Any | None = None
        self._connection: Any | None = None
        self._inflight = False
        self._process_restarts = 0
        self._cancellations = 0
        self._latest_startup_latency_s: float | None = None
        self._latest_request_latency_s: float | None = None
        self._generation = 0

    @property
    def deadline_enforced(self) -> bool:
        return True

    @property
    def process_restarts(self) -> int:
        with self._state_lock:
            return self._process_restarts

    @property
    def cancellations(self) -> int:
        with self._state_lock:
            return self._cancellations

    @property
    def latest_startup_latency_s(self) -> float | None:
        with self._state_lock:
            return self._latest_startup_latency_s

    @property
    def latest_request_latency_s(self) -> float | None:
        with self._state_lock:
            return self._latest_request_latency_s

    def _start(
        self,
        cancellation_event: threading.Event | None = None,
    ) -> tuple[Any, Any]:
        """Start the client without holding the state lock while it imports.

        Reset and stop must be able to terminate a child even while its factory
        is importing Torch or constructing a network client.  The request lock
        guarantees only one caller can start a child; the state lock protects
        publication and termination of that child, not the blocking ready wait.
        """

        with self._state_lock:
            # Invalidation may precede entry into infer(). Check under the same
            # lock that publishes a child, so reset either prevents or kills it.
            if self._cancel_event.is_set() or (
                cancellation_event is not None and cancellation_event.is_set()
            ):
                raise InferenceCancelled("Process request invalidated before startup")
            if self._process is not None and self._process.is_alive():
                if self._connection is None:
                    raise InferenceTransportError(
                        "Live process transport is missing its connection"
                    )
                return self._connection, self._process
            parent, child = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=_process_transport_main,
                args=(child, self._factory_spec),
                name="ActionStreamTransport",
                daemon=True,
            )
            process.start()
            child.close()
            self._process = process
            self._connection = parent
            self._process_restarts += 1

        try:
            if not parent.poll(self._startup_timeout_s):
                with self._state_lock:
                    if self._process is process:
                        self._terminate_locked()
                raise InferenceTransportError(
                    "Process transport did not start within "
                    f"{self._startup_timeout_s:.3f}s"
                )
            status, _, payload = parent.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            if self._cancel_event.is_set():
                raise InferenceCancelled("Process transport startup cancelled") from exc
            raise InferenceTransportError(
                "Process transport connection failed during startup"
            ) from exc
        if status != "ready":
            with self._state_lock:
                if self._process is process:
                    self._terminate_locked()
            raise InferenceTransportError(
                f"Process transport startup failed ({status}): {payload}"
            )
        if self._cancel_event.is_set():
            raise InferenceCancelled("Process transport startup cancelled")
        return parent, process

    def _terminate_locked(self) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        self._inflight = False
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(timeout=self._terminate_timeout_s)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=self._terminate_timeout_s)
        process.close()

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
        """Optional extension retaining the original infer() calling contract."""
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
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        def cancelled() -> bool:
            return self._cancel_event.is_set() or (
                cancellation_event is not None and cancellation_event.is_set()
            )

        with self._request_lock:
            self._cancel_event.clear()
            request_id = uuid.uuid4().hex
            startup_started = time.monotonic()
            request_started: float | None = None
            with self._state_lock:
                self._inflight = True
                generation = self._generation
            try:
                connection, process = self._start(cancellation_event)
                with self._state_lock:
                    if generation == self._generation and not cancelled():
                        self._latest_startup_latency_s = (
                            time.monotonic() - startup_started
                        )
                if cancelled():
                    raise InferenceCancelled("Process transport request cancelled")
                with self._state_lock:
                    if (
                        self._connection is not connection
                        or self._process is not process
                    ):
                        raise InferenceCancelled(
                            "Process transport request cancelled before send"
                        )
                connection.send(("infer", request_id, dict(observation), task))
                request_started = time.monotonic()
                deadline = time.monotonic() + timeout_s
                while True:
                    if cancelled():
                        raise InferenceCancelled("Process transport request cancelled")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        termination_error: BaseException | None = None
                        try:
                            with self._state_lock:
                                self._terminate_locked()
                        except BaseException as exc:
                            # Closing a Windows multiprocessing pipe can itself
                            # raise while the peer is being terminated.  The
                            # causal event is still the enforced deadline, not
                            # a transport disconnect; preserve that contract.
                            termination_error = exc
                        raise InferenceDeadlineExceeded(
                            f"Process transport exceeded {timeout_s:.3f}s deadline"
                        ) from termination_error
                    if connection.poll(min(remaining, 0.02)):
                        status, response_id, payload = connection.recv()
                        if response_id != request_id:
                            raise InferenceTransportError(
                                f"Process transport response id mismatch: {response_id}"
                            )
                        if status == "ok":
                            return payload
                        raise InferenceTransportError(
                            f"Process transport request failed ({status}): {payload}"
                        )
                    if not process.is_alive():
                        raise InferenceTransportError(
                            f"Process transport exited during request (exit={process.exitcode})"
                        )
            except InferenceDeadlineExceeded:
                # TimeoutError is an OSError subclass on Python, so it must be
                # preserved before the connection-error normalization below.
                raise
            except (EOFError, BrokenPipeError, OSError) as exc:
                if cancelled():
                    raise InferenceCancelled(
                        "Process transport request cancelled"
                    ) from exc
                raise InferenceTransportError(
                    "Process transport connection failed"
                ) from exc
            finally:
                with self._state_lock:
                    if (
                        request_started is not None
                        and generation == self._generation
                        and not cancelled()
                    ):
                        self._latest_request_latency_s = (
                            time.monotonic() - request_started
                        )
                    self._inflight = False

    def cancel(self) -> bool:
        self._cancel_event.set()
        with self._state_lock:
            if not self._inflight:
                return False
            self._generation += 1
            self._cancellations += 1
            self._terminate_locked()
            return True

    def reset(self) -> bool:
        """Restart the isolated client so no pre-reset provider state survives."""

        self._cancel_event.set()
        with self._state_lock:
            self._generation += 1
            self._latest_startup_latency_s = None
            self._latest_request_latency_s = None
            was_inflight = self._inflight
            if was_inflight:
                self._cancellations += 1
            self._terminate_locked()
            return was_inflight

    def close(self) -> None:
        self._cancel_event.set()
        with self._state_lock:
            self._generation += 1
            self._terminate_locked()
