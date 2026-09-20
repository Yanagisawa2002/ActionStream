"""Single-owner persistent executor for ActionStream RPC workers.

Socket threads may admit work, but only this executor invokes the worker or reset
callback. That invariant makes remote policy state ownership explicit.
"""

from __future__ import annotations

import json
import queue
import select
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from actionstream.rpc_protocol import _PROTOCOL_VERSION


@dataclass
class _ExecutorRequest:
    request: dict[str, Any]
    connection: socket.socket
    metadata: dict[str, Any]
    enqueued_at: float = field(default_factory=time.perf_counter)
    obsolete: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None


class RpcExecutor:
    """Serialize inference/reset calls through one persistent worker owner."""

    def __init__(
        self,
        infer: Callable[[Mapping[str, Any], str], torch.Tensor],
        *,
        reset: Callable[[], None] | None,
        stop_event: threading.Event,
        telemetry_jsonl_path: str | Path | None = None,
    ) -> None:
        self._infer = infer
        self._reset = reset
        self._stop = stop_event
        self.jobs: queue.Queue[_ExecutorRequest | None] = queue.Queue()
        self._telemetry_lock = threading.Lock()
        self._telemetry_path = (
            Path(telemetry_jsonl_path) if telemetry_jsonl_path else None
        )
        self._counters = dict(
            executor_starts=1,
            inference_calls=0,
            reset_calls=0,
            reset_successes=0,
            reset_failures=0,
            dropped_obsolete_before_compute=0,
            invalidated_results=0,
        )
        self.thread = threading.Thread(
            target=self._loop, name="ActionStreamRpcExecutor", daemon=True
        )
        self.thread.start()

    def telemetry(self) -> dict[str, int]:
        with self._telemetry_lock:
            return dict(self._counters)

    def _count(self, name: str) -> None:
        with self._telemetry_lock:
            self._counters[name] += 1

    def _record(self, event: str, job: _ExecutorRequest, **fields: Any) -> None:
        if self._telemetry_path is None:
            return
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

    def _loop(self) -> None:
        while True:
            job = self.jobs.get()
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
                        self._count("reset_successes")
                        self._record("reset_succeeded", job)
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
                    if job.request["kind"] == "reset":
                        self._count("reset_failures")
                        self._record("reset_failed", job, error_type=type(exc).__name__)
                    inference_finished = time.perf_counter()
                    response.update(
                        kind="error",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                finished = time.perf_counter()
                timing = dict(
                    server_queue_wait_s=started - job.enqueued_at,
                    server_compute_s=compute_s,
                    server_service_s=finished - started,
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

    def execute(
        self,
        connection: socket.socket,
        request: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        job = _ExecutorRequest(request, connection, metadata)
        self._record("enqueued", job)
        job.enqueued_at = time.perf_counter()
        self.jobs.put(job)
        try:
            while not job.done.wait(0.01):
                if self._obsolete(job):
                    return None
            return None if self._obsolete(job) else job.response
        finally:
            job.obsolete.set()

    def close(self) -> None:
        if self.thread.is_alive():
            self.jobs.put(None)
            self.thread.join()
