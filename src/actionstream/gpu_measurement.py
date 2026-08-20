"""Process-scoped NVIDIA GPU sampling for benchmark provenance."""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _number(value: str) -> float | None:
    cleaned = value.strip().replace(" MiB", "").replace(" %", "")
    if not cleaned or cleaned in {"N/A", "[Not Supported]"}:
        return None
    return float(cleaned)


class NvidiaSmiMonitor:
    """Sample device utilization and this process's driver residency to JSONL."""

    def __init__(self, path: Path, *, interval_s: float = 0.2) -> None:
        if interval_s <= 0:
            raise ValueError("GPU sample interval must be positive")
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_s = float(interval_s)
        self.pid = os.getpid()
        self._condition = threading.Condition()
        self._phase = "process_startup"
        self._context: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="nvidia-smi-monitor",
            daemon=True,
        )
        self._thread.start()

    def set_phase(self, phase: str, **context: Any) -> None:
        if not phase:
            raise ValueError("GPU measurement phase must be non-empty")
        with self._condition:
            self._phase = phase
            self._context = dict(context)
        self.mark("phase", phase=phase, **context)

    def mark(self, event: str, **fields: Any) -> None:
        self._write(
            {
                "schema_version": 1,
                "event": event,
                "utc_unix_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "pid": self.pid,
                **fields,
            }
        )

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(5.0, self.interval_s * 5.0))
            if thread.is_alive():
                raise RuntimeError("nvidia-smi monitor did not stop")
        self._thread = None

    def summary(self) -> dict[str, Any]:
        phases: dict[str, list[dict[str, Any]]] = {}
        for sample in self._samples:
            phases.setdefault(str(sample["phase"]), []).append(sample)
        return {
            "schema_version": 1,
            "sampler": "nvidia-smi",
            "pid": self.pid,
            "interval_seconds": self.interval_s,
            "sample_count": len(self._samples),
            "error_count": len(self._errors),
            "errors": self._errors[:20],
            "phases": {
                phase: self._summarize_phase(samples)
                for phase, samples in sorted(phases.items())
            },
            "jsonl_path": str(self.path),
        }

    @staticmethod
    def _summarize_phase(samples: list[dict[str, Any]]) -> dict[str, Any]:
        utility = [
            float(item["gpu_utilization_percent"])
            for item in samples
            if item.get("gpu_utilization_percent") is not None
        ]
        process_memory = [
            float(item["process_gpu_memory_mib"])
            for item in samples
            if item.get("process_gpu_memory_mib") is not None
        ]
        device_memory = [
            float(item["device_memory_used_mib"])
            for item in samples
            if item.get("device_memory_used_mib") is not None
        ]
        return {
            "sample_count": len(samples),
            "resident_sample_count": sum(
                bool(item.get("process_resident")) for item in samples
            ),
            "gpu_utilization_percent_p50": _percentile(utility, 50),
            "gpu_utilization_percent_p95": _percentile(utility, 95),
            "gpu_utilization_percent_max": max(utility, default=None),
            "process_gpu_memory_mib_p50": _percentile(process_memory, 50),
            "process_gpu_memory_mib_p95": _percentile(process_memory, 95),
            "process_gpu_memory_mib_max": max(process_memory, default=None),
            "device_memory_used_mib_p50": _percentile(device_memory, 50),
            "device_memory_used_mib_max": max(device_memory, default=None),
            "sample_interval_seconds_mean": (
                statistics.fmean(
                    (
                        samples[index]["monotonic_ns"]
                        - samples[index - 1]["monotonic_ns"]
                    )
                    / 1e9
                    for index in range(1, len(samples))
                )
                if len(samples) > 1
                else None
            ),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                sample = self._sample()
                self._samples.append(sample)
                self._write(sample)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self._errors.append(message)
                self.mark("sampler_error", error=message)
            self._stop.wait(max(0.0, self.interval_s - (time.monotonic() - started)))

    def _sample(self) -> dict[str, Any]:
        device = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        device_rows = [
            [field.strip() for field in line.split(",")]
            for line in device.stdout.splitlines()
            if line.strip()
        ]
        if len(device_rows) != 1 or len(device_rows[0]) != 5:
            raise RuntimeError(f"Expected one GPU row, got {device_rows!r}")
        index, gpu_uuid, utility, used, total = device_rows[0]

        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        process_memory: float | None = None
        for line in apps.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            app_pid, app_uuid, app_memory = fields
            if app_pid == str(self.pid) and app_uuid == gpu_uuid:
                process_memory = _number(app_memory)
                break

        with self._condition:
            phase = self._phase
            context = dict(self._context)
        return {
            "schema_version": 1,
            "event": "nvidia_smi_sample",
            "utc_unix_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": self.pid,
            "phase": phase,
            **context,
            "gpu_index": int(index),
            "gpu_uuid": gpu_uuid,
            "gpu_utilization_percent": _number(utility),
            "device_memory_used_mib": _number(used),
            "device_memory_total_mib": _number(total),
            "process_resident": process_memory is not None,
            "process_gpu_memory_mib": process_memory,
        }

    def _write(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        with (
            self._condition,
            self.path.open("a", encoding="utf-8", newline="\n") as stream,
        ):
            stream.write(encoded + "\n")
            stream.flush()
