"""Run deterministic loopback TCP fault matrices and emit machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from actionstream.inference_transport import (
    InferenceDeadlineExceeded,
    InferenceTransportError,
)
from actionstream.rpc_transport import (
    RpcFaultProfile,
    RpcInferenceServer,
    TcpInferenceTransport,
)


class _SyntheticWorker:
    def __init__(self, inference_s: float) -> None:
        self.inference_s = inference_s
        self.calls = 0
        self.resets = 0

    def __call__(self, observation, task):
        if self.inference_s:
            time.sleep(self.inference_s)
        self.calls += 1
        value = observation["value"]
        if not isinstance(value, torch.Tensor):
            raise TypeError("synthetic matrix requires tensor input")
        return value.to(torch.float32) + self.calls

    def reset(self) -> None:
        self.resets += 1


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _fault_profile(row: dict[str, Any]) -> RpcFaultProfile:
    return RpcFaultProfile(
        response_delay_s=float(row.get("response_delay_ms", 0)) / 1000,
        response_jitter_s=float(row.get("response_jitter_ms", 0)) / 1000,
        stall_every_n=int(row.get("stall_every_n", 0)),
        stall_s=float(row.get("stall_ms", 0)) / 1000,
        disconnect_before_infer_every_n=int(
            row.get("disconnect_before_infer_every_n", 0)
        ),
        drop_response_every_n=int(row.get("drop_response_every_n", 0)),
        server_error_every_n=int(row.get("server_error_every_n", 0)),
        seed=int(row.get("fault_seed", 0)),
    )


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    worker = _SyntheticWorker(float(case.get("synthetic_inference_ms", 0)) / 1000)
    faults = _fault_profile(case)
    outcomes: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    with RpcInferenceServer(
        worker,
        reset=worker.reset,
        fault_profile=faults,
    ) as server:
        transport = TcpInferenceTransport(
            server.host,
            server.port,
            connect_timeout_s=float(case.get("connect_timeout_ms", 500)) / 1000,
            control_timeout_s=0.5,
        )
        transport.reset()  # Confirm the worker baseline before the existing matrix.
        for ordinal in range(1, int(case["requests"]) + 1):
            started = time.perf_counter()
            outcome = "success"
            error = None
            try:
                result = transport.infer(
                    {"value": torch.tensor([ordinal], dtype=torch.float32)},
                    "fault-matrix",
                    timeout_s=float(case["deadline_ms"]) / 1000,
                )
                if result.shape != (1,) or not torch.isfinite(result).all():
                    raise RuntimeError("invalid synthetic RPC result")
            except InferenceDeadlineExceeded as exc:
                outcome = "deadline"
                error = str(exc)
            except InferenceTransportError as exc:
                telemetry = transport.telemetry()
                prior_server_errors = sum(
                    row["outcome"] == "server_error" for row in outcomes
                )
                outcome = (
                    "server_error"
                    if telemetry.server_errors > prior_server_errors
                    else "transport_error"
                )
                error = str(exc)
            elapsed_ms = 1000 * (time.perf_counter() - started)
            if outcome == "success":
                latencies_ms.append(elapsed_ms)
            outcomes.append(
                {
                    "ordinal": ordinal,
                    "outcome": outcome,
                    "elapsed_ms": elapsed_ms,
                    "error": error,
                }
            )
        telemetry = transport.telemetry()
        transport.close()
    counts = {
        name: sum(row["outcome"] == name for row in outcomes)
        for name in ("success", "deadline", "transport_error", "server_error")
    }
    result = {
        "name": case["name"],
        "fault_profile": asdict(faults),
        "requests": int(case["requests"]),
        "deadline_ms": float(case["deadline_ms"]),
        "counts": counts,
        "latency_ms": {
            "p50": statistics.median(latencies_ms) if latencies_ms else None,
            "p95": _quantile(latencies_ms, 0.95),
            "max": max(latencies_ms) if latencies_ms else None,
        },
        "worker_calls": worker.calls,
        "worker_resets": worker.resets,
        "transport": telemetry.to_dict(),
        "outcomes": outcomes,
    }
    expected = case.get("expected_counts", {})
    result["gates"] = {
        key: {
            "expected": int(value),
            "actual": counts[key],
            "pass": counts[key] == int(value),
        }
        for key, value in expected.items()
    }
    result["pass"] = all(row["pass"] for row in result["gates"].values())
    return result


def run_matrix(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != 1:
        raise ValueError("fault matrix schema_version must be 1")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fault matrix requires non-empty cases")
    names = [row.get("name") for row in cases]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("every fault case requires a name")
    if len(set(names)) != len(names):
        raise ValueError("fault case names must be unique")
    results = [_run_case(case) for case in cases]
    return {
        "schema_version": 1,
        "purpose": config.get("purpose", "rpc_transport_fault_matrix"),
        "cases": results,
        "pass": all(row["pass"] for row in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_matrix(config)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (args.output / "requests.jsonl").open("x", encoding="utf-8") as handle:
        for case in result["cases"]:
            for row in case["outcomes"]:
                handle.write(
                    json.dumps(
                        {"case": case["name"], **row},
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
