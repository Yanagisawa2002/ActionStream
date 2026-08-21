"""Frozen single-hypothesis fixture for response-delivery serialization.

The fixture changes exactly one variable: whether the injected transport delay
blocks the single inference worker or is handled by the response-delivery
scheduler.  Policy compute latency, control rate, chunk size, alignment,
fallback, hold budget, observations, and request mailbox semantics are shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from actionstream.lerobot_inference import (
    ActionStreamInferenceConfig,
    ActionStreamInferenceEngine,
)


@dataclass(frozen=True)
class TransportH1Protocol:
    raw: Mapping[str, Any]
    control_frequency_hz: float
    chunk_size: int
    compute_latency_seconds: float
    delivery_delay_seconds: float
    scored_control_steps: int
    repetitions: int
    bounded_hold_steps: int
    gates: Mapping[str, float | int]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def load_transport_h1_protocol(path: Path | str) -> TransportH1Protocol:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("transport H1 requires schema_version=1")
    if raw.get("protocol_status") != "frozen_before_formal_result":
        raise ValueError("transport H1 protocol must be frozen before formal result")
    if raw.get("hypothesis_id") != "transport_h1_serialized_delivery":
        raise ValueError("unexpected transport H1 hypothesis_id")

    shared = raw.get("shared_contract", {})
    variants = raw.get("variants", {})
    if set(variants) != {"serialized", "pipelined"}:
        raise ValueError("transport H1 requires serialized and pipelined variants")
    if variants["serialized"] != {"delivery_scheduler_enabled": False}:
        raise ValueError("serialized variant may only disable the delivery scheduler")
    if variants["pipelined"] != {"delivery_scheduler_enabled": True}:
        raise ValueError("pipelined variant may only enable the delivery scheduler")
    if shared.get("latest_only_fallback") is not False:
        raise ValueError("transport H1 must keep latest-only fallback disabled")

    positive_fields = {
        "control_frequency_hz": float,
        "chunk_size": int,
        "compute_latency_milliseconds": float,
        "delivery_delay_milliseconds": float,
        "scored_control_steps": int,
        "repetitions": int,
    }
    parsed: dict[str, float | int] = {}
    for key, value_type in positive_fields.items():
        value = shared.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        converted = value_type(value)
        if converted <= 0:
            raise ValueError(f"{key} must be positive")
        parsed[key] = converted
    bounded_hold_steps = shared.get("bounded_hold_steps")
    if (
        isinstance(bounded_hold_steps, bool)
        or not isinstance(bounded_hold_steps, int)
        or bounded_hold_steps < 0
    ):
        raise ValueError("bounded_hold_steps must be a non-negative integer")

    gates = raw.get("decision_gates", {})
    required_gates = {
        "minimum_request_rate_ratio",
        "minimum_depletion_reduction_fraction",
        "maximum_pipelined_depletion_fraction",
        "maximum_out_of_order_rejections",
        "maximum_fallback_activations",
    }
    if set(gates) != required_gates:
        raise ValueError("transport H1 decision gates are incomplete or expanded")

    return TransportH1Protocol(
        raw=raw,
        control_frequency_hz=float(parsed["control_frequency_hz"]),
        chunk_size=int(parsed["chunk_size"]),
        compute_latency_seconds=(
            float(parsed["compute_latency_milliseconds"]) / 1000.0
        ),
        delivery_delay_seconds=(
            float(parsed["delivery_delay_milliseconds"]) / 1000.0
        ),
        scored_control_steps=int(parsed["scored_control_steps"]),
        repetitions=int(parsed["repetitions"]),
        bounded_hold_steps=bounded_hold_steps,
        gates=gates,
    )


class _Resettable:
    def reset(self) -> None:
        return None


def _wait_for_first_chunk(
    engine: ActionStreamInferenceEngine,
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while engine.telemetry.queue_depth == 0:
        if engine.failed:
            raise RuntimeError(engine.failure_traceback or "transport H1 engine failed")
        if time.monotonic() >= deadline:
            raise TimeoutError("transport H1 timed out waiting for its first chunk")
        time.sleep(0.002)


def _run_variant(
    protocol: TransportH1Protocol,
    *,
    variant: str,
    repetition: int,
) -> dict[str, Any]:
    pipelined = variant == "pipelined"
    inference_calls = 0

    def infer(observation: Mapping[str, Any], _task: str) -> torch.Tensor:
        nonlocal inference_calls
        inference_calls += 1
        time.sleep(protocol.compute_latency_seconds)
        if not pipelined:
            time.sleep(protocol.delivery_delay_seconds)
        value = float(observation["control_step"])
        return torch.full((1, protocol.chunk_size, 1), value, dtype=torch.float32)

    engine = ActionStreamInferenceEngine(
        policy=SimpleNamespace(config=SimpleNamespace(use_amp=False)),
        preprocessor=_Resettable(),
        postprocessor=_Resettable(),
        hw_features={},
        task="transport H1 synthetic task",
        device="cpu",
        robot_type="mock",
        config=ActionStreamInferenceConfig(
            inference_timeout_s=max(
                5.0,
                2.0
                * (
                    protocol.compute_latency_seconds
                    + protocol.delivery_delay_seconds
                ),
            ),
            bounded_hold_steps=protocol.bounded_hold_steps,
            retry_backoff_s=0.0,
            max_consecutive_failures=2,
            join_timeout_s=2.0,
            latest_only_fallback=False,
            delivery_scheduler_enabled=pipelined,
        ),
        infer_chunk=infer,
        reset_provider=lambda: None,
        delivery_delay_provider=(
            (lambda _ordinal: protocol.delivery_delay_seconds)
            if pipelined
            else None
        ),
    )

    engine.reset()
    engine.start()
    engine.resume()
    engine.notify_observation({"control_step": 0})
    _wait_for_first_chunk(
        engine,
        timeout_seconds=max(
            10.0,
            4.0
            * (protocol.compute_latency_seconds + protocol.delivery_delay_seconds),
        ),
    )

    period = 1.0 / protocol.control_frequency_hz
    scheduled = time.monotonic()
    scored_started = scheduled
    depletion_pulls = 0
    nonempty_pulls = 0
    try:
        for control_step in range(protocol.scored_control_steps):
            remaining = scheduled - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            action = engine.get_action(None)
            if action is None:
                depletion_pulls += 1
            else:
                nonempty_pulls += 1
            if control_step + 1 < protocol.scored_control_steps:
                engine.notify_observation({"control_step": control_step + 1})
            scheduled += period
    finally:
        engine.pause()
        engine.stop()
    scored_finished = time.monotonic()
    telemetry = engine.telemetry
    wall_seconds = scored_finished - scored_started
    return {
        "variant": variant,
        "repetition": repetition,
        "status": "engine_failed" if telemetry.failed else "completed",
        "scored_control_steps": protocol.scored_control_steps,
        "scored_wall_seconds": wall_seconds,
        "inference_calls_observed": inference_calls,
        "inference_started": telemetry.inference_started,
        "inference_completed": telemetry.inference_completed,
        "inference_requests_per_second": (
            telemetry.inference_started / wall_seconds
        ),
        "depletion_pulls": depletion_pulls,
        "depletion_fraction": depletion_pulls / protocol.scored_control_steps,
        "nonempty_pulls": nonempty_pulls,
        "bounded_hold_steps": telemetry.hold_actions,
        "hold_exhausted": telemetry.hold_exhausted,
        "chunks_accepted": telemetry.chunks_accepted,
        "chunks_rejected_stale": telemetry.chunks_rejected_stale,
        "responses_scheduled": telemetry.responses_scheduled,
        "responses_delivered": telemetry.responses_delivered,
        "responses_rejected_out_of_order": (
            telemetry.responses_rejected_out_of_order
        ),
        "fallback_activations": telemetry.fallback_activations,
        "pending_responses_after_stop": telemetry.pending_responses,
        "failed": telemetry.failed,
    }


def _aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant"] == variant]
    return {
        "variant": variant,
        "repetitions": len(selected),
        "all_completed": all(row["status"] == "completed" for row in selected),
        "median_request_rate": median(
            float(row["inference_requests_per_second"]) for row in selected
        ),
        "median_depletion_fraction": median(
            float(row["depletion_fraction"]) for row in selected
        ),
        "total_out_of_order_rejections": sum(
            int(row["responses_rejected_out_of_order"]) for row in selected
        ),
        "total_fallback_activations": sum(
            int(row["fallback_activations"]) for row in selected
        ),
    }


def _evaluate(
    protocol: TransportH1Protocol,
    aggregates: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str]:
    serialized = aggregates["serialized"]
    pipelined = aggregates["pipelined"]
    request_rate_ratio = float(pipelined["median_request_rate"]) / max(
        float(serialized["median_request_rate"]), 1e-12
    )
    serialized_depletion = float(serialized["median_depletion_fraction"])
    pipelined_depletion = float(pipelined["median_depletion_fraction"])
    depletion_reduction = (
        (serialized_depletion - pipelined_depletion) / serialized_depletion
        if serialized_depletion > 0
        else 0.0
    )
    gates = {
        "all_runs_completed": {
            "observed": bool(
                serialized["all_completed"] and pipelined["all_completed"]
            ),
            "required": True,
        },
        "request_rate_ratio": {
            "observed": request_rate_ratio,
            "required_minimum": float(
                protocol.gates["minimum_request_rate_ratio"]
            ),
        },
        "depletion_reduction_fraction": {
            "observed": depletion_reduction,
            "required_minimum": float(
                protocol.gates["minimum_depletion_reduction_fraction"]
            ),
        },
        "pipelined_depletion_fraction": {
            "observed": pipelined_depletion,
            "required_maximum": float(
                protocol.gates["maximum_pipelined_depletion_fraction"]
            ),
        },
        "out_of_order_rejections": {
            "observed": int(pipelined["total_out_of_order_rejections"]),
            "required_maximum": int(
                protocol.gates["maximum_out_of_order_rejections"]
            ),
        },
        "fallback_activations": {
            "observed": int(pipelined["total_fallback_activations"]),
            "required_maximum": int(
                protocol.gates["maximum_fallback_activations"]
            ),
        },
    }
    passed = (
        gates["all_runs_completed"]["observed"] is True
        and float(gates["request_rate_ratio"]["observed"])
        >= float(gates["request_rate_ratio"]["required_minimum"])
        and float(gates["depletion_reduction_fraction"]["observed"])
        >= float(gates["depletion_reduction_fraction"]["required_minimum"])
        and float(gates["pipelined_depletion_fraction"]["observed"])
        <= float(gates["pipelined_depletion_fraction"]["required_maximum"])
        and int(gates["out_of_order_rejections"]["observed"])
        <= int(gates["out_of_order_rejections"]["required_maximum"])
        and int(gates["fallback_activations"]["observed"])
        <= int(gates["fallback_activations"]["required_maximum"])
    )
    return gates, "PASS" if passed else "FAIL"


def run_transport_h1(protocol_path: Path | str) -> dict[str, Any]:
    path = Path(protocol_path).resolve()
    protocol = load_transport_h1_protocol(path)
    rows: list[dict[str, Any]] = []
    for repetition in range(protocol.repetitions):
        for variant in ("serialized", "pipelined"):
            rows.append(
                _run_variant(
                    protocol,
                    variant=variant,
                    repetition=repetition,
                )
            )
    aggregates = {
        variant: _aggregate(rows, variant)
        for variant in ("serialized", "pipelined")
    }
    gates, verdict = _evaluate(protocol, aggregates)
    return {
        "schema_version": 1,
        "hypothesis_id": "transport_h1_serialized_delivery",
        "verdict": verdict,
        "protocol_path": str(path),
        "protocol_sha256": _sha256_json(protocol.raw),
        "source_commit": _git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": rows,
        "aggregates": aggregates,
        "decision_gates": gates,
        "claim_boundary": (
            "A fixture-level transport scheduling result; not learned-policy, "
            "GPU, remote-service soak, or real-robot evidence."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_transport_h1(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"verdict": result["verdict"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
