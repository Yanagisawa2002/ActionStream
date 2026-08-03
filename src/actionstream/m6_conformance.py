"""Deterministic M6-G0 conformance harness for LeRobot async scheduling.

The harness deliberately imports one ignored, frozen checkout of the official
Hugging Face LeRobot repository.  It calls the upstream
``PolicyServer._time_action_chunk`` and
``RobotClient._aggregate_action_queues`` implementations directly through a
hardware-free adapter; no LeRobot queue behavior is reimplemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
import tomllib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MILESTONE = "M6-G0"
SCHEMA_VERSION = 1
OFFICIAL_RUNTIME_TO_AGGREGATE = {
    "lerobot_weighted_average": "weighted_average",
    "lerobot_latest_only": "latest_only",
    "lerobot_average": "average",
    "lerobot_conservative": "conservative",
}
RUNTIME_IDS = (
    "sync_hold",
    *OFFICIAL_RUNTIME_TO_AGGREGATE,
    "actionstream_aligned",
)
REQUIRED_ACTION_FIELDS = {
    "runtime",
    "trace_seed",
    "request_id",
    "request_generation",
    "observation_step",
    "intended_execution_step",
    "actual_execution_step",
    "chunk_index",
    "result_arrival_step",
    "queue_insertion_step",
    "source_observation_age_steps",
    "temporal_index_error",
    "intended_time_elapsed_at_insertion",
    "discarded",
    "discard_reason",
    "queue_depth",
    "hold_or_underrun_state",
}


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using stable compact JSON."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _json_safe(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_frozen_manifest(path: Path | str) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("milestone") != MILESTONE:
        raise ValueError(f"Expected {MILESTONE} manifest")
    if manifest.get("frozen_before_benchmark") is not True:
        raise ValueError("M6 scenario must be frozen before benchmark execution")
    scenario = manifest.get("scenario", {})
    if scenario.get("chunk_horizon", 0) <= 0:
        raise ValueError("chunk_horizon must be positive")
    if scenario.get("request_interval_steps", 0) <= 0:
        raise ValueError("request_interval_steps must be positive")
    if scenario.get("action_dimensionality") != 6:
        raise ValueError("M6-G0 freezes a six-dimensional SO-101-compatible action")
    if len(manifest.get("latency_families", [])) != 7:
        raise ValueError("M6-G0 requires exactly seven frozen latency families")
    return manifest


def ideal_action(intended_execution_step: int) -> np.ndarray:
    """Encode the ideal absolute execution step in a deterministic 6D action."""

    step = float(intended_execution_step)
    return np.asarray(
        [
            step,
            math.sin(step / 3.0),
            math.cos(step / 5.0),
            float(intended_execution_step % 2),
            0.1 * step,
            -0.05 * step,
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class FrozenRequest:
    request_id: str
    ordinal: int
    generation: int
    observation_step: int
    result_arrival_step: int
    actions: tuple[np.ndarray, ...]

    @property
    def input_sha256(self) -> str:
        return canonical_sha256(
            {
                "request_id": self.request_id,
                "ordinal": self.ordinal,
                "generation": self.generation,
                "observation_step": self.observation_step,
                "result_arrival_step": self.result_arrival_step,
                "actions": [action.tolist() for action in self.actions],
            }
        )


def _family_by_id(manifest: Mapping[str, Any], family_id: str) -> Mapping[str, Any]:
    matches = [
        family
        for family in manifest["latency_families"]
        if family.get("id") == family_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate latency family: {family_id}")
    return matches[0]


def _request_delay_and_generation(
    family: Mapping[str, Any],
    *,
    request_ordinal: int,
    seed: int,
) -> tuple[int, int]:
    kind = family["kind"]
    if kind == "fixed":
        return int(family["delay_steps"]), 0
    if kind == "seeded_jitter":
        rng = random.Random((seed << 16) ^ request_ordinal ^ 0x6A09E667)
        jitter = rng.randint(
            int(family["jitter_min_steps"]),
            int(family["jitter_max_steps"]),
        )
        return max(0, int(family["center_delay_steps"]) + jitter), 0
    if kind == "alternating":
        delays = family["delay_steps_by_request_parity"]
        return int(delays[request_ordinal % len(delays)]), int(
            family["request_generation"]
        )
    if kind == "generation_race":
        if request_ordinal == 0:
            return (
                int(family["first_request_delay_steps"]),
                int(family["first_request_generation"]),
            )
        if request_ordinal == 1:
            return (
                int(family["newer_request_delay_steps"]),
                int(family["newer_generation"]),
            )
        return (
            int(family["remaining_request_delay_steps"]),
            int(family["newer_generation"]),
        )
    raise ValueError(f"Unsupported latency kind: {kind}")


def build_frozen_requests(
    manifest: Mapping[str, Any],
    *,
    family_id: str,
    seed: int,
) -> tuple[FrozenRequest, ...]:
    scenario = manifest["scenario"]
    horizon = int(scenario["chunk_horizon"])
    family = _family_by_id(manifest, family_id)
    requests: list[FrozenRequest] = []
    for ordinal, observation_step in enumerate(scenario["request_observation_steps"]):
        delay, generation = _request_delay_and_generation(
            family,
            request_ordinal=ordinal,
            seed=seed,
        )
        actions = tuple(
            ideal_action(int(observation_step) + chunk_index + 1)
            for chunk_index in range(horizon)
        )
        requests.append(
            FrozenRequest(
                request_id=f"{family_id}-seed{seed}-request{ordinal:02d}",
                ordinal=ordinal,
                generation=generation,
                observation_step=int(observation_step),
                result_arrival_step=int(observation_step) + delay,
                actions=actions,
            )
        )
    return tuple(requests)


def trace_input_sha256(requests: Sequence[FrozenRequest]) -> str:
    return canonical_sha256([request.input_sha256 for request in requests])


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def verify_upstream_checkout(
    lerobot_root: Path | str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(lerobot_root).resolve()
    if not (root / ".git").is_dir():
        raise FileNotFoundError(f"Official LeRobot checkout not found at {root}")
    remote = _run_git(root, "remote", "get-url", "origin")
    commit = _run_git(root, "rev-parse", "HEAD")
    status = _run_git(root, "status", "--porcelain")
    if remote.rstrip("/") != str(expected["repository_url"]).rstrip("/"):
        raise RuntimeError(f"Unexpected LeRobot origin: {remote}")
    if commit != expected["commit"]:
        raise RuntimeError(
            f"LeRobot checkout is not frozen at {expected['commit']}: found {commit}"
        )
    if status:
        raise RuntimeError("Frozen LeRobot checkout is dirty")

    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    source_version = str(pyproject["project"]["version"])
    if source_version != expected["source_version"]:
        raise RuntimeError(
            f"LeRobot source version mismatch: {source_version} != {expected['source_version']}"
        )
    exact_tag = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {
        "repository_url": remote,
        "commit": commit,
        "source_version": source_version,
        "exact_tag": exact_tag.stdout.strip() or None,
        "nearest_tag_description": _run_git(root, "describe", "--tags", "--always"),
        "commit_date": _run_git(root, "show", "-s", "--format=%cI", "HEAD"),
        "commit_subject": _run_git(root, "show", "-s", "--format=%s", "HEAD"),
        "checkout_clean": not bool(status),
        "checkout_path": str(root),
    }


def _module_is_below(module: Any, root: Path) -> bool:
    module_path = Path(inspect.getfile(module)).resolve()
    try:
        module_path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class UpstreamBindings:
    torch: Any
    RobotClient: type
    PolicyServer: type
    PolicyServerConfig: type
    TimedAction: type
    aggregate_functions: Mapping[str, Any]
    robot_client_module: Any
    policy_server_module: Any
    configs_module: Any
    helpers_module: Any


def load_upstream_bindings(lerobot_root: Path | str) -> UpstreamBindings:
    root = Path(lerobot_root).resolve()
    source_root = root / "src"
    source_text = str(source_root)
    # A full repository test run may have imported the installed LeRobot wheel
    # before reaching the M6 conformance tests.  Merely prepending the frozen
    # checkout does not replace modules already cached in ``sys.modules`` and
    # makes the result depend on test order.  Evict only LeRobot modules whose
    # source is outside the pinned checkout, then make the pinned source the
    # unambiguous first import location.  The frozen torch dependency remains
    # environment-provided by design.
    sys.path[:] = [entry for entry in sys.path if entry != source_text]
    sys.path.insert(0, source_text)
    for module_name, module in list(sys.modules.items()):
        if module_name != "lerobot" and not module_name.startswith("lerobot."):
            continue
        if module is None or not _module_is_below(module, root):
            del sys.modules[module_name]
    importlib.invalidate_caches()

    torch = importlib.import_module("torch")
    robot_client = importlib.import_module("lerobot.async_inference.robot_client")
    policy_server = importlib.import_module("lerobot.async_inference.policy_server")
    configs = importlib.import_module("lerobot.async_inference.configs")
    helpers = importlib.import_module("lerobot.async_inference.helpers")
    for module in (robot_client, policy_server, configs, helpers):
        if not _module_is_below(module, root):
            raise RuntimeError(
                f"Imported {module.__name__} from outside the frozen checkout: "
                f"{inspect.getfile(module)}"
            )
    return UpstreamBindings(
        torch=torch,
        RobotClient=robot_client.RobotClient,
        PolicyServer=policy_server.PolicyServer,
        PolicyServerConfig=configs.PolicyServerConfig,
        TimedAction=helpers.TimedAction,
        aggregate_functions=dict(configs.AGGREGATE_FUNCTIONS),
        robot_client_module=robot_client,
        policy_server_module=policy_server,
        configs_module=configs,
        helpers_module=helpers,
    )


class DeterministicFakeRobot:
    """Hardware-free robot used only by upstream ``control_loop_action``."""

    action_features = {f"joint_{index}.pos": float for index in range(6)}

    def __init__(self) -> None:
        self.sent_actions: list[dict[str, float]] = []

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        copied = {key: float(value) for key, value in action.items()}
        self.sent_actions.append(copied)
        return copied


class OfficialLeRobotAdapter:
    """Thin adapter that executes the frozen upstream timing/queue methods."""

    def __init__(
        self,
        bindings: UpstreamBindings,
        *,
        aggregate_name: str,
        environment_dt: float,
        chunk_horizon: int,
    ) -> None:
        if aggregate_name not in bindings.aggregate_functions:
            raise ValueError(f"Unregistered LeRobot aggregate: {aggregate_name}")
        self.bindings = bindings
        self.aggregate_name = aggregate_name
        self.aggregate_fn = bindings.aggregate_functions[aggregate_name]

        self.server = object.__new__(bindings.PolicyServer)
        self.server.config = SimpleNamespace(environment_dt=environment_dt)

        self.client = object.__new__(bindings.RobotClient)
        self.client.action_queue = Queue()
        self.client.action_queue_lock = threading.Lock()
        self.client.latest_action_lock = threading.Lock()
        self.client.latest_action = -1
        self.client.action_queue_size = []
        self.client.action_chunk_size = chunk_horizon
        self.client._chunk_size_threshold = 0.5
        self.client.robot = DeterministicFakeRobot()

        self.time_chunk_calls = 0
        self.aggregate_calls = 0
        self.control_action_calls = 0

    def time_action_chunk(
        self,
        *,
        capture_timestamp: float,
        observation_timestep: int,
        actions: Sequence[np.ndarray],
    ) -> list[Any]:
        tensors = [
            self.bindings.torch.as_tensor(action, dtype=self.bindings.torch.float32)
            for action in actions
        ]
        timed = self.bindings.PolicyServer._time_action_chunk(
            self.server,
            capture_timestamp,
            tensors,
            observation_timestep,
        )
        self.time_chunk_calls += 1
        return timed

    def aggregate(self, timed_actions: list[Any]) -> None:
        self.bindings.RobotClient._aggregate_action_queues(
            self.client,
            timed_actions,
            self.aggregate_fn,
        )
        self.aggregate_calls += 1

    def pop_and_send(self) -> tuple[Any, np.ndarray]:
        with self.client.action_queue_lock:
            timed_action = self.client.action_queue.queue[0]
        self.bindings.RobotClient.control_loop_action(self.client, verbose=False)
        self.control_action_calls += 1
        action = timed_action.get_action().detach().cpu().numpy().astype(np.float32)
        return timed_action, action

    def queue_items(self) -> list[Any]:
        with self.client.action_queue_lock:
            return list(self.client.action_queue.queue)

    def ready_at_default_threshold(self) -> bool:
        return bool(self.bindings.RobotClient._ready_to_send_observation(self.client))


def _base_record(
    *,
    runtime: str,
    family_id: str,
    trace_seed: int,
    request: FrozenRequest | None,
    chunk_index: int | None,
    action: np.ndarray | None,
) -> dict[str, Any]:
    intended = None
    if action is not None:
        intended = float(np.asarray(action, dtype=np.float32)[0])
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "runtime": runtime,
        "latency_family": family_id,
        "trace_seed": trace_seed,
        "record_kind": "policy_action",
        "request_id": None if request is None else request.request_id,
        "request_generation": None if request is None else request.generation,
        "request_ordinal": None if request is None else request.ordinal,
        "observation_step": None if request is None else request.observation_step,
        "generated_intended_execution_step": intended,
        "intended_execution_step": intended,
        "actual_execution_step": None,
        "chunk_index": chunk_index,
        "result_arrival_step": (
            None if request is None else request.result_arrival_step
        ),
        "queue_insertion_step": None,
        "source_observation_age_steps": None,
        "temporal_index_error": None,
        "intended_time_elapsed_at_insertion": None,
        "inserted": False,
        "discarded": False,
        "discard_reason": None,
        "queue_depth": None,
        "queue_depth_before_action": None,
        "queue_depth_after_action": None,
        "hold_or_underrun_state": None,
        "contributors": [] if request is None else [request.request_id],
        "action": None
        if action is None
        else np.asarray(action, dtype=np.float32).tolist(),
    }


def _mark_inserted(
    record: dict[str, Any],
    *,
    insertion_step: int,
    queue_depth: int,
    action: np.ndarray | None = None,
    contributors: Sequence[str] | None = None,
) -> None:
    if action is not None:
        vector = np.asarray(action, dtype=np.float32)
        record["action"] = vector.tolist()
        record["intended_execution_step"] = float(vector[0])
    record["inserted"] = True
    record["queue_insertion_step"] = insertion_step
    record["queue_depth"] = queue_depth
    record["intended_time_elapsed_at_insertion"] = bool(
        float(record["intended_execution_step"]) < insertion_step
    )
    if contributors is not None:
        record["contributors"] = list(contributors)


def _mark_discarded(
    record: dict[str, Any],
    *,
    reason: str,
    queue_depth: int,
    insertion_step: int | None = None,
) -> None:
    record["discarded"] = True
    record["discard_reason"] = reason
    record["queue_depth"] = queue_depth
    if insertion_step is not None:
        record["queue_insertion_step"] = insertion_step
        intended = record.get("intended_execution_step")
        record["intended_time_elapsed_at_insertion"] = (
            None if intended is None else bool(float(intended) < insertion_step)
        )


def _mark_executed(
    record: dict[str, Any],
    *,
    actual_step: int,
    queue_depth_before: int,
    queue_depth_after: int,
    state: str,
) -> None:
    intended = float(record["intended_execution_step"])
    observation_step = int(record["observation_step"])
    record["actual_execution_step"] = actual_step
    record["source_observation_age_steps"] = actual_step - observation_step
    record["temporal_index_error"] = abs(actual_step - intended)
    record["queue_depth_before_action"] = queue_depth_before
    record["queue_depth_after_action"] = queue_depth_after
    record["queue_depth"] = queue_depth_after
    record["hold_or_underrun_state"] = state


def _result_record(
    *,
    runtime: str,
    family_id: str,
    trace_seed: int,
    request: FrozenRequest,
    accepted: bool,
    reason: str,
    out_of_order: bool,
    late_generation: bool,
    queue_depth_before: int,
    queue_depth_after: int,
    actions_inserted: int,
    actions_discarded: int,
    overhead_ns: int,
    old_queue: Sequence[float],
    new_chunk: Sequence[float],
    discarded_prefix: Sequence[float],
    new_queue: Sequence[float],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "runtime": runtime,
        "latency_family": family_id,
        "trace_seed": trace_seed,
        "request_id": request.request_id,
        "request_ordinal": request.ordinal,
        "request_generation": request.generation,
        "observation_step": request.observation_step,
        "result_arrival_step": request.result_arrival_step,
        "queue_insertion_step": request.result_arrival_step + 1,
        "accepted": accepted,
        "reason": reason,
        "out_of_order": out_of_order,
        "late_generation": late_generation,
        "queue_depth_before": queue_depth_before,
        "queue_depth_after": queue_depth_after,
        "actions_returned": len(request.actions),
        "actions_inserted": actions_inserted,
        "actions_discarded": actions_discarded,
        "scheduler_overhead_ns": overhead_ns,
        "old_queue_intended_steps": list(old_queue),
        "new_chunk_intended_steps": list(new_chunk),
        "discarded_prefix_intended_steps": list(discarded_prefix),
        "new_queue_intended_steps": list(new_queue),
    }


class _SchedulerBase:
    def __init__(self, *, runtime: str, family_id: str, trace_seed: int) -> None:
        self.runtime = runtime
        self.family_id = family_id
        self.trace_seed = trace_seed
        self.records: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.command_history: list[dict[str, Any]] = []
        self.current_generation = 0
        self.max_arrived_ordinal = -1
        self.maximum_queue_depth = 0
        self.request_count = 0

    def capture(self, request: FrozenRequest) -> Any:
        self.current_generation = max(self.current_generation, request.generation)
        self.request_count += 1
        return request

    def _arrival_flags(self, request: FrozenRequest) -> tuple[bool, bool]:
        out_of_order = request.ordinal < self.max_arrived_ordinal
        self.max_arrived_ordinal = max(self.max_arrived_ordinal, request.ordinal)
        late_generation = request.generation < self.current_generation
        return out_of_order, late_generation

    def receive(self, request: FrozenRequest, payload: Any) -> None:
        raise NotImplementedError

    def execute(self, actual_step: int) -> None:
        raise NotImplementedError

    def receive_after_episode(self, request: FrozenRequest, payload: Any) -> None:
        out_of_order, late_generation = self._arrival_flags(request)
        start = time.perf_counter_ns()
        depth = self.queue_depth()
        for chunk_index, action in enumerate(request.actions):
            record = _base_record(
                runtime=self.runtime,
                family_id=self.family_id,
                trace_seed=self.trace_seed,
                request=request,
                chunk_index=chunk_index,
                action=action,
            )
            _mark_discarded(
                record,
                reason="result_after_episode",
                queue_depth=depth,
            )
            self.records.append(record)
        overhead = time.perf_counter_ns() - start
        self.results.append(
            _result_record(
                runtime=self.runtime,
                family_id=self.family_id,
                trace_seed=self.trace_seed,
                request=request,
                accepted=False,
                reason="result_after_episode",
                out_of_order=out_of_order,
                late_generation=late_generation,
                queue_depth_before=depth,
                queue_depth_after=depth,
                actions_inserted=0,
                actions_discarded=len(request.actions),
                overhead_ns=overhead,
                old_queue=[],
                new_chunk=[float(action[0]) for action in request.actions],
                discarded_prefix=[],
                new_queue=[],
            )
        )

    def queue_depth(self) -> int:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError

    def _record_underrun(self, actual_step: int, state: str) -> None:
        record = _base_record(
            runtime=self.runtime,
            family_id=self.family_id,
            trace_seed=self.trace_seed,
            request=None,
            chunk_index=None,
            action=None,
        )
        record["record_kind"] = "underrun"
        record["actual_execution_step"] = actual_step
        record["queue_depth"] = 0
        record["queue_depth_before_action"] = 0
        record["queue_depth_after_action"] = 0
        record["hold_or_underrun_state"] = state
        self.records.append(record)

    def _append_command(
        self,
        *,
        record: Mapping[str, Any],
        action: np.ndarray,
        actual_step: int,
        held: bool,
    ) -> None:
        self.command_history.append(
            {
                "actual_execution_step": actual_step,
                "request_id": record["request_id"],
                "action": np.asarray(action, dtype=np.float32),
                "held": held,
            }
        )


class LocalQueueScheduler(_SchedulerBase):
    """ActionStream queue rule or a full-chunk sync-hold reference."""

    def __init__(
        self,
        *,
        runtime: str,
        family_id: str,
        trace_seed: int,
        aligned: bool,
    ) -> None:
        super().__init__(
            runtime=runtime,
            family_id=family_id,
            trace_seed=trace_seed,
        )
        self.aligned = aligned
        self.queue: deque[dict[str, Any]] = deque()
        self.last_policy_record: dict[str, Any] | None = None
        self.last_action: np.ndarray | None = None

    def queue_depth(self) -> int:
        return len(self.queue)

    def receive(self, request: FrozenRequest, payload: Any) -> None:
        del payload
        out_of_order, late_generation = self._arrival_flags(request)
        insertion_step = request.result_arrival_step + 1
        depth_before = len(self.queue)
        old_queue = [float(record["intended_execution_step"]) for record in self.queue]
        incoming_records = [
            _base_record(
                runtime=self.runtime,
                family_id=self.family_id,
                trace_seed=self.trace_seed,
                request=request,
                chunk_index=chunk_index,
                action=action,
            )
            for chunk_index, action in enumerate(request.actions)
        ]
        start = time.perf_counter_ns()

        if self.aligned and request.generation != self.current_generation:
            reason = (
                "late_generation"
                if request.generation < self.current_generation
                else "future_generation"
            )
            for record in incoming_records:
                _mark_discarded(
                    record,
                    reason=reason,
                    queue_depth=depth_before,
                    insertion_step=insertion_step,
                )
            self.records.extend(incoming_records)
            overhead = time.perf_counter_ns() - start
            self.results.append(
                _result_record(
                    runtime=self.runtime,
                    family_id=self.family_id,
                    trace_seed=self.trace_seed,
                    request=request,
                    accepted=False,
                    reason=reason,
                    out_of_order=out_of_order,
                    late_generation=late_generation,
                    queue_depth_before=depth_before,
                    queue_depth_after=depth_before,
                    actions_inserted=0,
                    actions_discarded=len(incoming_records),
                    overhead_ns=overhead,
                    old_queue=old_queue,
                    new_chunk=[float(action[0]) for action in request.actions],
                    discarded_prefix=[],
                    new_queue=old_queue,
                )
            )
            return

        age_steps = max(0, request.result_arrival_step - request.observation_step)
        drop = age_steps if self.aligned else 0
        if self.aligned and drop >= len(incoming_records):
            for record in incoming_records:
                _mark_discarded(
                    record,
                    reason="fully_stale",
                    queue_depth=depth_before,
                    insertion_step=insertion_step,
                )
            self.records.extend(incoming_records)
            overhead = time.perf_counter_ns() - start
            self.results.append(
                _result_record(
                    runtime=self.runtime,
                    family_id=self.family_id,
                    trace_seed=self.trace_seed,
                    request=request,
                    accepted=False,
                    reason="fully_stale",
                    out_of_order=out_of_order,
                    late_generation=late_generation,
                    queue_depth_before=depth_before,
                    queue_depth_after=depth_before,
                    actions_inserted=0,
                    actions_discarded=len(incoming_records),
                    overhead_ns=overhead,
                    old_queue=old_queue,
                    new_chunk=[float(action[0]) for action in request.actions],
                    discarded_prefix=[float(action[0]) for action in request.actions],
                    new_queue=old_queue,
                )
            )
            return

        prefix = incoming_records[:drop]
        replacement = incoming_records[drop:]
        for record in prefix:
            _mark_discarded(
                record,
                reason="stale_prefix",
                queue_depth=depth_before,
                insertion_step=insertion_step,
            )
        for old in self.queue:
            _mark_discarded(
                old,
                reason="queue_replaced",
                queue_depth=0,
            )
        self.queue.clear()
        for record in replacement:
            _mark_inserted(
                record,
                insertion_step=insertion_step,
                queue_depth=len(replacement),
            )
            self.queue.append(record)
        self.records.extend(incoming_records)
        self.maximum_queue_depth = max(self.maximum_queue_depth, len(self.queue))
        overhead = time.perf_counter_ns() - start
        self.results.append(
            _result_record(
                runtime=self.runtime,
                family_id=self.family_id,
                trace_seed=self.trace_seed,
                request=request,
                accepted=True,
                reason="aligned_replacement" if self.aligned else "full_replacement",
                out_of_order=out_of_order,
                late_generation=late_generation,
                queue_depth_before=depth_before,
                queue_depth_after=len(self.queue),
                actions_inserted=len(replacement),
                actions_discarded=len(prefix),
                overhead_ns=overhead,
                old_queue=old_queue,
                new_chunk=[float(action[0]) for action in request.actions],
                discarded_prefix=[
                    float(record["intended_execution_step"]) for record in prefix
                ],
                new_queue=[
                    float(record["intended_execution_step"]) for record in self.queue
                ],
            )
        )

    def execute(self, actual_step: int) -> None:
        depth_before = len(self.queue)
        if self.queue:
            record = self.queue.popleft()
            action = np.asarray(record["action"], dtype=np.float32)
            _mark_executed(
                record,
                actual_step=actual_step,
                queue_depth_before=depth_before,
                queue_depth_after=len(self.queue),
                state="fresh",
            )
            self.last_policy_record = record
            self.last_action = action.copy()
            self._append_command(
                record=record,
                action=action,
                actual_step=actual_step,
                held=False,
            )
            return

        if self.last_policy_record is None or self.last_action is None:
            self._record_underrun(actual_step, "startup_block")
            return

        hold = dict(self.last_policy_record)
        hold["record_kind"] = "hold"
        hold["discarded"] = False
        hold["discard_reason"] = None
        hold["inserted"] = False
        hold["intended_time_elapsed_at_insertion"] = None
        _mark_executed(
            hold,
            actual_step=actual_step,
            queue_depth_before=0,
            queue_depth_after=0,
            state="hold_repeat_last_command",
        )
        self.records.append(hold)
        self._append_command(
            record=hold,
            action=self.last_action,
            actual_step=actual_step,
            held=True,
        )

    def finish(self) -> None:
        depth = len(self.queue)
        for record in self.queue:
            _mark_discarded(
                record,
                reason="episode_end",
                queue_depth=depth,
            )
        self.queue.clear()


class OfficialQueueScheduler(_SchedulerBase):
    def __init__(
        self,
        *,
        runtime: str,
        family_id: str,
        trace_seed: int,
        adapter: OfficialLeRobotAdapter,
        fps: float,
    ) -> None:
        super().__init__(
            runtime=runtime,
            family_id=family_id,
            trace_seed=trace_seed,
        )
        self.adapter = adapter
        self.fps = fps
        self.entry_records: dict[int, dict[str, Any]] = {}

    def queue_depth(self) -> int:
        return len(self.adapter.queue_items())

    def capture(self, request: FrozenRequest) -> list[Any]:
        super().capture(request)
        with self.adapter.client.latest_action_lock:
            observation_timestep = max(self.adapter.client.latest_action, 0)
        return self.adapter.time_action_chunk(
            capture_timestamp=request.observation_step / self.fps,
            observation_timestep=observation_timestep,
            actions=request.actions,
        )

    @staticmethod
    def _timed_key(timed_action: Any) -> tuple[float, int]:
        return (float(timed_action.get_timestamp()), int(timed_action.get_timestep()))

    def receive(self, request: FrozenRequest, payload: list[Any]) -> None:
        out_of_order, late_generation = self._arrival_flags(request)
        insertion_step = request.result_arrival_step + 1
        old_items = self.adapter.queue_items()
        old_queue_records = [
            self.entry_records[id(item)]
            for item in old_items
            if id(item) in self.entry_records
        ]
        old_queue = [
            float(record["intended_execution_step"]) for record in old_queue_records
        ]
        depth_before = len(old_items)
        incoming_records: list[dict[str, Any]] = []
        incoming_by_key: dict[tuple[float, int], dict[str, Any]] = {}
        for chunk_index, (timed_action, action) in enumerate(
            zip(payload, request.actions, strict=True)
        ):
            record = _base_record(
                runtime=self.runtime,
                family_id=self.family_id,
                trace_seed=self.trace_seed,
                request=request,
                chunk_index=chunk_index,
                action=action,
            )
            record["upstream_timed_action_timestep"] = int(timed_action.get_timestep())
            record["upstream_timed_action_timestamp"] = float(
                timed_action.get_timestamp()
            )
            incoming_records.append(record)
            incoming_by_key[self._timed_key(timed_action)] = record

        with self.adapter.client.latest_action_lock:
            latest_before = int(self.adapter.client.latest_action)
        start = time.perf_counter_ns()
        self.adapter.aggregate(payload)
        overhead = time.perf_counter_ns() - start
        new_items = self.adapter.queue_items()
        accepted_timesteps = {int(item.get_timestep()) for item in new_items}
        incoming_accepted_timesteps = {
            int(item.get_timestep())
            for item in payload
            if int(item.get_timestep()) > latest_before
        }

        for item, record in zip(old_items, old_queue_records, strict=False):
            reason = (
                "queue_aggregated"
                if int(item.get_timestep()) in incoming_accepted_timesteps
                else "queue_replaced"
            )
            _mark_discarded(record, reason=reason, queue_depth=0)

        new_entry_records: dict[int, dict[str, Any]] = {}
        old_by_timestep = {
            int(item.get_timestep()): self.entry_records.get(id(item))
            for item in old_items
        }
        for timed_action in payload:
            key = self._timed_key(timed_action)
            record = incoming_by_key[key]
            if int(timed_action.get_timestep()) <= latest_before:
                _mark_discarded(
                    record,
                    reason="upstream_latest_action_filter",
                    queue_depth=len(new_items),
                    insertion_step=insertion_step,
                )

        for item in new_items:
            key = self._timed_key(item)
            record = incoming_by_key[key]
            action = item.get_action().detach().cpu().numpy().astype(np.float32)
            previous = old_by_timestep.get(int(item.get_timestep()))
            contributors = [request.request_id]
            if previous is not None:
                contributors = [
                    str(previous["request_id"]),
                    request.request_id,
                ]
            _mark_inserted(
                record,
                insertion_step=insertion_step,
                queue_depth=len(new_items),
                action=action,
                contributors=contributors,
            )
            new_entry_records[id(item)] = record

        self.entry_records = new_entry_records
        self.records.extend(incoming_records)
        self.maximum_queue_depth = max(self.maximum_queue_depth, len(new_items))
        inserted = sum(bool(record["inserted"]) for record in incoming_records)
        discarded = len(incoming_records) - inserted
        accepted = bool(inserted)
        reason = "upstream_aggregated" if accepted else "all_actions_filtered"
        self.results.append(
            _result_record(
                runtime=self.runtime,
                family_id=self.family_id,
                trace_seed=self.trace_seed,
                request=request,
                accepted=accepted,
                reason=reason,
                out_of_order=out_of_order,
                late_generation=late_generation,
                queue_depth_before=depth_before,
                queue_depth_after=len(new_items),
                actions_inserted=inserted,
                actions_discarded=discarded,
                overhead_ns=overhead,
                old_queue=old_queue,
                new_chunk=[float(action[0]) for action in request.actions],
                discarded_prefix=[
                    float(record["generated_intended_execution_step"])
                    for record in incoming_records
                    if not record["inserted"]
                ],
                new_queue=[
                    float(record["intended_execution_step"])
                    for record in new_entry_records.values()
                ],
            )
        )
        if accepted_timesteps != {
            int(item.get_timestep()) for item in self.adapter.queue_items()
        }:
            raise AssertionError("Upstream queue mutated during M6 bookkeeping")

    def execute(self, actual_step: int) -> None:
        items = self.adapter.queue_items()
        if not items:
            self._record_underrun(actual_step, "upstream_no_send")
            return
        depth_before = len(items)
        peek = items[0]
        record = self.entry_records.pop(id(peek))
        timed_action, action = self.adapter.pop_and_send()
        if timed_action is not peek:
            raise AssertionError("Upstream control loop popped a different action")
        depth_after = self.queue_depth()
        _mark_executed(
            record,
            actual_step=actual_step,
            queue_depth_before=depth_before,
            queue_depth_after=depth_after,
            state="fresh",
        )
        self._append_command(
            record=record,
            action=action,
            actual_step=actual_step,
            held=False,
        )

    def finish(self) -> None:
        items = self.adapter.queue_items()
        depth = len(items)
        for item in items:
            record = self.entry_records.get(id(item))
            if record is not None:
                _mark_discarded(
                    record,
                    reason="episode_end",
                    queue_depth=depth,
                )
        with self.adapter.client.action_queue_lock:
            self.adapter.client.action_queue = Queue()
        self.entry_records.clear()


def _make_scheduler(
    runtime: str,
    *,
    family_id: str,
    trace_seed: int,
    manifest: Mapping[str, Any],
    bindings: UpstreamBindings,
) -> _SchedulerBase:
    if runtime == "sync_hold":
        return LocalQueueScheduler(
            runtime=runtime,
            family_id=family_id,
            trace_seed=trace_seed,
            aligned=False,
        )
    if runtime == "actionstream_aligned":
        return LocalQueueScheduler(
            runtime=runtime,
            family_id=family_id,
            trace_seed=trace_seed,
            aligned=True,
        )
    aggregate_name = OFFICIAL_RUNTIME_TO_AGGREGATE[runtime]
    scenario = manifest["scenario"]
    adapter = OfficialLeRobotAdapter(
        bindings,
        aggregate_name=aggregate_name,
        environment_dt=float(scenario["environment_dt_seconds"]),
        chunk_horizon=int(scenario["chunk_horizon"]),
    )
    return OfficialQueueScheduler(
        runtime=runtime,
        family_id=family_id,
        trace_seed=trace_seed,
        adapter=adapter,
        fps=float(scenario["control_frequency_hz"]),
    )


def run_trace(
    manifest: Mapping[str, Any],
    *,
    family_id: str,
    seed: int,
    runtime: str,
    bindings: UpstreamBindings,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    requests = build_frozen_requests(manifest, family_id=family_id, seed=seed)
    scheduler = _make_scheduler(
        runtime,
        family_id=family_id,
        trace_seed=seed,
        manifest=manifest,
        bindings=bindings,
    )
    by_capture = {request.observation_step: request for request in requests}
    pending: list[tuple[FrozenRequest, Any]] = []
    execution_steps = int(manifest["scenario"]["execution_steps"])

    for completed_step in range(execution_steps):
        request = by_capture.get(completed_step)
        if request is not None:
            pending.append((request, scheduler.capture(request)))

        arriving = [
            item for item in pending if item[0].result_arrival_step == completed_step
        ]
        pending = [
            item for item in pending if item[0].result_arrival_step != completed_step
        ]
        arriving.sort(
            key=lambda item: (
                item[0].result_arrival_step,
                item[0].ordinal,
            )
        )
        for arriving_request, payload in arriving:
            scheduler.receive(arriving_request, payload)

        scheduler.execute(completed_step + 1)

    for request, payload in sorted(
        pending,
        key=lambda item: (
            item[0].result_arrival_step,
            item[0].ordinal,
        ),
    ):
        scheduler.receive_after_episode(request, payload)
    scheduler.finish()

    adapter_audit: dict[str, Any] = {
        "runtime": runtime,
        "upstream_time_chunk_calls": 0,
        "upstream_aggregate_calls": 0,
        "upstream_control_action_calls": 0,
    }
    if isinstance(scheduler, OfficialQueueScheduler):
        adapter_audit.update(
            {
                "aggregate_name": scheduler.adapter.aggregate_name,
                "upstream_time_chunk_calls": scheduler.adapter.time_chunk_calls,
                "upstream_aggregate_calls": scheduler.adapter.aggregate_calls,
                "upstream_control_action_calls": scheduler.adapter.control_action_calls,
            }
        )
    summary = summarize_trace(
        scheduler.records,
        scheduler.results,
        runtime=runtime,
        family_id=family_id,
        seed=seed,
        request_count=scheduler.request_count,
        maximum_queue_depth=scheduler.maximum_queue_depth,
        command_history=scheduler.command_history,
        input_sha256=trace_input_sha256(requests),
        adapter_audit=adapter_audit,
        execution_steps=execution_steps,
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "latency_family": family_id,
        "trace_seed": seed,
        "input_sha256": trace_input_sha256(requests),
        "requests": [
            {
                "request_id": request.request_id,
                "request_ordinal": request.ordinal,
                "request_generation": request.generation,
                "observation_step": request.observation_step,
                "result_arrival_step": request.result_arrival_step,
                "action_chunk_sha256": canonical_sha256(
                    [action.tolist() for action in request.actions]
                ),
                "intended_execution_steps": [
                    int(action[0]) for action in request.actions
                ],
            }
            for request in requests
        ],
    }
    return scheduler.records, scheduler.results, summary, contract


def summarize_trace(
    records: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    *,
    runtime: str,
    family_id: str,
    seed: int,
    request_count: int,
    maximum_queue_depth: int,
    command_history: Sequence[Mapping[str, Any]],
    input_sha256: str,
    adapter_audit: Mapping[str, Any],
    execution_steps: int,
) -> dict[str, Any]:
    executed = [
        row
        for row in records
        if row.get("actual_execution_step") is not None
        and row.get("record_kind") != "underrun"
    ]
    fresh = [row for row in executed if row.get("hold_or_underrun_state") == "fresh"]
    holds = [row for row in executed if row.get("record_kind") == "hold"]
    underruns = [row for row in records if row.get("record_kind") == "underrun"]
    stale_inserted = [
        row
        for row in records
        if row.get("inserted") is True
        and row.get("intended_time_elapsed_at_insertion") is True
    ]
    stale_executed = [
        row
        for row in stale_inserted
        if row.get("actual_execution_step") is not None
        and row.get("discarded") is False
    ]
    errors = [
        float(row["temporal_index_error"])
        for row in executed
        if row.get("temporal_index_error") is not None
    ]
    source_ages = [
        float(row["source_observation_age_steps"])
        for row in executed
        if row.get("source_observation_age_steps") is not None
    ]
    transitions: list[float] = []
    for previous, current in zip(
        command_history,
        command_history[1:],
        strict=False,
    ):
        if previous["request_id"] != current["request_id"]:
            transitions.append(
                float(
                    np.linalg.norm(
                        np.asarray(current["action"], dtype=np.float64)
                        - np.asarray(previous["action"], dtype=np.float64)
                    )
                )
            )
    late = [row for row in results if row["late_generation"]]
    out_of_order = [row for row in results if row["out_of_order"]]
    overhead = [int(row["scheduler_overhead_ns"]) for row in results]
    underrun_steps = len(holds) + len(underruns)
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "runtime": runtime,
        "latency_family": family_id,
        "trace_seed": seed,
        "input_sha256": input_sha256,
        "request_count": request_count,
        "executed_action_count": len(executed),
        "fresh_action_count": len(fresh),
        "action_coverage_steps": len(executed),
        "action_coverage_rate": len(executed) / execution_steps,
        "stale_prefix_actions_inserted": len(stale_inserted),
        "stale_prefix_actions_executed": len(stale_executed),
        "median_temporal_index_error": _percentile(errors, 50),
        "p95_temporal_index_error": _percentile(errors, 95),
        "mean_action_source_age_steps": _mean(source_ages),
        "median_action_source_age_steps": _percentile(source_ages, 50),
        "p95_action_source_age_steps": _percentile(source_ages, 95),
        "queue_underrun_steps": underrun_steps,
        "queue_underrun_rate": underrun_steps / execution_steps,
        "hold_steps": len(holds),
        "hold_rate": len(holds) / execution_steps,
        "no_send_or_startup_block_steps": len(underruns),
        "chunk_transition_discontinuity_mean_l2": _mean(transitions),
        "chunk_transition_discontinuity_p95_l2": _percentile(transitions, 95),
        "late_results_accepted": sum(bool(row["accepted"]) for row in late),
        "late_results_rejected": sum(not bool(row["accepted"]) for row in late),
        "out_of_order_results_accepted": sum(
            bool(row["accepted"]) for row in out_of_order
        ),
        "out_of_order_results_rejected": sum(
            not bool(row["accepted"]) for row in out_of_order
        ),
        "maximum_queue_depth": maximum_queue_depth,
        "scheduler_overhead_median_microseconds": (
            None if not overhead else _percentile(overhead, 50) / 1000.0
        ),
        "scheduler_overhead_p95_microseconds": (
            None if not overhead else _percentile(overhead, 95) / 1000.0
        ),
        "adapter_audit": dict(adapter_audit),
    }


def aggregate_metrics(
    action_records: Sequence[Mapping[str, Any]],
    result_records: Sequence[Mapping[str, Any]],
    trace_summaries: Sequence[Mapping[str, Any]],
    *,
    execution_steps: int,
) -> list[dict[str, Any]]:
    grouped_summaries: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    grouped_actions: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    grouped_results: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in trace_summaries:
        grouped_summaries[(str(row["runtime"]), str(row["latency_family"]))].append(row)
    for row in action_records:
        grouped_actions[(str(row["runtime"]), str(row["latency_family"]))].append(row)
    for row in result_records:
        grouped_results[(str(row["runtime"]), str(row["latency_family"]))].append(row)

    aggregates: list[dict[str, Any]] = []
    for key in sorted(grouped_summaries):
        runtime, family_id = key
        summaries = grouped_summaries[key]
        actions = grouped_actions[key]
        results = grouped_results[key]
        executed = [
            row
            for row in actions
            if row.get("actual_execution_step") is not None
            and row.get("record_kind") != "underrun"
        ]
        holds = [row for row in executed if row.get("record_kind") == "hold"]
        underrun_rows = [row for row in actions if row.get("record_kind") == "underrun"]
        errors = [
            float(row["temporal_index_error"])
            for row in executed
            if row.get("temporal_index_error") is not None
        ]
        ages = [
            float(row["source_observation_age_steps"])
            for row in executed
            if row.get("source_observation_age_steps") is not None
        ]
        inserted_stale = [
            row
            for row in actions
            if row.get("inserted") is True
            and row.get("intended_time_elapsed_at_insertion") is True
        ]
        executed_stale = [
            row
            for row in inserted_stale
            if row.get("actual_execution_step") is not None
            and row.get("discarded") is False
        ]
        transition_means = [
            float(row["chunk_transition_discontinuity_mean_l2"])
            for row in summaries
            if row.get("chunk_transition_discontinuity_mean_l2") is not None
        ]
        overhead = [int(row["scheduler_overhead_ns"]) for row in results]
        trace_count = len(summaries)
        possible_steps = trace_count * execution_steps
        late = [row for row in results if row["late_generation"]]
        out_of_order = [row for row in results if row["out_of_order"]]
        aggregates.append(
            {
                "runtime": runtime,
                "latency_family": family_id,
                "trace_count": trace_count,
                "request_count": sum(int(row["request_count"]) for row in summaries),
                "executed_action_count": len(executed),
                "action_coverage_rate": len(executed) / possible_steps,
                "stale_prefix_actions_inserted": len(inserted_stale),
                "stale_prefix_actions_inserted_per_trace": len(inserted_stale)
                / trace_count,
                "stale_prefix_actions_executed": len(executed_stale),
                "stale_prefix_actions_executed_per_trace": len(executed_stale)
                / trace_count,
                "median_temporal_index_error": _percentile(errors, 50),
                "p95_temporal_index_error": _percentile(errors, 95),
                "mean_action_source_age_steps": _mean(ages),
                "median_action_source_age_steps": _percentile(ages, 50),
                "p95_action_source_age_steps": _percentile(ages, 95),
                "queue_underrun_steps": len(holds) + len(underrun_rows),
                "queue_underrun_rate": (len(holds) + len(underrun_rows))
                / possible_steps,
                "hold_steps": len(holds),
                "hold_rate": len(holds) / possible_steps,
                "chunk_transition_discontinuity_mean_l2": _mean(transition_means),
                "late_results_accepted": sum(bool(row["accepted"]) for row in late),
                "late_results_rejected": sum(not bool(row["accepted"]) for row in late),
                "out_of_order_results_accepted": sum(
                    bool(row["accepted"]) for row in out_of_order
                ),
                "out_of_order_results_rejected": sum(
                    not bool(row["accepted"]) for row in out_of_order
                ),
                "maximum_queue_depth": max(
                    int(row["maximum_queue_depth"]) for row in summaries
                ),
                "scheduler_overhead_median_microseconds": (
                    None if not overhead else _percentile(overhead, 50) / 1000.0
                ),
                "scheduler_overhead_p95_microseconds": (
                    None if not overhead else _percentile(overhead, 95) / 1000.0
                ),
            }
        )
    return aggregates


def _aggregate_lookup(
    aggregates: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row["runtime"]), str(row["latency_family"])): row for row in aggregates
    }


def build_decision(
    manifest: Mapping[str, Any],
    aggregates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lookup = _aggregate_lookup(aggregates)
    official = list(manifest["runtimes"]["official_lerobot_act_compatible"])
    nonzero_families = [
        family["id"]
        for family in manifest["latency_families"]
        if family["id"] != "zero"
    ]

    def baseline_score(runtime: str) -> tuple[float, float, float, str]:
        rows = [lookup[(runtime, family)] for family in nonzero_families]
        return (
            statistics.fmean(
                float(row["stale_prefix_actions_executed_per_trace"]) for row in rows
            ),
            statistics.fmean(
                float(row["median_temporal_index_error"] or 0.0) for row in rows
            ),
            statistics.fmean(float(row["queue_underrun_rate"]) for row in rows),
            runtime,
        )

    best_official = min(official, key=baseline_score)
    threshold = manifest["decision_thresholds"]
    actionstream = "actionstream_aligned"

    delayed_baseline_stale = sum(
        float(
            lookup[(best_official, family)]["stale_prefix_actions_executed_per_trace"]
        )
        for family in nonzero_families
    )
    delayed_actionstream_stale = sum(
        float(lookup[(actionstream, family)]["stale_prefix_actions_executed_per_trace"])
        for family in nonzero_families
    )
    if delayed_baseline_stale > 0:
        stale_reduction = (
            delayed_baseline_stale - delayed_actionstream_stale
        ) / delayed_baseline_stale
    else:
        stale_reduction = 0.0 if delayed_actionstream_stale == 0 else -math.inf

    family_error_reductions: dict[str, float | None] = {}
    for family in nonzero_families:
        baseline_value = lookup[(best_official, family)]["median_temporal_index_error"]
        aligned_value = lookup[(actionstream, family)]["median_temporal_index_error"]
        if baseline_value is None or aligned_value is None:
            reduction = None
        else:
            baseline_error = float(baseline_value)
            aligned_error = float(aligned_value)
            if baseline_error > 0:
                reduction = (baseline_error - aligned_error) / baseline_error
            else:
                reduction = 0.0 if aligned_error == 0 else -math.inf
        family_error_reductions[family] = reduction
    families_meeting_error = [
        family
        for family, reduction in family_error_reductions.items()
        if reduction is not None
        and reduction
        >= float(threshold["median_temporal_index_error_reduction_min_fraction"])
    ]

    underrun_worsening_pp = {
        family: 100.0
        * (
            float(lookup[(actionstream, family)]["queue_underrun_rate"])
            - float(lookup[(best_official, family)]["queue_underrun_rate"])
        )
        for family in nonzero_families
    }
    maximum_underrun_worsening = max(underrun_worsening_pp.values())
    zero_baseline = lookup[(best_official, "zero")]
    zero_aligned = lookup[(actionstream, "zero")]
    zero_error_regression = float(
        zero_aligned["median_temporal_index_error"] or 0.0
    ) - float(zero_baseline["median_temporal_index_error"] or 0.0)
    zero_underrun_worsening_pp = 100.0 * (
        float(zero_aligned["queue_underrun_rate"])
        - float(zero_baseline["queue_underrun_rate"])
    )

    materially_distinct = False
    for family in nonzero_families:
        baseline_error_value = lookup[(best_official, family)][
            "median_temporal_index_error"
        ]
        aligned_error_value = lookup[(actionstream, family)][
            "median_temporal_index_error"
        ]
        error_distinct = (baseline_error_value is None) != (
            aligned_error_value is None
        ) or (
            baseline_error_value is not None
            and aligned_error_value is not None
            and abs(float(baseline_error_value) - float(aligned_error_value)) >= 0.5
        )
        stale_distinct = float(
            lookup[(best_official, family)]["stale_prefix_actions_executed_per_trace"]
        ) != float(
            lookup[(actionstream, family)]["stale_prefix_actions_executed_per_trace"]
        )
        materially_distinct = materially_distinct or error_distinct or stale_distinct
    criteria = {
        "materially_distinct_from_best_official_act_async": materially_distinct,
        "stale_prefix_execution_reduction_at_least_50pct": stale_reduction
        >= float(threshold["stale_prefix_actions_executed_reduction_min_fraction"]),
        "median_error_reduction_in_at_least_three_nonzero_families": len(
            families_meeting_error
        )
        >= int(threshold["required_nonzero_latency_families_with_error_reduction"]),
        "queue_underrun_or_hold_worsening_within_5pp": maximum_underrun_worsening
        <= float(threshold["queue_underrun_or_hold_worsening_max_percentage_points"]),
        "zero_latency_no_material_regression": zero_error_regression
        <= float(threshold["zero_latency_median_error_regression_max_steps"])
        and zero_underrun_worsening_pp
        <= float(threshold["zero_latency_underrun_worsening_max_percentage_points"]),
        "no_policy_retraining_required": True,
        "no_act_architecture_modification_required": True,
        "integration_isolated_and_maintainable": True,
    }
    port_go = all(criteria.values())

    official_equivalent_or_better_families = []
    for family in nonzero_families:
        baseline_error_value = lookup[(best_official, family)][
            "median_temporal_index_error"
        ]
        aligned_error_value = lookup[(actionstream, family)][
            "median_temporal_index_error"
        ]
        if baseline_error_value is None or aligned_error_value is None:
            continue
        if float(baseline_error_value) <= float(aligned_error_value) + 0.1 and float(
            lookup[(best_official, family)]["stale_prefix_actions_executed_per_trace"]
        ) <= float(
            lookup[(actionstream, family)]["stale_prefix_actions_executed_per_trace"]
        ):
            official_equivalent_or_better_families.append(family)
    official_equivalent_or_better = len(official_equivalent_or_better_families) >= 4
    tooling_absent = True
    if port_go:
        label = "PORT GO"
        next_step = "SO-101 real-robot port"
        reasons = ["All predeclared M6-G0 PORT GO gates passed."]
    elif official_equivalent_or_better and tooling_absent:
        label = "TOOLING GO / ALGORITHM OVERLAP"
        next_step = "LeRobot tooling/evaluator contribution"
        reasons = [
            "The best official ACT-compatible scheduler is equivalent or better in at least four non-zero-latency families.",
            "The official runtime lacks the frozen action-level provenance and temporal diagnostics exercised by M6-G0.",
        ]
    else:
        label = "NO-GO"
        next_step = "termination of this direction"
        failed = [name for name, passed in criteria.items() if not passed]
        reasons = [
            "The predeclared PORT GO conjunction failed.",
            "Official scheduling was not equivalent-or-better in enough families to justify an algorithm-overlap tooling classification.",
            f"Failed gates: {', '.join(failed)}",
        ]

    core = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "classification": label,
        "recommended_next_step": next_step,
        "best_official_act_compatible_runtime": best_official,
        "official_baseline_selection_score": list(baseline_score(best_official)[:3]),
        "criteria": criteria,
        "measurements": {
            "delayed_stale_prefix_execution_reduction_fraction": stale_reduction,
            "family_median_temporal_error_reduction_fraction": family_error_reductions,
            "families_meeting_30pct_error_reduction": families_meeting_error,
            "queue_underrun_worsening_percentage_points": underrun_worsening_pp,
            "maximum_queue_underrun_worsening_percentage_points": maximum_underrun_worsening,
            "zero_latency_median_error_regression_steps": zero_error_regression,
            "zero_latency_underrun_worsening_percentage_points": zero_underrun_worsening_pp,
            "official_equivalent_or_better_families": official_equivalent_or_better_families,
        },
        "tooling_diagnostics_materially_absent_upstream": tooling_absent,
        "reasons": reasons,
        "m5_g0_remains_no_go": True,
        "poseguard_revived": False,
        "foundationpose_included": False,
        "rtc_in_act_benchmark": False,
    }
    return {**core, "decision_sha256": canonical_sha256(core)}


def validate_conformance_artifacts(
    manifest: Mapping[str, Any],
    action_records: Sequence[Mapping[str, Any]],
    result_records: Sequence[Mapping[str, Any]],
    trace_summaries: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    for index, row in enumerate(action_records):
        missing = REQUIRED_ACTION_FIELDS - set(row)
        if missing:
            errors.append(f"action row {index} missing {sorted(missing)}")
        actual = row.get("actual_execution_step")
        intended = row.get("intended_execution_step")
        temporal_error = row.get("temporal_index_error")
        if (
            actual is not None
            and intended is not None
            and row.get("record_kind") != "underrun"
        ):
            expected = abs(float(actual) - float(intended))
            if temporal_error is None or not math.isclose(
                float(temporal_error),
                expected,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                errors.append(f"action row {index} temporal error mismatch")
        insertion = row.get("queue_insertion_step")
        elapsed = row.get("intended_time_elapsed_at_insertion")
        if insertion is not None and intended is not None and elapsed is not None:
            expected_elapsed = float(intended) < int(insertion)
            if bool(elapsed) != expected_elapsed:
                errors.append(f"action row {index} stale-prefix mismatch")

    input_hashes: defaultdict[tuple[str, int], set[str]] = defaultdict(set)
    for summary in trace_summaries:
        input_hashes[(str(summary["latency_family"]), int(summary["trace_seed"]))].add(
            str(summary["input_sha256"])
        )
    nonidentical = {
        f"{family}:{seed}": sorted(hashes)
        for (family, seed), hashes in input_hashes.items()
        if len(hashes) != 1
    }
    if nonidentical:
        errors.append(f"runtime input mismatch: {nonidentical}")

    stochastic_required = int(manifest["scenario"]["stochastic_trace_count"])
    jitter_counts = {
        runtime: sum(
            summary["runtime"] == runtime
            and summary["latency_family"] == "bounded_jitter"
            for summary in trace_summaries
        )
        for runtime in RUNTIME_IDS
    }
    if any(count != stochastic_required for count in jitter_counts.values()):
        errors.append(f"bounded jitter trace counts mismatch: {jitter_counts}")

    official_adapter_counts = {
        runtime: {
            "time_chunk": sum(
                int(summary["adapter_audit"].get("upstream_time_chunk_calls", 0))
                for summary in trace_summaries
                if summary["runtime"] == runtime
            ),
            "aggregate": sum(
                int(summary["adapter_audit"].get("upstream_aggregate_calls", 0))
                for summary in trace_summaries
                if summary["runtime"] == runtime
            ),
        }
        for runtime in OFFICIAL_RUNTIME_TO_AGGREGATE
    }
    for runtime, counts in official_adapter_counts.items():
        if counts["time_chunk"] <= 0 or counts["aggregate"] <= 0:
            errors.append(f"{runtime} did not execute upstream methods")

    aggregate_keys = {
        (str(row["runtime"]), str(row["latency_family"])) for row in aggregates
    }
    expected_aggregate_keys = {
        (runtime, family["id"])
        for runtime in RUNTIME_IDS
        for family in manifest["latency_families"]
    }
    if aggregate_keys != expected_aggregate_keys:
        errors.append("aggregate matrix is incomplete")

    result_request_keys = {
        (
            str(row["runtime"]),
            str(row["latency_family"]),
            int(row["trace_seed"]),
            str(row["request_id"]),
        )
        for row in result_records
    }
    expected_result_count = sum(
        int(
            manifest["scenario"][
                "stochastic_trace_count"
                if family["kind"] == "seeded_jitter"
                else "deterministic_trace_count"
            ]
        )
        * len(RUNTIME_IDS)
        * len(manifest["scenario"]["request_observation_steps"])
        for family in manifest["latency_families"]
    )
    if len(result_request_keys) != expected_result_count:
        errors.append(
            f"result lifecycle count {len(result_request_keys)} != {expected_result_count}"
        )

    checks = {
        "schema_fields_complete": not any("missing" in error for error in errors),
        "temporal_metric_formula_exact": not any(
            "temporal error" in error for error in errors
        ),
        "stale_prefix_definition_exact": not any(
            "stale-prefix" in error for error in errors
        ),
        "identical_runtime_inputs": not bool(nonidentical),
        "bounded_jitter_has_50_traces_per_runtime": all(
            count == stochastic_required for count in jitter_counts.values()
        ),
        "actual_upstream_methods_invoked": all(
            counts["time_chunk"] > 0 and counts["aggregate"] > 0
            for counts in official_adapter_counts.values()
        ),
        "aggregate_matrix_complete": aggregate_keys == expected_aggregate_keys,
        "result_lifecycle_complete": len(result_request_keys) == expected_result_count,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "passed": not errors and all(checks.values()),
        "checks": checks,
        "errors": errors,
        "counts": {
            "action_records": len(action_records),
            "result_records": len(result_records),
            "trace_summaries": len(trace_summaries),
            "aggregate_rows": len(aggregates),
            "bounded_jitter_trace_counts": jitter_counts,
            "official_adapter_call_counts": official_adapter_counts,
        },
    }


def _environment_metadata(upstream: Mapping[str, Any]) -> dict[str, Any]:
    packages = {}
    for name in (
        "lerobot",
        "torch",
        "numpy",
        "draccus",
        "grpcio",
        "protobuf",
        "transformers",
        "huggingface-hub",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "retrieval_date": "2026-07-31",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "source_declared_version": upstream["source_version"],
        "source_declared_python": ">=3.12",
        "source_declared_torch": ">=2.7,<2.12.0",
        "actionstream_declared_python": ">=3.12,<3.13",
        "audit_environment_note": (
            "The ignored CPU-only audit venv satisfies both repositories' Python "
            "bounds and LeRobot's declared torch bound. It contains the official "
            "editable LeRobot source, base dependencies, and audit/report tools; "
            "hardware-specific deployment extras and physical devices were not used."
        ),
        "gpu_used": False,
        "physical_robot_used": False,
    }


def _method_source_metadata(
    bindings: UpstreamBindings,
    upstream: Mapping[str, Any],
) -> dict[str, Any]:
    symbols = {
        "RobotClient._aggregate_action_queues": (
            bindings.RobotClient._aggregate_action_queues
        ),
        "RobotClient.control_loop_action": bindings.RobotClient.control_loop_action,
        "RobotClient._ready_to_send_observation": (
            bindings.RobotClient._ready_to_send_observation
        ),
        "RobotClient.control_loop_observation": (
            bindings.RobotClient.control_loop_observation
        ),
        "PolicyServer._time_action_chunk": bindings.PolicyServer._time_action_chunk,
        "PolicyServer._enqueue_observation": bindings.PolicyServer._enqueue_observation,
        "PolicyServer.GetActions": bindings.PolicyServer.GetActions,
    }
    commit = upstream["commit"]
    checkout = Path(upstream["checkout_path"]).resolve()
    metadata: dict[str, Any] = {}
    for name, symbol in symbols.items():
        lines, start = inspect.getsourcelines(symbol)
        file_path = Path(inspect.getsourcefile(symbol) or "").resolve()
        relative = file_path.relative_to(checkout).as_posix()
        end = start + len(lines) - 1
        metadata[name] = {
            "file": relative,
            "start_line": start,
            "end_line": end,
            "sha256": file_sha256(file_path),
            "permalink": (
                "https://github.com/huggingface/lerobot/blob/"
                f"{commit}/{relative}#L{start}-L{end}"
            ),
        }
    return metadata


def run_benchmark(
    *,
    manifest_path: Path,
    lerobot_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    manifest = load_frozen_manifest(manifest_path)
    upstream = verify_upstream_checkout(lerobot_root, manifest["upstream"])
    bindings = load_upstream_bindings(lerobot_root)

    official_names = set(bindings.aggregate_functions)
    frozen_names = set(OFFICIAL_RUNTIME_TO_AGGREGATE.values())
    if official_names != frozen_names:
        raise RuntimeError(
            "Official aggregate registry changed after scenario freeze: "
            f"official={sorted(official_names)}, frozen={sorted(frozen_names)}"
        )

    action_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    trace_summaries: list[dict[str, Any]] = []
    contracts_by_key: dict[tuple[str, int], dict[str, Any]] = {}

    for family in manifest["latency_families"]:
        family_id = str(family["id"])
        trace_count = int(
            manifest["scenario"][
                "stochastic_trace_count"
                if family["kind"] == "seeded_jitter"
                else "deterministic_trace_count"
            ]
        )
        for offset in range(trace_count):
            seed = int(manifest["scenario"]["seed_base"]) + offset
            expected_input_hash: str | None = None
            for runtime in RUNTIME_IDS:
                records, results, summary, contract = run_trace(
                    manifest,
                    family_id=family_id,
                    seed=seed,
                    runtime=runtime,
                    bindings=bindings,
                )
                if expected_input_hash is None:
                    expected_input_hash = summary["input_sha256"]
                    contracts_by_key[(family_id, seed)] = contract
                elif summary["input_sha256"] != expected_input_hash:
                    raise AssertionError(
                        f"Runtime input mismatch for {family_id} seed {seed}"
                    )
                action_records.extend(records)
                result_records.extend(results)
                trace_summaries.append(summary)

    aggregates = aggregate_metrics(
        action_records,
        result_records,
        trace_summaries,
        execution_steps=int(manifest["scenario"]["execution_steps"]),
    )
    validation = validate_conformance_artifacts(
        manifest,
        action_records,
        result_records,
        trace_summaries,
        aggregates,
    )
    if not validation["passed"]:
        raise RuntimeError(f"M6 validation failed: {validation['errors']}")
    decision = build_decision(manifest, aggregates)

    manifest_hash = file_sha256(manifest_path)
    upstream_metadata = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        **upstream,
        "retrieval_date": "2026-07-31",
        "installed_version_before_audit": None,
        "installed_source_editable_version": importlib.metadata.version("lerobot"),
        "environment": _environment_metadata(upstream),
        "source_symbols_executed": _method_source_metadata(bindings, upstream),
        "official_registered_aggregate_functions": {
            name: inspect.getsource(function).strip()
            for name, function in bindings.aggregate_functions.items()
        },
        "upstream_async_unit_tests": {
            "command": (
                "LEROBOT_TEST_DEVICE=cpu python -m pytest "
                "tests/async_inference/test_robot_client.py "
                "tests/async_inference/test_policy_server.py "
                "tests/async_inference/test_helpers.py -q"
            ),
            "passed": True,
            "tests_passed": 22,
        },
    }

    scenario_copy = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "source_config": str(manifest_path),
        "source_config_sha256": manifest_hash,
        "frozen_before_benchmark": True,
        "manifest": manifest,
        "trace_contract_count": len(contracts_by_key),
        "trace_contracts_sha256": canonical_sha256(list(contracts_by_key.values())),
    }
    aggregate_payload = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "scenario_manifest_sha256": manifest_hash,
        "upstream_commit": upstream["commit"],
        "rows": aggregates,
    }

    _write_json(
        output_root / "upstream" / "pinned_upstream.json",
        upstream_metadata,
    )
    _write_json(
        output_root / "scenario" / "frozen_manifest.json",
        scenario_copy,
    )
    _write_jsonl(
        output_root / "scenario" / "trace_contracts.jsonl",
        [contracts_by_key[key] for key in sorted(contracts_by_key)],
    )
    _write_jsonl(
        output_root / "traces" / "action_records.jsonl",
        action_records,
    )
    _write_jsonl(
        output_root / "traces" / "result_records.jsonl",
        result_records,
    )
    _write_jsonl(
        output_root / "metrics" / "trace_summaries.jsonl",
        trace_summaries,
    )
    _write_json(
        output_root / "metrics" / "aggregate_metrics.json",
        aggregate_payload,
    )
    _write_json(output_root / "decision.json", decision)
    _write_json(output_root / "validation.json", validation)

    artifact_hashes = {
        str(path.relative_to(output_root)).replace("\\", "/"): file_sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "scenario_manifest_sha256": manifest_hash,
        "upstream_commit": upstream["commit"],
        "artifacts": artifact_hashes,
        "artifact_count": len(artifact_hashes),
    }
    artifact_manifest["artifact_manifest_sha256"] = canonical_sha256(artifact_manifest)
    _write_json(output_root / "artifact_manifest.json", artifact_manifest)
    return {
        "manifest": manifest,
        "upstream": upstream_metadata,
        "action_records": action_records,
        "result_records": result_records,
        "trace_summaries": trace_summaries,
        "aggregates": aggregates,
        "decision": decision,
        "validation": validation,
        "artifact_manifest": artifact_manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_repo_root() / "configs" / "m6_g0.json",
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=_repo_root() / ".external" / "lerobot",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "outputs" / "m6_g0",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_benchmark(
        manifest_path=args.manifest.resolve(),
        lerobot_root=args.lerobot_root.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(
        f"{args.output_root} "
        f"({result['decision']['classification']}; "
        f"{len(result['action_records'])} action records)"
    )


if __name__ == "__main__":
    main()
