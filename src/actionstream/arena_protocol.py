"""Frozen Isaac Lab-Arena benchmark contracts and provenance helpers.

This module is deliberately importable without Isaac Sim.  It validates the
scientific contract and expands a deterministic paired matrix; the actual Arena
runner remains responsible for proving that the pinned simulator, task assets,
policy checkpoint, observation bridge, and action bridge executed successfully.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence


_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_STATUSES = {"success", "partial", "blocked", "failed", "not_applicable"}


class ArenaProtocolError(ValueError):
    """Raised when a frozen Arena protocol violates a pairing or provenance gate."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArenaCell:
    """One paired benchmark cell before simulator execution."""

    split: str
    task_family: str
    reset_seed: int
    network_trace: str
    model: str
    runtime: str
    num_envs: int

    @property
    def pair_id(self) -> str:
        return (
            f"{self.split}__{self.task_family}__seed{self.reset_seed}"
            f"__{self.network_trace}__{self.model}"
        )

    @property
    def cell_id(self) -> str:
        return f"{self.pair_id}__{self.runtime}"


@dataclass(frozen=True)
class FrozenArenaProtocol:
    """Validated immutable view over the JSON benchmark declaration."""

    path: Path
    sha256: str
    raw: Mapping[str, Any]

    @property
    def tasks(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw["tasks"])

    @property
    def models(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw["models"])

    @property
    def runtimes(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw["runtimes"])

    def cells(self, split: str) -> tuple[ArenaCell, ...]:
        if split not in {"development", "holdout"}:
            raise ArenaProtocolError(f"unknown split: {split}")
        split_cfg = self.raw["splits"][split]
        task_rows = self.tasks
        model_keys = [str(row["key"]) for row in self.models]
        runtime_keys = [str(row["key"]) for row in self.runtimes]
        rows: list[ArenaCell] = []
        for task, seed, trace, model, runtime in product(
            task_rows,
            split_cfg["reset_seeds"],
            split_cfg["network_trace_ids"],
            model_keys,
            runtime_keys,
        ):
            rows.append(
                ArenaCell(
                    split=split,
                    task_family=str(task["family"]),
                    reset_seed=int(seed),
                    network_trace=str(trace),
                    model=model,
                    runtime=runtime,
                    num_envs=int(task["arena_env_args"]["num_envs"]),
                )
            )
        return tuple(rows)

    def arena_job_for(self, cell: ArenaCell) -> dict[str, Any]:
        """Render one official Arena eval-runner job.

        Arena 0.2.1 applies ``--seed`` at process scope, not per job.  Therefore
        callers must execute one reset seed per process and pass ``cell.reset_seed``
        on the command line.  Keeping this renderer one-cell-at-a-time prevents a
        multi-seed JSON file from silently reusing one simulator seed.
        """

        task = next(row for row in self.tasks if row["family"] == cell.task_family)
        policy_type = str(self.raw["arena_integration"]["policy_type"])
        return {
            "jobs": [
                {
                    "name": cell.cell_id,
                    "arena_env_args": dict(task["arena_env_args"]),
                    "num_episodes": int(self.raw["episodes_per_vector_env"])
                    * int(cell.num_envs),
                    "policy_type": policy_type,
                    "policy_config_dict": {
                        "protocol_path": str(self.path),
                        "protocol_sha256": self.sha256,
                        "split": cell.split,
                        "task_family": cell.task_family,
                        "reset_seed": cell.reset_seed,
                        "network_trace_id": cell.network_trace,
                        "model": cell.model,
                        "runtime": cell.runtime,
                        "num_envs": cell.num_envs,
                    },
                }
            ]
        }


def _nonempty_unique(values: Sequence[Any], *, name: str) -> set[Any]:
    if not values:
        raise ArenaProtocolError(f"{name} must not be empty")
    unique = set(values)
    if len(unique) != len(values):
        raise ArenaProtocolError(f"{name} contains duplicates")
    return unique


def _validate_protocol(raw: Mapping[str, Any]) -> None:
    if raw.get("schema_version") != 1:
        raise ArenaProtocolError("schema_version must be 1")
    if not str(raw.get("experiment_id", "")).strip():
        raise ArenaProtocolError("experiment_id must be non-empty")
    if not str(raw.get("frozen_utc", "")).endswith("Z"):
        raise ArenaProtocolError("frozen_utc must be an explicit UTC timestamp")

    source = raw.get("arena_source", {})
    if (
        source.get("repository_url")
        != "https://github.com/isaac-sim/IsaacLab-Arena.git"
    ):
        raise ArenaProtocolError(
            "arena_source must name the official IsaacLab-Arena repository"
        )
    if not _SHA40.fullmatch(str(source.get("commit", ""))):
        raise ArenaProtocolError(
            "arena_source.commit must be a 40-character lowercase Git SHA"
        )
    if not str(source.get("release", "")).strip():
        raise ArenaProtocolError("arena_source.release must be pinned")

    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or len(tasks) < 3:
        raise ArenaProtocolError("at least three task families are required")
    task_families = _nonempty_unique(
        [row.get("family") for row in tasks], name="task families"
    )
    environments = _nonempty_unique(
        [row.get("arena_env_args", {}).get("environment") for row in tasks],
        name="Arena environments",
    )
    if len(task_families) < 3 or len(environments) < 3:
        raise ArenaProtocolError(
            "initial-state variants cannot be relabelled as task families"
        )
    for row in tasks:
        args = row.get("arena_env_args", {})
        if int(args.get("num_envs", 0)) < 2:
            raise ArenaProtocolError(
                "every Arena task must use vectorized num_envs >= 2"
            )
        if not bool(args.get("enable_cameras", False)):
            raise ArenaProtocolError("learned visual-policy tasks must enable cameras")

    models = raw.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ArenaProtocolError("X-VLA and one RTC-compatible model are required")
    model_keys = _nonempty_unique([row.get("key") for row in models], name="models")
    if "xvla" not in model_keys:
        raise ArenaProtocolError("X-VLA must be included")
    if not any(bool(row.get("rtc_compatible")) for row in models):
        raise ArenaProtocolError("at least one model must declare RTC compatibility")
    for row in models:
        if not _SHA40.fullmatch(str(row.get("revision", ""))):
            raise ArenaProtocolError(
                f"model {row.get('key')} revision must be a Git SHA"
            )

    runtimes = raw.get("runtimes")
    if not isinstance(runtimes, list):
        raise ArenaProtocolError("runtimes must be a list")
    runtime_keys = _nonempty_unique(
        [row.get("key") for row in runtimes], name="runtimes"
    )
    required_runtimes = {
        "sync",
        "latest_only",
        "rtc",
        "actionstream_aligned",
        "actionstream_guarded",
    }
    if runtime_keys != required_runtimes:
        raise ArenaProtocolError(
            f"runtime matrix must be exactly {sorted(required_runtimes)}, got {sorted(runtime_keys)}"
        )
    for row in runtimes:
        if not str(row.get("implementation_source", "")).strip():
            raise ArenaProtocolError(
                f"runtime {row.get('key')} lacks implementation provenance"
            )

    splits = raw.get("splits", {})
    if set(splits) != {"development", "holdout"}:
        raise ArenaProtocolError("splits must contain development and holdout only")
    dev_seeds = _nonempty_unique(
        splits["development"].get("reset_seeds", []), name="development seeds"
    )
    holdout_seeds = _nonempty_unique(
        splits["holdout"].get("reset_seeds", []), name="holdout seeds"
    )
    if dev_seeds & holdout_seeds:
        raise ArenaProtocolError("development and holdout reset seeds must be disjoint")
    dev_traces = _nonempty_unique(
        splits["development"].get("network_trace_ids", []),
        name="development network traces",
    )
    holdout_traces = _nonempty_unique(
        splits["holdout"].get("network_trace_ids", []), name="holdout network traces"
    )
    if dev_traces & holdout_traces:
        raise ArenaProtocolError(
            "development and holdout network traces must be disjoint"
        )

    trace_rows = raw.get("network_traces")
    if not isinstance(trace_rows, list):
        raise ArenaProtocolError("network_traces must be a list")
    trace_ids = _nonempty_unique(
        [row.get("id") for row in trace_rows], name="network traces"
    )
    referenced = dev_traces | holdout_traces
    if trace_ids != referenced:
        raise ArenaProtocolError(
            "network trace definitions must exactly match split references"
        )
    for row in trace_rows:
        samples = row.get("latency_ms")
        if not isinstance(samples, list) or len(samples) < 32:
            raise ArenaProtocolError(
                f"network trace {row.get('id')} must freeze at least 32 samples"
            )
        if any(not isinstance(value, int | float) or value < 0 for value in samples):
            raise ArenaProtocolError(
                f"network trace {row.get('id')} contains invalid latency"
            )

    if raw.get("episodes_per_vector_env", 0) < 1:
        raise ArenaProtocolError("episodes_per_vector_env must be positive")
    gates = raw.get("execution_gates", {})
    required_gates = {
        "observation_contract",
        "action_contract",
        "learned_checkpoint",
        "gpu_inference",
        "video_content_checked",
    }
    if set(gates) != required_gates or any(
        value != "required" for value in gates.values()
    ):
        raise ArenaProtocolError(
            "all learned-policy execution gates must remain required"
        )


def load_arena_protocol(path: str | Path) -> FrozenArenaProtocol:
    protocol_path = Path(path).resolve()
    raw = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ArenaProtocolError("protocol root must be a JSON object")
    _validate_protocol(raw)
    return FrozenArenaProtocol(
        path=protocol_path, sha256=sha256_file(protocol_path), raw=raw
    )


def validate_arena_result(
    record: Mapping[str, Any], protocol: FrozenArenaProtocol
) -> None:
    """Reject result rows that blur structural, learned, and executed evidence."""

    if record.get("protocol_sha256") != protocol.sha256:
        raise ArenaProtocolError("result protocol hash does not match the frozen file")
    status = record.get("status")
    if status not in _ALLOWED_STATUSES:
        raise ArenaProtocolError(f"invalid result status: {status}")
    required = {
        "cell_id",
        "arena_commit",
        "lerobot_commit",
        "policy_revision",
        "evidence_level",
    }
    missing = sorted(key for key in required if not record.get(key))
    if missing:
        raise ArenaProtocolError(f"result lacks provenance fields: {missing}")
    if record["arena_commit"] != protocol.raw["arena_source"]["commit"]:
        raise ArenaProtocolError("result used a different Arena commit")
    evidence_level = record["evidence_level"]
    if evidence_level not in {
        "structural",
        "simulator_smoke",
        "learned_policy_rollout",
    }:
        raise ArenaProtocolError(f"invalid evidence_level: {evidence_level}")
    if status == "success" and evidence_level != "learned_policy_rollout":
        raise ArenaProtocolError(
            "success requires executed learned-policy rollout evidence"
        )
    if evidence_level == "learned_policy_rollout":
        metrics = record.get("metrics", {})
        needed_metrics = {
            "success",
            "gpu_parallel_env_steps_per_second",
            "queue_age_steps",
            "fallback_count",
        }
        if not needed_metrics.issubset(metrics):
            raise ArenaProtocolError(
                "learned rollout lacks required Arena/runtime metrics"
            )
        if not record.get("video_paths") or not record.get("episode_provenance_path"):
            raise ArenaProtocolError(
                "learned rollout requires video and per-episode provenance"
            )
