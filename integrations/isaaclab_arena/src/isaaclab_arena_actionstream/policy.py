"""Isaac Lab-Arena ``PolicyBase`` wrapper around ActionStream inference.

The pinned Arena 0.2.1 runner does not expose a policy-close hook for local
policies.  ``is_remote`` is intentionally true so its documented
``shutdown_remote`` cleanup path joins the ActionStream worker and closes the
learned provider.  No network server is implied by that compatibility hook.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from actionstream.arena_bridge import (
    DroidObservationSnapshot,
    absolute_targets_to_droid_relative_ik,
    droid_bounded_hold_action,
    snapshot_droid_observation,
)
from actionstream.arena_protocol import (
    ArenaCell,
    ArenaProtocolError,
    load_arena_protocol,
)
from actionstream.lerobot_inference import (
    ActionStreamInferenceConfig,
    ActionStreamInferenceEngine,
)
from isaaclab_arena.policy.policy_base import PolicyBase

from .provider import ArenaChunkProvider, resolve_provider_factory


@dataclass
class ActionStreamArenaPolicyConfig:
    protocol_path: str
    protocol_sha256: str
    split: str
    task_family: str
    reset_seed: int
    network_trace_id: str
    model: str
    runtime: str
    num_envs: int
    provider_factory: str = (
        "isaaclab_arena_actionstream.lerobot_provider:create_provider"
    )
    device: str = "cuda"
    telemetry_directory: str = "outputs/isaaclab_arena_v1/telemetry"


class _TraceReplay:
    def __init__(self, values_ms: list[int | float]) -> None:
        if not values_ms:
            raise ValueError("network trace cannot be empty")
        self._values = tuple(float(value) for value in values_ms)
        self._offset = 0
        self._lock = threading.Lock()

    def next_seconds(self) -> float:
        with self._lock:
            value = self._values[self._offset % len(self._values)]
            self._offset += 1
        return value / 1000.0


class ActionStreamArenaPolicy(PolicyBase):
    """Vector-batched Arena policy for aligned and guarded ActionStream cells.

    The five-runtime matrix is frozen in the protocol, but this class refuses
    ``sync``, official ``latest_only``, and official ``rtc`` cells.  Those
    baselines require separate vector-safe adapters over their upstream
    implementations; silently emulating them here would mislabel local code as
    official LeRobot.  The refusal is part of the evidence boundary.
    """

    config_class = ActionStreamArenaPolicyConfig

    def __init__(self, config: ActionStreamArenaPolicyConfig):
        super().__init__(config)
        protocol = load_arena_protocol(config.protocol_path)
        if protocol.sha256 != config.protocol_sha256:
            raise ArenaProtocolError("Arena policy received a non-frozen protocol file")
        cells = {cell.cell_id: cell for cell in protocol.cells(config.split)}
        requested = ArenaCell(
            split=config.split,
            task_family=config.task_family,
            reset_seed=int(config.reset_seed),
            network_trace=config.network_trace_id,
            model=config.model,
            runtime=config.runtime,
            num_envs=int(config.num_envs),
        )
        if requested.cell_id not in cells or cells[requested.cell_id] != requested:
            raise ArenaProtocolError(
                f"cell is not in the frozen matrix: {requested.cell_id}"
            )
        if config.runtime not in {"actionstream_aligned", "actionstream_guarded"}:
            raise NotImplementedError(
                f"{config.runtime} must run through its pinned upstream LeRobot adapter; "
                "ActionStreamArenaPolicy will not impersonate that baseline"
            )

        trace_row = next(
            row
            for row in protocol.raw["network_traces"]
            if row["id"] == config.network_trace_id
        )
        self._trace = _TraceReplay(trace_row["latency_ms"])
        self._protocol = protocol
        self._current_snapshot: DroidObservationSnapshot | None = None
        self._step = 0
        self._adapter_hold_count = 0
        self._partial_reset_count = 0
        self._closed = False
        self._telemetry_lock = threading.Lock()

        factory = resolve_provider_factory(config.provider_factory)
        self._provider: ArenaChunkProvider = factory(asdict(config), config.model)
        engine_config = ActionStreamInferenceConfig(
            inference_timeout_s=5.0,
            bounded_hold_steps=0 if config.runtime == "actionstream_aligned" else 2,
            latest_only_fallback=config.runtime == "actionstream_guarded",
        )
        self._engine = ActionStreamInferenceEngine(
            policy=None,
            preprocessor=None,
            postprocessor=None,
            hw_features={},
            task="",
            device=config.device,
            robot_type="isaaclab_arena_droid",
            config=engine_config,
            infer_chunk=self._infer_chunk,
            reset_provider=self._provider.reset,
        )
        self._engine.start()
        self._engine.resume()

        telemetry_directory = Path(config.telemetry_directory).expanduser().resolve()
        telemetry_directory.mkdir(parents=True, exist_ok=True)
        self._telemetry_path = telemetry_directory / f"{requested.cell_id}.jsonl"
        if self._telemetry_path.exists():
            raise FileExistsError(
                f"refusing to overwrite Arena telemetry: {self._telemetry_path}"
            )
        self._write_event(
            {
                "event": "start",
                "cell_id": requested.cell_id,
                "protocol_sha256": protocol.sha256,
                "arena_commit": protocol.raw["arena_source"]["commit"],
                "lerobot_commit": protocol.raw["lerobot_source"]["commit"],
                "model_revision": next(
                    row for row in protocol.models if row["key"] == config.model
                )["revision"],
            }
        )

    def _write_event(self, event: dict[str, Any]) -> None:
        record = {"wall_time_ns": time.time_ns(), **event}
        encoded = json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with self._telemetry_lock:
            with self._telemetry_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")

    def _infer_chunk(self, observation: dict[str, Any], task: str) -> torch.Tensor:
        snapshot = observation.get("snapshot")
        if not isinstance(snapshot, DroidObservationSnapshot):
            raise TypeError("Arena inference mailbox lacks a validated DROID snapshot")
        delay_s = self._trace.next_seconds()
        if delay_s:
            time.sleep(delay_s)
        targets = self._provider.predict_targets(snapshot, task, rtc=False)
        targets = torch.as_tensor(targets, dtype=torch.float32, device="cpu")
        expected_tail = (snapshot.num_envs, 8)
        if (
            targets.ndim != 3
            or tuple(targets.shape[1:]) != expected_tail
            or targets.shape[0] < 1
        ):
            raise ValueError(
                f"provider must return [T,{snapshot.num_envs},8], got {tuple(targets.shape)}"
            )
        if not torch.isfinite(targets).all():
            raise ValueError("provider returned non-finite absolute targets")
        return targets.reshape(targets.shape[0], -1)

    def get_action(self, env, observation) -> torch.Tensor:
        action_device = torch.device(env.unwrapped.device)
        snapshot = snapshot_droid_observation(observation)
        if snapshot.num_envs != self.config.num_envs:
            raise ValueError(
                f"Arena vector batch changed: expected {self.config.num_envs}, got {snapshot.num_envs}"
            )
        self._current_snapshot = snapshot
        self._engine.notify_observation({"snapshot": snapshot})
        target = self._engine.get_action(None)
        source = "queue"
        if target is None:
            action = droid_bounded_hold_action(snapshot)
            self._adapter_hold_count += 1
            source = "depleted_relative_hold"
        else:
            action = absolute_targets_to_droid_relative_ik(
                target.reshape(snapshot.num_envs, 8), snapshot
            )
        telemetry = self._engine.telemetry.to_dict()
        self._write_event(
            {
                "event": "action",
                "step": self._step,
                "source": source,
                "adapter_hold_count": self._adapter_hold_count,
                "partial_reset_count": self._partial_reset_count,
                "runtime": self.config.runtime,
                "telemetry": telemetry,
            }
        )
        self._step += 1
        return action.to(device=action_device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None and int(env_ids.numel()) < self.config.num_envs:
            # The backend currently has one vector-batched queue.  A partial
            # episode reset invalidates the whole batch safely and is recorded;
            # it does not reset other simulator environments.
            self._partial_reset_count += 1
        self._engine.reset()
        self._current_snapshot = None
        self._write_event(
            {
                "event": "reset",
                "env_ids": None if env_ids is None else env_ids.detach().cpu().tolist(),
                "partial_reset_count": self._partial_reset_count,
            }
        )

    def set_task_description(self, task_description: str | None) -> str:
        task = super().set_task_description(task_description)
        self._engine.set_task(task or "")
        return task

    @property
    def is_remote(self) -> bool:
        return True

    def shutdown_remote(self, kill_server: bool = False) -> None:
        del kill_server
        if self._closed:
            return
        self._engine.stop()
        self._provider.close()
        self._write_event(
            {
                "event": "stop",
                "adapter_hold_count": self._adapter_hold_count,
                "partial_reset_count": self._partial_reset_count,
                "telemetry": self._engine.telemetry.to_dict(),
            }
        )
        self._closed = True

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        for field in ActionStreamArenaPolicyConfig.__dataclass_fields__.values():
            option = f"--{field.name}"
            if field.default is not MISSING:
                parser.add_argument(option, default=field.default)
            else:
                parser.add_argument(option, required=True)
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ActionStreamArenaPolicy":
        values = {
            name: getattr(args, name)
            for name in ActionStreamArenaPolicyConfig.__dataclass_fields__
        }
        values["reset_seed"] = int(values["reset_seed"])
        values["num_envs"] = int(values["num_envs"])
        return ActionStreamArenaPolicy(ActionStreamArenaPolicyConfig(**values))
