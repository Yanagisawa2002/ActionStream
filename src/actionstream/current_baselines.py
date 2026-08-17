"""Closed-loop current LeRobot Async/RTC comparison on LIBERO.

The transport and LIBERO loop live here, but the scheduling operations do not:
official Async cells call the pinned ``PolicyServer``/``RobotClient`` methods and
RTC cells call both the policy RTC signature and upstream RTC ``ActionQueue``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from actionstream.adaptive_runtime import (
    AdaptiveSelector,
    AdaptiveSelectorConfig,
    load_selector_config,
)
from actionstream.lerobot_backend import (
    StepOutput,
    extract_success,
    immutable_observation_snapshot,
    thaw_observation_snapshot,
)
from actionstream.libero_config import ensure_isolated_libero_config
from actionstream.m6_conformance import (
    OfficialLeRobotAdapter,
    load_upstream_bindings,
    verify_upstream_checkout,
)
from actionstream.runtime import (
    ActionQueue as ActionStreamQueue,
    InferencePayload,
    InferenceRequest,
    InferenceResult,
    LatestRequestWorker,
    QueueNotReady,
)


RUNTIMES = (
    "sync_hold",
    "lerobot_weighted_average",
    "lerobot_latest_only",
    "actionstream_aligned",
    "actionstream_adaptive",
    "lerobot_rtc",
)
OFFICIAL_ASYNC_AGGREGATES = {
    "lerobot_weighted_average": "weighted_average",
    "lerobot_latest_only": "latest_only",
}


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    override = os.environ.get("ACTIONSTREAM_SOURCE_COMMIT")
    if override:
        return override
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a nonempty comma-separated list")
    return values


def _parse_int_csv(value: str) -> list[int]:
    try:
        return [int(item) for item in _parse_csv(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str
    control_mode: str
    chunk_size: int
    request_interval_steps: int
    rtc_expected: bool
    rtc_execution_horizon: int
    required: bool
    rename_map: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelSpec:
        spec = cls(
            key=str(value["key"]),
            model_id=str(value["model_id"]),
            revision=str(value["revision"]),
            control_mode=str(value["control_mode"]),
            chunk_size=int(value["chunk_size"]),
            request_interval_steps=int(value["request_interval_steps"]),
            rtc_expected=bool(value["rtc_expected"]),
            rtc_execution_horizon=int(value.get("rtc_execution_horizon", 10)),
            required=bool(value.get("required", False)),
            rename_map={str(key): str(item) for key, item in value.get("rename_map", {}).items()},
        )
        if spec.control_mode not in {"absolute", "relative"}:
            raise ValueError(f"Invalid control mode for {spec.key}: {spec.control_mode}")
        if spec.chunk_size <= 0 or spec.request_interval_steps <= 0:
            raise ValueError(f"Invalid chunk/request interval for {spec.key}")
        return spec


@dataclass(frozen=True)
class DelayTrace:
    key: str
    milliseconds: tuple[int, ...]
    repeat: bool
    definition: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DelayTrace:
        key = str(value["key"])
        kind = str(value["kind"])
        if kind == "fixed":
            milliseconds = (int(value["milliseconds"]),)
            repeat = True
        elif kind == "seeded_uniform":
            center = int(value["center_milliseconds"])
            half_width = int(value["half_width_milliseconds"])
            length = int(value["trace_length"])
            if half_width < 0 or length <= 0:
                raise ValueError(f"Invalid jitter definition for {key}")
            rng = random.Random(int(value["seed"]))
            milliseconds = tuple(
                rng.randint(center - half_width, center + half_width) for _ in range(length)
            )
            repeat = False
        elif kind == "seeded_burst":
            center = int(value["base_center_milliseconds"])
            half_width = int(value["base_half_width_milliseconds"])
            probability = float(value["burst_probability"])
            burst_minimum = int(value["burst_min_milliseconds"])
            burst_maximum = int(value["burst_max_milliseconds"])
            length = int(value["trace_length"])
            if (
                half_width < 0
                or length <= 0
                or not 0.0 <= probability <= 1.0
                or burst_minimum < 0
                or burst_minimum > burst_maximum
            ):
                raise ValueError(f"Invalid burst definition for {key}")
            rng = random.Random(int(value["seed"]))
            milliseconds = tuple(
                rng.randint(burst_minimum, burst_maximum)
                if rng.random() < probability
                else rng.randint(center - half_width, center + half_width)
                for _ in range(length)
            )
            repeat = False
        else:
            raise ValueError(f"Unsupported delay profile kind: {kind}")
        if any(item < 0 for item in milliseconds):
            raise ValueError(f"Delay profile {key} contains a negative delay")
        return cls(key=key, milliseconds=milliseconds, repeat=repeat, definition=dict(value))

    def seconds_at(self, ordinal: int) -> float:
        if ordinal < 0:
            raise ValueError("Delay trace ordinal must be non-negative")
        if self.repeat:
            value = self.milliseconds[ordinal % len(self.milliseconds)]
        else:
            if ordinal >= len(self.milliseconds):
                raise IndexError(
                    f"Delay trace {self.key} exhausted at request {ordinal}; "
                    f"frozen length={len(self.milliseconds)}"
                )
            value = self.milliseconds[ordinal]
        return value / 1000.0

    @property
    def trace_sha256(self) -> str:
        return _sha256_json(list(self.milliseconds))


@dataclass(frozen=True)
class Protocol:
    raw: Mapping[str, Any]
    models: Mapping[str, ModelSpec]
    delays: Mapping[str, DelayTrace]
    adaptive_selector: AdaptiveSelectorConfig | None


def load_protocol(path: Path | str) -> Protocol:
    protocol_path = Path(path).resolve()
    raw = json.loads(protocol_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Expected current LeRobot baseline schema_version=1")
    source = raw.get("source", {})
    if len(str(source.get("commit", ""))) != 40:
        raise ValueError("Protocol must freeze a full LeRobot commit SHA")
    model_items = [ModelSpec.from_mapping(item) for item in raw.get("models", [])]
    delay_items = [DelayTrace.from_mapping(item) for item in raw.get("delay_profiles", [])]
    models = {item.key: item for item in model_items}
    delays = {item.key: item for item in delay_items}
    if len(models) != len(model_items) or len(delays) != len(delay_items):
        raise ValueError("Model and delay profile keys must be unique")
    if not models or not delays:
        raise ValueError("Protocol requires at least one model and delay profile")
    unknown_runtimes = set(raw.get("runtimes", [])) - set(RUNTIMES)
    if unknown_runtimes:
        raise ValueError(f"Unknown runtimes in protocol: {sorted(unknown_runtimes)}")
    adaptive_selector: AdaptiveSelectorConfig | None = None
    adaptive_raw = raw.get("adaptive_runtime")
    if adaptive_raw is not None:
        reference = Path(str(adaptive_raw["selector_protocol_path"]))
        candidates = (
            [reference]
            if reference.is_absolute()
            else [
                Path.cwd() / reference,
                protocol_path.parent / reference,
                protocol_path.parent.parent / reference,
            ]
        )
        selector_path = next((item for item in candidates if item.is_file()), None)
        if selector_path is None:
            raise FileNotFoundError(
                f"Adaptive selector protocol not found: {reference}; "
                f"checked={[str(item) for item in candidates]}"
            )
        adaptive_selector = load_selector_config(
            selector_path,
            expected_sha256=str(adaptive_raw["selector_protocol_sha256"]),
        )
    if "actionstream_adaptive" in raw.get("runtimes", []) and adaptive_selector is None:
        raise ValueError("Adaptive runtime requires a frozen adaptive_runtime protocol")
    return Protocol(
        raw=raw,
        models=models,
        delays=delays,
        adaptive_selector=adaptive_selector,
    )


@dataclass(frozen=True)
class CurrentInferenceOutput:
    actions: np.ndarray
    raw_actions: np.ndarray
    model_latency_seconds: float
    raw_shape: tuple[int, ...]
    raw_dtype: str
    predicted_inference_delay_steps: int | None = None


class CurrentLeRobotBackend:
    """Policy-native current LeRobot model, processors, and one LIBERO env."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        task_ids: Sequence[int],
        suite: str,
        episode_length: int,
        seed: int,
        device: str = "cuda",
    ) -> None:
        ensure_isolated_libero_config()
        if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
            raise RuntimeError("CurrentLeRobotBackend requires MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl")

        from lerobot.configs import PreTrainedConfig
        from lerobot.envs import make_env, make_env_config, make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.utils.random_utils import set_seed

        self.spec = spec
        self.model_id = spec.model_id
        self.model_revision = spec.revision
        self.suite = suite
        self.task_ids = list(task_ids)
        self.episode_length = int(episode_length)
        self._set_seed = set_seed
        self._rtc_latency_max_seconds = 0.0

        set_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        self.env_cfg = make_env_config(
            "libero",
            task=suite,
            task_ids=self.task_ids,
            control_mode=spec.control_mode,
            episode_length=self.episode_length,
            max_parallel_tasks=1,
        )
        policy_cfg = PreTrainedConfig.from_pretrained(spec.model_id, revision=spec.revision)
        policy_cfg.device = device
        policy_cfg.pretrained_path = Path(spec.model_id)
        policy_cfg.pretrained_revision = spec.revision
        actual_chunk = int(getattr(policy_cfg, "chunk_size", 0))
        if actual_chunk != spec.chunk_size:
            raise RuntimeError(
                f"Frozen chunk mismatch for {spec.key}: expected {spec.chunk_size}, got {actual_chunk}"
            )
        self.policy_cfg = policy_cfg
        self.envs = make_env(self.env_cfg, n_envs=1, use_async_envs=False)
        self.policy = make_policy(
            cfg=policy_cfg,
            env_cfg=self.env_cfg,
            rename_map=dict(spec.rename_map),
        ).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=spec.model_id,
            pretrained_revision=spec.revision,
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy.config.device)},
                "rename_observations_processor": {"rename_map": dict(spec.rename_map)},
            },
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=self.env_cfg,
            policy_cfg=policy_cfg,
        )

    @property
    def supports_rtc(self) -> bool:
        capability = getattr(self.policy, "supports_rtc", None)
        return bool(callable(capability) and capability())

    def configure_rtc(self, *, enabled: bool) -> None:
        """Match LeRobot's rollout context: attach RTCProcessor only for RTC cells."""
        if enabled and not self.supports_rtc:
            raise RuntimeError(f"Policy {self.spec.key} does not declare RTC support")
        if not self.supports_rtc:
            return
        self.policy.config.rtc_config = None
        if not enabled:
            # LeRobot's initializer only attaches a processor when a config is
            # present; it does not clear a processor from an earlier RTC cell.
            # Clear both references so runtime ordering cannot contaminate the
            # following non-RTC baseline.
            self.policy.rtc_processor = None
            model = getattr(self.policy, "model", None)
            if model is not None:
                model.rtc_processor = None
            return
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        self.policy.config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=self.spec.rtc_execution_horizon,
        )
        initializer = getattr(self.policy, "init_rtc_processor", None)
        if not callable(initializer):
            raise RuntimeError(
                f"Policy {self.spec.key} declares RTC support without init_rtc_processor()"
            )
        initializer()

    def _env(self, task_id: int) -> Any:
        try:
            return self.envs[self.suite][task_id]
        except KeyError as exc:
            raise ValueError(f"Task {task_id} was not configured") from exc

    def _sub_env(self, task_id: int) -> Any:
        env = self._env(task_id)
        if len(env.envs) != 1:
            raise RuntimeError(f"Expected one synchronous sub-environment, got {len(env.envs)}")
        return env.envs[0]

    def official_render_sub_env(self, task_id: int) -> Any:
        """Expose the reset synchronous sub-env for audited state-only rendering."""

        return self._sub_env(task_id)

    def reset_episode(
        self,
        *,
        task_id: int,
        seed: int,
        initial_state_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        sub_env = self._sub_env(task_id)
        sub_env.init_state_id = int(initial_state_index)
        self._set_seed(int(seed))
        self._rtc_latency_max_seconds = 0.0
        self.policy.reset()
        for pipeline in (
            self.env_preprocessor,
            self.preprocessor,
            self.postprocessor,
            self.env_postprocessor,
        ):
            pipeline.reset()
        observation, info = self._env(task_id).reset(seed=[int(seed)])
        return observation, info, str(sub_env.task_description)

    def controller_frequency_hz(self, task_id: int) -> float:
        sub_env = self._sub_env(task_id)
        frequency = float(sub_env._env.env.control_freq)
        if frequency <= 0:
            raise RuntimeError(f"Invalid LIBERO controller frequency: {frequency}")
        return frequency

    def _prepare_observation(
        self,
        observation: dict[str, Any],
        instruction: str,
    ) -> dict[str, Any]:
        from lerobot.envs import preprocess_observation

        prepared = preprocess_observation(observation)
        prepared["task"] = [instruction]
        return self.preprocessor(self.env_preprocessor(prepared))

    def _finalize_chunk(self, raw_chunk: torch.Tensor, *, rtc: bool) -> torch.Tensor:
        from lerobot.utils.constants import ACTION

        if raw_chunk.ndim != 3 or raw_chunk.shape[0] != 1:
            raise RuntimeError(f"Expected [1,T,D] policy chunk, got {tuple(raw_chunk.shape)}")
        if rtc:
            policy_chunk = self.postprocessor(raw_chunk)
            if policy_chunk.ndim != 3:
                raise RuntimeError(
                    f"RTC policy postprocessor returned {tuple(policy_chunk.shape)}, expected [1,T,D]"
                )
            policy_steps = [policy_chunk[:, index, :] for index in range(policy_chunk.shape[1])]
        else:
            policy_steps = [
                self.postprocessor(raw_chunk[:, index, :]) for index in range(raw_chunk.shape[1])
            ]
        final_steps: list[torch.Tensor] = []
        for index, action in enumerate(policy_steps):
            final = self.env_postprocessor({ACTION: action})[ACTION]
            if final.ndim != 2 or final.shape != (1, 7):
                raise RuntimeError(
                    f"Environment postprocessor returned {tuple(final.shape)} at step {index}"
                )
            if not torch.isfinite(final).all():
                raise RuntimeError(f"Non-finite final command at chunk step {index}")
            final_steps.append(final)
        return torch.stack(final_steps, dim=1)

    @staticmethod
    def _normalize_rtc_prefix(actions: torch.Tensor, target_steps: int) -> torch.Tensor:
        if actions.ndim != 2:
            raise ValueError(f"RTC prefix must be [T,D], got {tuple(actions.shape)}")
        if len(actions) >= target_steps:
            return actions[:target_steps]
        padded = torch.zeros(
            (target_steps, actions.shape[1]),
            device=actions.device,
            dtype=actions.dtype,
        )
        padded[: len(actions)] = actions
        return padded

    def infer_action_chunk(
        self,
        observation: dict[str, Any],
        instruction: str,
        *,
        rtc_prefix: np.ndarray | None = None,
        rtc_execution_horizon: int | None = None,
        controller_period_seconds: float | None = None,
    ) -> CurrentInferenceOutput:
        rtc = rtc_execution_horizon is not None
        batch = self._prepare_observation(observation, instruction)
        kwargs: dict[str, Any] = {}
        predicted_delay_steps: int | None = None
        if rtc:
            if not self.supports_rtc:
                raise RuntimeError(f"Policy {self.spec.key} does not declare RTC support")
            if controller_period_seconds is None or controller_period_seconds <= 0:
                raise ValueError("RTC inference requires a positive controller period")
            predicted_delay_steps = math.ceil(
                self._rtc_latency_max_seconds / controller_period_seconds
            )
            prefix_tensor = None
            if rtc_prefix is not None:
                prefix_tensor = torch.as_tensor(
                    rtc_prefix,
                    device=torch.device(self.policy.config.device),
                )
                prefix_tensor = self._normalize_rtc_prefix(
                    prefix_tensor,
                    int(rtc_execution_horizon),
                )
            kwargs = {
                "inference_delay": predicted_delay_steps,
                "prev_chunk_left_over": prefix_tensor,
            }

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        # Do not use inference_mode here. Current LeRobot RTC temporarily
        # re-enables autograd inside RTCProcessor.denoise_step() to compute the
        # guidance correction. inference_mode is stronger than no_grad and
        # makes that upstream path fail even inside torch.enable_grad().
        with torch.no_grad():
            raw_chunk = self.policy.predict_action_chunk(batch, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_latency = time.perf_counter() - started
        if rtc:
            self._rtc_latency_max_seconds = max(self._rtc_latency_max_seconds, model_latency)
        final_chunk = self._finalize_chunk(raw_chunk, rtc=rtc)
        actions = final_chunk[0].detach().cpu().numpy().astype(np.float32, copy=True)
        raw_actions = raw_chunk[0].detach().cpu().numpy().copy()
        if actions.shape != (self.spec.chunk_size, 7):
            raise RuntimeError(
                f"Processed chunk mismatch for {self.spec.key}: {actions.shape}"
            )
        return CurrentInferenceOutput(
            actions=actions,
            raw_actions=raw_actions,
            model_latency_seconds=model_latency,
            raw_shape=tuple(raw_chunk.shape),
            raw_dtype=str(raw_chunk.dtype),
            predicted_inference_delay_steps=predicted_delay_steps,
        )

    def step(self, task_id: int, action: np.ndarray) -> StepOutput:
        command = np.asarray(action, dtype=np.float32)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError(f"Expected one finite 7D command, got {command}")
        observation, reward, terminated, truncated, info = self._env(task_id).step(command[None, :])
        reward_scalar = float(np.asarray(reward).reshape(-1)[0])
        return StepOutput(
            observation=observation,
            reward=reward_scalar,
            terminated=bool(np.asarray(terminated).reshape(-1)[0]),
            truncated=bool(np.asarray(truncated).reshape(-1)[0]),
            success=extract_success(info, reward_scalar),
            info=info,
        )

    @property
    def peak_cuda_memory_mib(self) -> float:
        return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0

    def close(self) -> None:
        for task_map in self.envs.values():
            for env in task_map.values():
                env.close()


class _RuntimeQueue:
    def queue_depth(self) -> int:
        raise NotImplementedError

    def should_request(self, control_step: int) -> bool:
        raise NotImplementedError

    def merge(self, result: InferenceResult, control_step: int) -> dict[str, Any]:
        raise NotImplementedError

    def pop(self) -> tuple[np.ndarray, bool]:
        raise NotImplementedError

    def rtc_context(self) -> tuple[np.ndarray | None, int | None]:
        return None, None


class _OfficialAsyncQueue(_RuntimeQueue):
    def __init__(self, bindings: Any, *, aggregate: str, period: float, chunk_size: int) -> None:
        self.adapter = OfficialLeRobotAdapter(
            bindings,
            aggregate_name=aggregate,
            environment_dt=period,
            chunk_horizon=chunk_size,
        )
        self.adapter.client.robot.action_features = {
            f"action_{index}": float for index in range(7)
        }
        self._last_action: np.ndarray | None = None
        self.hold_steps = 0
        self.merges = 0

    def queue_depth(self) -> int:
        return len(self.adapter.queue_items())

    def should_request(self, control_step: int) -> bool:
        return self.adapter.ready_at_default_threshold()

    def merge(self, result: InferenceResult, control_step: int) -> dict[str, Any]:
        before = self.queue_depth()
        timed = self.adapter.time_action_chunk(
            capture_timestamp=result.request_timestamp,
            observation_timestep=result.observation_control_step,
            actions=list(result.actions),
        )
        self.adapter.aggregate(timed)
        self.merges += 1
        items = self.adapter.queue_items()
        return {
            "queue_before": before,
            "queue_after": len(items),
            "incoming": len(timed),
            "latest_action_timestep": int(self.adapter.client.latest_action),
            "queued_timesteps": [int(item.get_timestep()) for item in items],
            "dropped_prefix_steps": sum(
                item.get_timestep() <= self.adapter.client.latest_action for item in timed
            ),
        }

    def pop(self) -> tuple[np.ndarray, bool]:
        if self.queue_depth():
            _, action = self.adapter.pop_and_send()
            self._last_action = action.copy()
            return action, False
        if self._last_action is None:
            raise QueueNotReady("Official Async queue has no first action")
        self.hold_steps += 1
        return self._last_action.copy(), True


class _AlignedQueue(_RuntimeQueue):
    def __init__(self, request_interval_steps: int) -> None:
        self.queue = ActionStreamQueue()
        self.queue.reset_episode(uuid.uuid4().hex)
        self.episode_id = self.queue._episode_id
        self.request_interval_steps = request_interval_steps
        self._last_request_step: int | None = None

    def queue_depth(self) -> int:
        return self.queue.queue_length

    def should_request(self, control_step: int) -> bool:
        return (
            control_step % self.request_interval_steps == 0
            and control_step != self._last_request_step
        )

    def note_request(self, control_step: int) -> None:
        self._last_request_step = control_step

    def merge(self, result: InferenceResult, control_step: int) -> dict[str, Any]:
        before = self.queue.queue_length
        outcome = self.queue.replace(
            result,
            current_control_step=control_step,
            mode="async_aligned",
        )
        return {
            "queue_before": before,
            "queue_after": outcome.queue_length,
            "incoming": outcome.incoming_chunk_steps,
            "accepted": outcome.accepted,
            "reason": outcome.reason,
            "age_steps": outcome.age_steps,
            "dropped_prefix_steps": outcome.dropped_prefix_steps,
        }

    def pop(self) -> tuple[np.ndarray, bool]:
        return self.queue.next_action()

    @property
    def hold_steps(self) -> int:
        return self.queue.hold_steps


class _AdaptiveActionQueue(ActionStreamQueue):
    """Adaptive-only queue extension that preserves the frozen core runtime."""

    def discard_pending_for_hold(self) -> int:
        discarded = len(self._queue)
        self._queue.clear()
        return discarded


class _AdaptiveQueue(_RuntimeQueue):
    def __init__(
        self,
        *,
        spec: ModelSpec,
        selector_config: AdaptiveSelectorConfig,
    ) -> None:
        self.queue = _AdaptiveActionQueue()
        self.request_interval_steps = spec.request_interval_steps
        self.selector = AdaptiveSelector(
            selector_config,
            chunk_size=spec.chunk_size,
            control_mode=spec.control_mode,
        )
        self._last_request_step: int | None = None
        self._last_action: np.ndarray | None = None
        self._last_execution_mode: str | None = None
        self.decision_counts: Counter[str] = Counter()
        self.regime_counts: Counter[str] = Counter()
        self.mode_switches = 0
        self.safe_alternate_overrides = 0
        self.safe_hold_merges = 0
        self.pending_actions_discarded_by_guard = 0
        self.reset_episode(uuid.uuid4().hex)

    def reset_episode(self, episode_id: str) -> None:
        self.queue.reset_episode(episode_id)
        self.selector.reset()
        self._last_request_step = None
        self._last_action = None
        self._last_execution_mode = None
        self.decision_counts.clear()
        self.regime_counts.clear()
        self.mode_switches = 0
        self.safe_alternate_overrides = 0
        self.safe_hold_merges = 0
        self.pending_actions_discarded_by_guard = 0

    def queue_depth(self) -> int:
        return self.queue.queue_length

    def should_request(self, control_step: int) -> bool:
        return (
            control_step % self.request_interval_steps == 0
            and control_step != self._last_request_step
        )

    def note_request(self, control_step: int) -> None:
        self._last_request_step = control_step

    def _record_decision(self, mode: str, regime: str, *, override: bool) -> None:
        self.decision_counts[mode] += 1
        self.regime_counts[regime] += 1
        if self._last_execution_mode is not None and mode != self._last_execution_mode:
            self.mode_switches += 1
        self._last_execution_mode = mode
        if override:
            self.safe_alternate_overrides += 1

    def merge(self, result: InferenceResult, control_step: int) -> dict[str, Any]:
        before = self.queue.queue_length
        decision = self.selector.decide(
            result.actions,
            observation_control_step=result.observation_control_step,
            current_control_step=control_step,
            queue_depth_steps=before,
            last_action=self._last_action,
        )
        self._record_decision(
            decision.execution_mode,
            decision.regime,
            override=decision.safe_alternate_override,
        )
        if decision.execution_mode == "safe_hold":
            if not self.queue.has_safe_action:
                raise RuntimeError(
                    "Adaptive risk gate rejected the first chunk; no safe hold action exists"
                )
            discarded = self.queue.discard_pending_for_hold()
            self.pending_actions_discarded_by_guard += discarded
            self.safe_hold_merges += 1
            return {
                "queue_before": before,
                "queue_after": self.queue.queue_length,
                "incoming": len(result.actions),
                "accepted": False,
                "reason": "adaptive_safe_hold",
                "age_steps": decision.result_age_steps,
                "dropped_prefix_steps": len(result.actions),
                "guard_discarded_pending_steps": discarded,
                "adaptive_decision": decision.as_dict(),
            }
        if decision.merge_mode is None:
            raise RuntimeError("Adaptive executable decision is missing a merge mode")
        outcome = self.queue.replace(
            result,
            current_control_step=control_step,
            mode=decision.merge_mode,
        )
        return {
            "queue_before": before,
            "queue_after": outcome.queue_length,
            "incoming": outcome.incoming_chunk_steps,
            "accepted": outcome.accepted,
            "reason": outcome.reason,
            "age_steps": outcome.age_steps,
            "dropped_prefix_steps": outcome.dropped_prefix_steps,
            "adaptive_decision": decision.as_dict(),
        }

    def pop(self) -> tuple[np.ndarray, bool]:
        action, held = self.queue.next_action()
        self._last_action = action.copy()
        return action, held

    @property
    def hold_steps(self) -> int:
        return self.queue.hold_steps

    def summary(self) -> dict[str, Any]:
        return {
            "adaptive_selector_id": self.selector.config.selector_id,
            "adaptive_selector_sha256": self.selector.config.source_sha256,
            "adaptive_decision_counts": dict(sorted(self.decision_counts.items())),
            "adaptive_regime_counts": dict(sorted(self.regime_counts.items())),
            "adaptive_mode_switches": self.mode_switches,
            "adaptive_safe_alternate_overrides": self.safe_alternate_overrides,
            "adaptive_safe_hold_merges": self.safe_hold_merges,
            "adaptive_pending_actions_discarded_by_guard": (
                self.pending_actions_discarded_by_guard
            ),
        }


class _OfficialRTCQueue(_RuntimeQueue):
    def __init__(
        self,
        *,
        execution_horizon: int,
        chunk_size: int,
        request_interval_steps: int,
        period: float,
    ) -> None:
        from lerobot.policies.rtc import ActionQueue
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        self.queue = ActionQueue(
            RTCConfig(enabled=True, execution_horizon=int(execution_horizon))
        )
        self.execution_horizon = int(execution_horizon)
        self.threshold = max(
            self.execution_horizon,
            int(chunk_size) - 2 * int(request_interval_steps),
        )
        self.period = float(period)
        self._last_action: np.ndarray | None = None
        self.hold_steps = 0

    def queue_depth(self) -> int:
        return int(self.queue.qsize())

    def should_request(self, control_step: int) -> bool:
        return self.queue_depth() <= self.threshold

    def rtc_context(self) -> tuple[np.ndarray | None, int | None]:
        prefix = self.queue.get_left_over()
        prefix_array = None if prefix is None else prefix.detach().cpu().numpy().copy()
        return prefix_array, int(self.queue.get_action_index())

    def merge(self, result: InferenceResult, control_step: int) -> dict[str, Any]:
        raw = np.asarray(result.metadata["raw_actions"])
        before = self.queue_depth()
        real_delay = max(
            0,
            math.ceil((result.delivery_timestamp - result.start_timestamp) / self.period),
        )
        self.queue.merge(
            torch.as_tensor(raw),
            torch.as_tensor(result.actions),
            real_delay,
            result.metadata.get("action_index_before_inference"),
            task=str(result.metadata.get("task_instruction", "")),
        )
        return {
            "queue_before": before,
            "queue_after": self.queue_depth(),
            "incoming": len(result.actions),
            "real_delay_steps": real_delay,
            "predicted_inference_delay_steps": result.metadata.get(
                "predicted_inference_delay_steps"
            ),
            "dropped_prefix_steps": min(real_delay, len(result.actions)),
            "action_index_before_inference": result.metadata.get(
                "action_index_before_inference"
            ),
        }

    def pop(self) -> tuple[np.ndarray, bool]:
        action = self.queue.get()
        if action is not None:
            array = action.detach().cpu().numpy().astype(np.float32, copy=True)
            self._last_action = array
            return array, False
        if self._last_action is None:
            raise QueueNotReady("Official RTC queue has no first action")
        self.hold_steps += 1
        return self._last_action.copy(), True


def _make_runtime_queue(
    runtime: str,
    *,
    bindings: Any,
    spec: ModelSpec,
    period: float,
    adaptive_selector: AdaptiveSelectorConfig | None = None,
) -> _RuntimeQueue:
    if runtime in OFFICIAL_ASYNC_AGGREGATES:
        return _OfficialAsyncQueue(
            bindings,
            aggregate=OFFICIAL_ASYNC_AGGREGATES[runtime],
            period=period,
            chunk_size=spec.chunk_size,
        )
    if runtime == "actionstream_aligned":
        return _AlignedQueue(spec.request_interval_steps)
    if runtime == "actionstream_adaptive":
        if adaptive_selector is None:
            raise ValueError("Adaptive runtime requires a frozen selector config")
        return _AdaptiveQueue(spec=spec, selector_config=adaptive_selector)
    if runtime == "lerobot_rtc":
        return _OfficialRTCQueue(
            execution_horizon=spec.rtc_execution_horizon,
            chunk_size=spec.chunk_size,
            request_interval_steps=spec.request_interval_steps,
            period=period,
        )
    raise ValueError(f"No asynchronous queue for {runtime}")


def _write_trace(path: Path, actions: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "actions": list(actions), "inference_events": list(events)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _capture_frame(observation: Mapping[str, Any]) -> np.ndarray:
    from actionstream.m4_capture import _extract_agentview_frame

    return _extract_agentview_frame(observation)


def _write_video(
    path: Path,
    frames: Sequence[np.ndarray],
    *,
    runtime: str,
    model_key: str,
    profile_key: str,
    success: bool,
    fps: int,
) -> None:
    import av
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 640
    stream.height = 640
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "medium"}
    try:
        for index, raw in enumerate(frames):
            image = Image.fromarray(raw, mode="RGB").resize((560, 560))
            canvas = Image.new("RGB", (640, 640), (8, 15, 28))
            canvas.paste(image, (40, 58))
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 12), f"{model_key} | {runtime} | {profile_key}", fill=(235, 240, 247))
            draw.text(
                (20, 34),
                f"step {index:03d} | {'SUCCESS' if success else 'RUN/FAIL'}",
                fill=(64, 210, 135) if success else (255, 177, 66),
            )
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _base_record(
    *,
    experiment_id: str,
    run_id: str,
    backend: CurrentLeRobotBackend,
    runtime: str,
    profile: DelayTrace,
    task_id: int,
    episode_index: int,
    initial_state_index: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "source_commit": _git_commit(),
        "model_key": backend.spec.key,
        "model_id": backend.spec.model_id,
        "model_revision": backend.spec.revision,
        "control_mode": backend.spec.control_mode,
        "runtime": runtime,
        "delay_profile": profile.key,
        "delay_trace_sha256": profile.trace_sha256,
        "task_id": task_id,
        "episode_index": episode_index,
        "initial_state_index": initial_state_index,
        "seed": seed,
    }


def run_async_episode(
    backend: CurrentLeRobotBackend,
    *,
    experiment_id: str,
    bindings: Any,
    runtime: str,
    adaptive_selector: AdaptiveSelectorConfig | None,
    profile: DelayTrace,
    task_id: int,
    episode_index: int,
    initial_state_index: int,
    seed: int,
    run_id: str,
    trace_path: Path,
    video_path: Path | None = None,
) -> dict[str, Any]:
    observation, _, instruction = backend.reset_episode(
        task_id=task_id,
        seed=seed,
        initial_state_index=initial_state_index,
    )
    fps = backend.controller_frequency_hz(task_id)
    period = 1.0 / fps
    queue = _make_runtime_queue(
        runtime,
        bindings=bindings,
        spec=backend.spec,
        period=period,
        adaptive_selector=adaptive_selector,
    )
    episode_id = f"{run_id}-{runtime}-{backend.spec.key}-{task_id}-{episode_index}"
    if isinstance(queue, _AlignedQueue):
        queue.queue.reset_episode(episode_id)
        queue.episode_id = episode_id
    elif isinstance(queue, _AdaptiveQueue):
        queue.reset_episode(episode_id)

    def infer(request: InferenceRequest) -> InferencePayload:
        thawed = thaw_observation_snapshot(request.observation)
        prefix: np.ndarray | None = None
        action_index: int | None = None
        if runtime == "lerobot_rtc":
            prefix, action_index = queue.rtc_context()
        output = backend.infer_action_chunk(
            thawed,
            request.task_instruction,
            rtc_prefix=prefix,
            rtc_execution_horizon=(
                backend.spec.rtc_execution_horizon if runtime == "lerobot_rtc" else None
            ),
            controller_period_seconds=period if runtime == "lerobot_rtc" else None,
        )
        return InferencePayload(
            actions=output.actions,
            model_inference_latency_seconds=output.model_latency_seconds,
            metadata={
                "raw_actions": output.raw_actions,
                "raw_shape": list(output.raw_shape),
                "raw_dtype": output.raw_dtype,
                "predicted_inference_delay_steps": output.predicted_inference_delay_steps,
                "action_index_before_inference": action_index,
                "task_instruction": request.task_instruction,
            },
        )

    worker = LatestRequestWorker(
        infer,
        delivery_delay_seconds=lambda _request, ordinal: profile.seconds_at(ordinal),
    )
    worker.reset_episode(episode_id)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    control_step = 0
    success = False
    last_submit_step: int | None = None
    action_rows: list[dict[str, Any]] = []
    inference_events: list[dict[str, Any]] = []
    frames: list[np.ndarray] = [_capture_frame(observation)] if video_path else []
    episode_started = time.monotonic()
    control_epoch: float | None = None
    scheduled: float | None = None
    previous_action: np.ndarray | None = None
    discontinuities: list[float] = []
    acceleration_peaks: list[float] = []
    previous_delta: np.ndarray | None = None

    def submit() -> None:
        nonlocal last_submit_step
        worker.submit(
            InferenceRequest(
                observation=immutable_observation_snapshot(observation),
                task_instruction=instruction,
                episode_id=episode_id,
                observation_control_step=control_step,
                request_timestamp=time.monotonic(),
                queue_depth_at_request_steps=queue.queue_depth(),
                queue_headroom_at_request_steps=queue.queue_depth(),
            )
        )
        last_submit_step = control_step
        if isinstance(queue, (_AlignedQueue, _AdaptiveQueue)):
            queue.note_request(control_step)

    def merge_ready() -> int:
        merged = 0
        for result in worker.drain_results():
            merge = queue.merge(result, control_step)
            inference_events.append(
                {
                    "observation_control_step": result.observation_control_step,
                    "request_timestamp": result.request_timestamp,
                    "start_timestamp": result.start_timestamp,
                    "end_timestamp": result.end_timestamp,
                    "delivery_timestamp": result.delivery_timestamp,
                    "model_inference_latency_seconds": result.model_inference_latency_seconds,
                    "injected_delivery_delay_seconds": result.metadata[
                        "injected_delivery_delay_seconds"
                    ],
                    "delay_trace_index": result.metadata["delay_trace_index"],
                    "merge_control_step": control_step,
                    "merge": merge,
                }
            )
            merged += 1
        return merged

    submit()
    try:
        while control_step < backend.episode_length:
            merge_ready()
            try:
                action, held = queue.pop()
            except QueueNotReady:
                if not worker.wait_for_result(timeout=300):
                    raise TimeoutError("Timed out waiting for the first policy chunk")
                merge_ready()
                action, held = queue.pop()
                control_epoch = time.monotonic()
                scheduled = control_epoch

            if control_epoch is None:
                control_epoch = time.monotonic()
                scheduled = control_epoch
            if scheduled is None:
                raise RuntimeError("Missing real-time schedule")
            remaining = scheduled - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            dispatched = time.monotonic()
            depth_before = queue.queue_depth() + (0 if held else 1)
            step_output = backend.step(task_id, action)
            if previous_action is not None:
                delta = np.asarray(action) - previous_action
                discontinuities.append(float(np.linalg.norm(delta)))
                if previous_delta is not None:
                    acceleration_peaks.append(float(np.linalg.norm(delta - previous_delta)))
                previous_delta = delta
            previous_action = np.asarray(action).copy()
            action_rows.append(
                {
                    "control_step": control_step,
                    "dispatch_timestamp": dispatched,
                    "queue_depth_before": depth_before,
                    "queue_depth_after": queue.queue_depth(),
                    "held": held,
                    "action": np.asarray(action, dtype=np.float32).tolist(),
                }
            )
            control_step += 1
            observation = step_output.observation
            success = success or step_output.success
            if video_path:
                frames.append(_capture_frame(observation))
            if success or step_output.terminated or step_output.truncated:
                break
            scheduled += period

            if queue.should_request(control_step) and last_submit_step != control_step:
                submit()
    finally:
        worker.cancel_pending()
        worker.wait_idle(timeout=300)
        merge_ready()
        worker.close()

    finished = time.monotonic()
    _write_trace(trace_path, action_rows, inference_events)
    if video_path:
        _write_video(
            video_path,
            frames,
            runtime=runtime,
            model_key=backend.spec.key,
            profile_key=profile.key,
            success=success,
            fps=int(round(fps)),
        )
    record = _base_record(
        experiment_id=experiment_id,
        run_id=run_id,
        backend=backend,
        runtime=runtime,
        profile=profile,
        task_id=task_id,
        episode_index=episode_index,
        initial_state_index=initial_state_index,
        seed=seed,
    )
    record.update(
        {
            "status": "completed",
            "success": success,
            "environment_steps": control_step,
            "wall_clock_episode_seconds": finished - episode_started,
            "inference_calls": worker.calls_started,
            "hold_steps": int(getattr(queue, "hold_steps", 0)),
            "hold_fraction": (
                float(getattr(queue, "hold_steps", 0)) / control_step if control_step else None
            ),
            "inference_latency_p50_seconds": _percentile(
                [event["model_inference_latency_seconds"] for event in inference_events], 50
            ),
            "inference_latency_p95_seconds": _percentile(
                [event["model_inference_latency_seconds"] for event in inference_events], 95
            ),
            "delivery_latency_p50_seconds": _percentile(
                [event["delivery_timestamp"] - event["request_timestamp"] for event in inference_events],
                50,
            ),
            "delivery_latency_p95_seconds": _percentile(
                [event["delivery_timestamp"] - event["request_timestamp"] for event in inference_events],
                95,
            ),
            "dropped_prefix_steps": sum(
                int(event["merge"].get("dropped_prefix_steps", 0)) for event in inference_events
            ),
            "action_discontinuity_mean_l2": (
                float(np.mean(discontinuities)) if discontinuities else None
            ),
            "action_discontinuity_max_l2": max(discontinuities, default=None),
            "action_acceleration_max_l2": max(acceleration_peaks, default=None),
            "controller_frequency_hz": fps,
            "chunk_size": backend.spec.chunk_size,
            "request_interval_steps": backend.spec.request_interval_steps,
            "peak_cuda_memory_mib": backend.peak_cuda_memory_mib,
            "trace_path": str(trace_path),
            "trace_sha256": _sha256_file(trace_path),
            "video_path": str(video_path) if video_path else None,
            "video_sha256": _sha256_file(video_path) if video_path else None,
        }
    )
    if isinstance(queue, _AdaptiveQueue):
        record.update(queue.summary())
    return record


def run_sync_episode(
    backend: CurrentLeRobotBackend,
    *,
    experiment_id: str,
    profile: DelayTrace,
    task_id: int,
    episode_index: int,
    initial_state_index: int,
    seed: int,
    run_id: str,
    trace_path: Path,
    video_path: Path | None = None,
) -> dict[str, Any]:
    observation, _, instruction = backend.reset_episode(
        task_id=task_id,
        seed=seed,
        initial_state_index=initial_state_index,
    )
    fps = backend.controller_frequency_hz(task_id)
    period = 1.0 / fps
    pending: list[np.ndarray] = []
    actions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    frames: list[np.ndarray] = [_capture_frame(observation)] if video_path else []
    success = False
    step = 0
    inference_ordinal = 0
    started = time.monotonic()
    previous_action: np.ndarray | None = None
    discontinuities: list[float] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    while step < backend.episode_length:
        if not pending:
            request_time = time.monotonic()
            output = backend.infer_action_chunk(observation, instruction)
            inference_end = time.monotonic()
            injected = profile.seconds_at(inference_ordinal)
            if injected:
                time.sleep(injected)
            delivered = time.monotonic()
            pending.extend(output.actions)
            events.append(
                {
                    "observation_control_step": step,
                    "request_timestamp": request_time,
                    "end_timestamp": inference_end,
                    "delivery_timestamp": delivered,
                    "model_inference_latency_seconds": output.model_latency_seconds,
                    "injected_delivery_delay_seconds": injected,
                    "delay_trace_index": inference_ordinal,
                }
            )
            inference_ordinal += 1
        action = np.asarray(pending.pop(0), dtype=np.float32)
        dispatched = time.monotonic()
        output_step = backend.step(task_id, action)
        if previous_action is not None:
            discontinuities.append(float(np.linalg.norm(action - previous_action)))
        previous_action = action.copy()
        actions.append(
            {
                "control_step": step,
                "dispatch_timestamp": dispatched,
                "queue_depth_before": len(pending) + 1,
                "queue_depth_after": len(pending),
                "held": False,
                "action": action.tolist(),
            }
        )
        step += 1
        observation = output_step.observation
        success = success or output_step.success
        if video_path:
            frames.append(_capture_frame(observation))
        if success or output_step.terminated or output_step.truncated:
            break
        time.sleep(max(0.0, period - (time.monotonic() - dispatched)))

    finished = time.monotonic()
    _write_trace(trace_path, actions, events)
    if video_path:
        _write_video(
            video_path,
            frames,
            runtime="sync_hold",
            model_key=backend.spec.key,
            profile_key=profile.key,
            success=success,
            fps=int(round(fps)),
        )
    record = _base_record(
        experiment_id=experiment_id,
        run_id=run_id,
        backend=backend,
        runtime="sync_hold",
        profile=profile,
        task_id=task_id,
        episode_index=episode_index,
        initial_state_index=initial_state_index,
        seed=seed,
    )
    latencies = [event["model_inference_latency_seconds"] for event in events]
    record.update(
        {
            "status": "completed",
            "success": success,
            "environment_steps": step,
            "wall_clock_episode_seconds": finished - started,
            "inference_calls": len(events),
            "hold_steps": 0,
            "hold_fraction": 0.0,
            "inference_latency_p50_seconds": _percentile(latencies, 50),
            "inference_latency_p95_seconds": _percentile(latencies, 95),
            "delivery_latency_p50_seconds": _percentile(
                [event["delivery_timestamp"] - event["request_timestamp"] for event in events], 50
            ),
            "delivery_latency_p95_seconds": _percentile(
                [event["delivery_timestamp"] - event["request_timestamp"] for event in events], 95
            ),
            "dropped_prefix_steps": 0,
            "action_discontinuity_mean_l2": (
                float(np.mean(discontinuities)) if discontinuities else None
            ),
            "action_discontinuity_max_l2": max(discontinuities, default=None),
            "action_acceleration_max_l2": None,
            "controller_frequency_hz": fps,
            "chunk_size": backend.spec.chunk_size,
            "request_interval_steps": backend.spec.chunk_size,
            "peak_cuda_memory_mib": backend.peak_cuda_memory_mib,
            "trace_path": str(trace_path),
            "trace_sha256": _sha256_file(trace_path),
            "video_path": str(video_path) if video_path else None,
            "video_sha256": _sha256_file(video_path) if video_path else None,
        }
    )
    return record


def _selected(values: Sequence[str] | None, available: Mapping[str, Any], label: str) -> list[str]:
    selected = list(values) if values else list(available)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown {label}: {sorted(unknown)}")
    return selected


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    protocol = load_protocol(args.protocol)
    source_receipt = verify_upstream_checkout(args.lerobot_root, protocol.raw["source"])
    model_keys = _selected(args.models, protocol.models, "models")
    profile_keys = _selected(args.profiles, protocol.delays, "profiles")
    runtimes = list(args.runtimes or protocol.raw["runtimes"])
    unknown_runtimes = set(runtimes) - set(RUNTIMES)
    if unknown_runtimes:
        raise ValueError(f"Unknown runtimes: {sorted(unknown_runtimes)}")
    environment = protocol.raw["environment"]
    task_ids = list(args.task_ids or environment["task_ids"])
    state_indices = list(args.initial_state_indices or environment["initial_state_indices"])
    episodes = args.episodes_per_task or int(protocol.raw["compact_matrix"]["episodes_per_task"])
    if len(state_indices) < episodes:
        raise ValueError("Not enough initial-state indices for requested episodes")
    episode_length = int(args.episode_length or environment["episode_length"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "episodes.jsonl"
    receipt_path = output_dir / "run_receipt.json"
    if not args.append and (metrics_path.exists() or receipt_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing run in {output_dir}")

    bindings = load_upstream_bindings(args.lerobot_root)
    run_id = args.run_id or f"current-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    records: list[dict[str, Any]] = []
    compatibility: list[dict[str, Any]] = []
    base_seed = int(environment["base_seed"])
    mode = "a" if args.append else "w"
    with metrics_path.open(mode, encoding="utf-8", buffering=1) as stream:
        for model_key in model_keys:
            spec = protocol.models[model_key]
            backend = CurrentLeRobotBackend(
                spec=spec,
                task_ids=task_ids,
                suite=str(environment["suite"]),
                episode_length=episode_length,
                seed=base_seed,
                device=args.device,
            )
            try:
                actual_rtc = backend.supports_rtc
                compatibility.append(
                    {
                        "model_key": model_key,
                        "model_id": spec.model_id,
                        "revision": spec.revision,
                        "rtc_expected": spec.rtc_expected,
                        "rtc_supported": actual_rtc,
                        "rtc_expectation_match": actual_rtc is spec.rtc_expected,
                    }
                )
                if actual_rtc is not spec.rtc_expected:
                    raise RuntimeError(
                        f"RTC capability drift for {model_key}: expected={spec.rtc_expected}, actual={actual_rtc}"
                    )
                for profile_key in profile_keys:
                    profile = protocol.delays[profile_key]
                    for runtime in runtimes:
                        if runtime == "lerobot_rtc" and not actual_rtc:
                            compatibility.append(
                                {
                                    "model_key": model_key,
                                    "runtime": runtime,
                                    "status": "not_applicable",
                                    "reason": "policy.supports_rtc() is false",
                                }
                            )
                            continue
                        backend.configure_rtc(enabled=runtime == "lerobot_rtc")
                        for task_id in task_ids:
                            for episode_index in range(episodes):
                                state = state_indices[episode_index]
                                seed = base_seed + episode_index
                                stem = (
                                    f"{model_key}__{runtime}__{profile_key}__"
                                    f"task{task_id}__ep{episode_index}"
                                )
                                trace_path = output_dir / "traces" / f"{stem}.json"
                                capture = bool(args.capture and task_id == task_ids[0] and episode_index == 0)
                                video_path = output_dir / "videos" / f"{stem}.mp4" if capture else None
                                if runtime == "sync_hold":
                                    record = run_sync_episode(
                                        backend,
                                        experiment_id=str(protocol.raw["experiment_id"]),
                                        profile=profile,
                                        task_id=task_id,
                                        episode_index=episode_index,
                                        initial_state_index=state,
                                        seed=seed,
                                        run_id=run_id,
                                        trace_path=trace_path,
                                        video_path=video_path,
                                    )
                                else:
                                    record = run_async_episode(
                                        backend,
                                        experiment_id=str(protocol.raw["experiment_id"]),
                                        bindings=bindings,
                                        runtime=runtime,
                                        adaptive_selector=protocol.adaptive_selector,
                                        profile=profile,
                                        task_id=task_id,
                                        episode_index=episode_index,
                                        initial_state_index=state,
                                        seed=seed,
                                        run_id=run_id,
                                        trace_path=trace_path,
                                        video_path=video_path,
                                    )
                                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                                records.append(record)
                                print(
                                    f"{model_key} {runtime} {profile_key} task={task_id} "
                                    f"episode={episode_index} success={record['success']} "
                                    f"steps={record['environment_steps']}",
                                    flush=True,
                                )
            finally:
                backend.close()

    receipt = {
        "schema_version": 1,
        "experiment_id": protocol.raw["experiment_id"],
        "run_id": run_id,
        "source_commit": _git_commit(),
        "protocol_path": str(Path(args.protocol).resolve()),
        "protocol_sha256": _sha256_file(Path(args.protocol)),
        "upstream": source_receipt,
        "compatibility": compatibility,
        "adaptive_selector": (
            None
            if protocol.adaptive_selector is None
            else {
                "selector_id": protocol.adaptive_selector.selector_id,
                "selector_protocol_sha256": protocol.adaptive_selector.source_sha256,
            }
        ),
        "selection": {
            "models": model_keys,
            "runtimes": runtimes,
            "profiles": profile_keys,
            "task_ids": task_ids,
            "episodes_per_task": episodes,
            "initial_state_indices": state_indices[:episodes],
            "episode_length": episode_length,
        },
        "record_count": len(records),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256_file(metrics_path),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/current_lerobot_baselines.json"),
    )
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", type=_parse_csv)
    parser.add_argument("--runtimes", type=_parse_csv)
    parser.add_argument("--profiles", type=_parse_csv)
    parser.add_argument("--task-ids", type=_parse_int_csv)
    parser.add_argument("--initial-state-indices", type=_parse_int_csv)
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--capture", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
