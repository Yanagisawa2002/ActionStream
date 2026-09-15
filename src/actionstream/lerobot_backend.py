"""Thin, pinned LeRobot/LIBERO adapter for custom ActionStream runners."""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from actionstream.libero_config import ensure_isolated_libero_config
from actionstream.processors import postprocess_action_chunk_per_timestep


MODEL_ID = "lerobot/xvla-libero"
MODEL_REVISION = "12e8783e996944f5c97e490d37d4c145484ed70a"


@dataclass(frozen=True)
class InferenceOutput:
    actions: np.ndarray
    model_latency_seconds: float
    raw_shape: tuple[int, ...]
    raw_dtype: str


@dataclass(frozen=True)
class StepOutput:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    info: dict[str, Any]


def immutable_observation_snapshot(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    """Deep-copy and recursively freeze an observation request payload."""

    def freeze(value: Any) -> Any:
        if isinstance(value, Mapping):
            return MappingProxyType({key: freeze(child) for key, child in value.items()})
        if isinstance(value, np.ndarray):
            copied = value.copy()
            copied.setflags(write=False)
            return copied
        if isinstance(value, list | tuple):
            return tuple(freeze(child) for child in value)
        return copy.deepcopy(value)

    return freeze(observation)


def thaw_observation_snapshot(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Make a worker-local writable observation from an immutable request."""

    def thaw(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: thaw(child) for key, child in value.items()}
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, tuple):
            return [thaw(child) for child in value]
        return copy.deepcopy(value)

    return thaw(observation)


def _bool_at_zero(value: Any) -> bool:
    if value is None:
        return False
    array = np.asarray(value)
    return bool(array.reshape(-1)[0]) if array.size else False


def extract_success(info: dict[str, Any], reward: float) -> bool:
    """Read success across Gymnasium vector-info layouts."""
    if reward > 0:
        return True

    if _bool_at_zero(info.get("is_success")):
        return True

    final_info = info.get("final_info")
    if isinstance(final_info, dict):
        return _bool_at_zero(final_info.get("is_success"))
    if final_info is not None:
        for item in np.asarray(final_info, dtype=object).reshape(-1):
            if isinstance(item, dict) and bool(item.get("is_success", False)):
                return True
    return False


class LeRobotBackend:
    """Own the exact policy, processor pipelines, and synchronous vector envs."""

    def __init__(
        self,
        *,
        task_ids: list[int],
        seed: int = 142,
        suite: str = "libero_object",
        episode_length: int = 800,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
        device: str = "cuda",
        tokenizer_path: str | None = None,
    ) -> None:
        ensure_isolated_libero_config()
        if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
            raise RuntimeError("LeRobotBackend requires MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl")

        # These imports may initialize LIBERO, and therefore must remain after
        # ensure_isolated_libero_config().
        from lerobot.configs import PreTrainedConfig
        from lerobot.envs import make_env, make_env_config, make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.utils.random_utils import set_seed

        set_seed(seed)
        # Match the CUDA math settings enabled by LeRobot 0.6.0
        # ``lerobot_eval.eval_main`` before it creates the policy.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        self.seed = seed
        self._set_seed = set_seed
        self.suite = suite
        self.task_ids = list(task_ids)
        self.episode_length = episode_length
        self.model_id = model_id
        self.model_revision = model_revision

        self.env_cfg = make_env_config(
            "libero",
            task=suite,
            task_ids=self.task_ids,
            control_mode="absolute",
            episode_length=episode_length,
            max_parallel_tasks=1,
        )
        policy_cfg = PreTrainedConfig.from_pretrained(model_id, revision=model_revision)
        policy_cfg.device = device
        policy_cfg.pretrained_path = Path(model_id)
        policy_cfg.pretrained_revision = model_revision
        if policy_cfg.chunk_size != 30 or policy_cfg.n_action_steps != 30:
            raise RuntimeError(
                "ActionStream parity requires checkpoint chunk_size=n_action_steps=30, "
                f"got {policy_cfg.chunk_size}/{policy_cfg.n_action_steps}"
            )
        self.policy_cfg = policy_cfg

        self.envs = make_env(self.env_cfg, n_envs=1, use_async_envs=False)
        self.policy = make_policy(cfg=policy_cfg, env_cfg=self.env_cfg, rename_map={}).eval()
        processor_overrides = {
            "device_processor": {"device": str(self.policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        }
        if tokenizer_path is not None:
            processor_overrides["tokenizer_processor"] = {
                "tokenizer_name": str(Path(tokenizer_path).resolve())
            }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=model_id,
            pretrained_revision=model_revision,
            preprocessor_overrides=processor_overrides,
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=self.env_cfg,
            policy_cfg=policy_cfg,
        )

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

    def reset_runtime(self) -> None:
        """Clear all state that could leak across episodes."""
        self.policy.reset()
        for pipeline in (
            self.env_preprocessor,
            self.preprocessor,
            self.postprocessor,
            self.env_postprocessor,
        ):
            pipeline.reset()

    def reset_episode(
        self,
        *,
        task_id: int,
        seed: int,
        initial_state_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        sub_env = self._sub_env(task_id)
        sub_env.init_state_id = int(initial_state_index)
        # X-VLA samples its action chunk from torch.randn. Reset all policy RNG
        # sources per paired episode so differing call counts in one mode cannot
        # shift the random stream of later episodes in another mode.
        self._set_seed(int(seed))
        self.reset_runtime()
        observation, info = self._env(task_id).reset(seed=[int(seed)])
        instruction = str(sub_env.task_description)
        return observation, info, instruction

    def controller_frequency_hz(self, task_id: int) -> float:
        sub_env = self._sub_env(task_id)
        if sub_env._env is None:
            raise RuntimeError("Reset the environment before reading its controller frequency")
        frequency = float(sub_env._env.env.control_freq)
        if frequency <= 0:
            raise RuntimeError(f"Invalid LIBERO controller frequency: {frequency}")
        return frequency

    def prepare_observation(
        self,
        observation: dict[str, Any],
        task_instruction: str,
    ) -> dict[str, Any]:
        from lerobot.envs import preprocess_observation

        prepared = preprocess_observation(observation)
        prepared["task"] = [task_instruction]
        prepared = self.env_preprocessor(prepared)
        return self.preprocessor(prepared)

    def infer_action_chunk(
        self,
        observation: dict[str, Any],
        task_instruction: str,
    ) -> InferenceOutput:
        batch = self.prepare_observation(observation, task_instruction)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            raw_chunk = self.policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        model_latency = time.perf_counter() - started

        final_chunk = postprocess_action_chunk_per_timestep(
            raw_chunk,
            policy_postprocessor=self.postprocessor,
            env_postprocessor=self.env_postprocessor,
        )
        if tuple(final_chunk.shape[:2]) != (1, 30) or final_chunk.shape[-1] != 7:
            raise RuntimeError(f"Unexpected processed chunk shape: {tuple(final_chunk.shape)}")
        actions = final_chunk[0].cpu().numpy().astype(np.float32, copy=True)
        if not np.isfinite(actions).all():
            raise RuntimeError("Official processors produced non-finite environment actions")
        return InferenceOutput(
            actions=actions,
            model_latency_seconds=model_latency,
            raw_shape=tuple(raw_chunk.shape),
            raw_dtype=str(raw_chunk.dtype),
        )

    def step(self, task_id: int, action: np.ndarray) -> StepOutput:
        command = np.asarray(action, dtype=np.float32)
        if command.shape != (7,):
            raise ValueError(f"Expected one final 7D command, got {command.shape}")
        if not np.isfinite(command).all():
            raise ValueError("Refusing to execute a non-finite command")

        observation, reward, terminated, truncated, info = self._env(task_id).step(command[None, :])
        reward_scalar = float(np.asarray(reward).reshape(-1)[0])
        return StepOutput(
            observation=observation,
            reward=reward_scalar,
            terminated=_bool_at_zero(terminated),
            truncated=_bool_at_zero(truncated),
            success=extract_success(info, reward_scalar),
            info=info,
        )

    @property
    def peak_cuda_memory_mib(self) -> float:
        return torch.cuda.max_memory_allocated() / 2**20

    def close(self) -> None:
        for task_map in self.envs.values():
            for env in task_map.values():
                env.close()
