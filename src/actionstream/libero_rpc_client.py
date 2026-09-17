"""Environment-only LIBERO client for one remote ActionStream/X-VLA episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from actionstream.lerobot_backend import immutable_observation_snapshot
from actionstream.lerobot_inference import (
    ActionStreamInferenceConfig,
    ActionStreamInferenceEngine,
)
from actionstream.libero_config import ensure_isolated_libero_config
from actionstream.rpc_transport import TcpInferenceTransport


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _bool_at_zero(value: Any) -> bool:
    if value is None:
        return False
    array = np.asarray(value)
    return bool(array.reshape(-1)[0]) if array.size else False


def _extract_success(info: dict[str, Any], reward: float) -> bool:
    if reward > 0 or _bool_at_zero(info.get("is_success")):
        return True
    final_info = info.get("final_info")
    if isinstance(final_info, dict):
        return _bool_at_zero(final_info.get("is_success"))
    if final_info is not None:
        for item in np.asarray(final_info, dtype=object).reshape(-1):
            if isinstance(item, dict) and bool(item.get("is_success", False)):
                return True
    return False


class LiberoEnvironmentClient:
    """LIBERO environment without policy weights or policy processors."""

    def __init__(
        self,
        *,
        suite: str,
        task_id: int,
        episode_length: int,
        seed: int,
    ) -> None:
        ensure_isolated_libero_config()
        if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get(
            "PYOPENGL_PLATFORM"
        ) != "egl":
            raise RuntimeError(
                "LIBERO RPC client requires MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl"
            )
        from lerobot.envs import make_env, make_env_config
        from lerobot.utils.random_utils import set_seed

        self._set_seed = set_seed
        self._set_seed(seed)
        self.suite = suite
        self.task_id = int(task_id)
        self.env_cfg = make_env_config(
            "libero",
            task=suite,
            task_ids=[self.task_id],
            control_mode="absolute",
            episode_length=episode_length,
            max_parallel_tasks=1,
        )
        self.envs = make_env(self.env_cfg, n_envs=1, use_async_envs=False)

    def _env(self):
        return self.envs[self.suite][self.task_id]

    def _sub_env(self):
        env = self._env()
        if len(env.envs) != 1:
            raise RuntimeError("Expected exactly one synchronous LIBERO environment")
        return env.envs[0]

    def reset(self, *, seed: int, initial_state_index: int):
        sub_env = self._sub_env()
        if not 0 <= initial_state_index < len(sub_env._init_states):
            raise ValueError(
                f"Initial-state index {initial_state_index} is unavailable for "
                f"{self.suite}/{self.task_id}"
            )
        sub_env.init_state_id = int(initial_state_index)
        self._set_seed(int(seed))
        observation, info = self._env().reset(seed=[int(seed)])
        instruction = str(sub_env.task_description)
        if sub_env._env is None:
            raise RuntimeError("LIBERO environment did not initialize")
        frequency = float(sub_env._env.env.control_freq)
        if frequency <= 0:
            raise RuntimeError("Invalid LIBERO controller frequency")
        if any(robot.controller.use_delta for robot in sub_env._env.env.robots):
            raise RuntimeError("Remote benchmark requires absolute action mode")
        return observation, info, instruction, frequency

    def step(self, action: np.ndarray):
        command = np.asarray(action, dtype=np.float32)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError(f"Expected one finite 7D action, got {command.shape}")
        observation, reward, terminated, truncated, info = self._env().step(
            command[None, :]
        )
        reward_scalar = float(np.asarray(reward).reshape(-1)[0])
        return SimpleNamespace(
            observation=observation,
            reward=reward_scalar,
            terminated=_bool_at_zero(terminated),
            truncated=_bool_at_zero(truncated),
            success=_extract_success(info, reward_scalar),
            info=info,
        )

    def close(self) -> None:
        for task_map in self.envs.values():
            for env in task_map.values():
                env.close()


class _ResetStub:
    def reset(self) -> None:
        return None


def run_episode(args) -> dict[str, Any]:
    telemetry_path = args.output / "engine_telemetry.jsonl"
    transport = TcpInferenceTransport(
        args.host,
        args.port,
        connect_timeout_s=args.connect_timeout_s,
        control_timeout_s=args.control_timeout_s,
    )
    config = ActionStreamInferenceConfig(
        inference_timeout_s=args.inference_timeout_s,
        bounded_hold_steps=args.bounded_hold_steps,
        retry_backoff_s=args.retry_backoff_s,
        max_consecutive_failures=args.max_consecutive_failures,
        join_timeout_s=args.join_timeout_s,
        latest_only_fallback=args.latest_only_fallback,
        transport_mode="direct",
        delivery_scheduler_enabled=False,
        minimum_request_interval_steps=args.minimum_request_interval_steps,
        telemetry_jsonl_path=str(telemetry_path),
    )
    reset_stub = _ResetStub()
    engine = ActionStreamInferenceEngine(
        policy=reset_stub,
        preprocessor=reset_stub,
        postprocessor=reset_stub,
        hw_features={},
        task="remote-libero",
        device="cpu",
        robot_type="libero",
        config=config,
        infer_chunk=lambda observation, task: (_ for _ in ()).throw(
            AssertionError("Injected TCP transport must own inference")
        ),
        reset_provider=lambda: None,
        transport=transport,
    )
    env = LiberoEnvironmentClient(
        suite=args.suite,
        task_id=args.task_id,
        episode_length=args.max_control_steps,
        seed=args.seed,
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "suite": args.suite,
        "task_id": args.task_id,
        "initial_state_index": args.initial_state_index,
        "seed": args.seed,
        "server": {"host": args.host, "port": args.port},
        "status": "NOT_RUN",
        "success": None,
        "control_steps": 0,
        "wall_s": None,
        "instruction": None,
        "engine_config": asdict(config),
    }
    started = time.perf_counter()
    try:
        engine.start()
        engine.resume()
        engine.reset()
        observation, _, instruction, frequency = env.reset(
            seed=args.seed,
            initial_state_index=args.initial_state_index,
        )
        receipt["instruction"] = instruction
        receipt["controller_frequency_hz"] = frequency
        if not math.isclose(frequency, args.expected_frequency_hz, rel_tol=0, abs_tol=1e-6):
            raise RuntimeError(
                f"Controller frequency changed: {frequency} != {args.expected_frequency_hz}"
            )
        engine.set_task(instruction)
        step_latencies = []
        waiting_ms = []
        success = terminated = truncated = False
        for step in range(args.max_control_steps):
            tick = time.perf_counter()
            snapshot = immutable_observation_snapshot(observation)
            engine.notify_observation(dict(snapshot))
            wait_started = time.perf_counter()
            action = None
            while action is None:
                if engine.failed:
                    raise RuntimeError(
                        f"ActionStream engine failed: {engine.failure_traceback}"
                    )
                action = engine.get_action(None)
                if action is None:
                    if time.perf_counter() - wait_started > args.action_wait_timeout_s:
                        raise TimeoutError("No remote action became available before wait timeout")
                    time.sleep(args.action_poll_s)
            waiting_ms.append(1000 * (time.perf_counter() - wait_started))
            result = env.step(action.detach().cpu().numpy())
            receipt["control_steps"] = step + 1
            observation = result.observation
            success = result.success
            terminated = result.terminated
            truncated = result.truncated
            step_latencies.append(1000 * (time.perf_counter() - tick))
            if success or terminated or truncated:
                break
            remaining = 1 / frequency - (time.perf_counter() - tick)
            if remaining > 0:
                time.sleep(remaining)
        receipt["success"] = bool(success)
        receipt["status"] = "COMPLETED"
        receipt["end_reason"] = (
            "success"
            if success
            else "terminated"
            if terminated
            else "truncated"
            if truncated
            else "control_step_limit"
        )
        receipt["timing"] = {
            "action_wait_p50_ms": float(np.median(waiting_ms)) if waiting_ms else None,
            "action_wait_p95_ms": float(np.quantile(waiting_ms, 0.95))
            if waiting_ms
            else None,
            "control_wall_p50_ms": float(np.median(step_latencies))
            if step_latencies
            else None,
            "control_wall_p95_ms": float(np.quantile(step_latencies, 0.95))
            if step_latencies
            else None,
        }
        receipt["engine_telemetry"] = engine.telemetry.to_dict()
        receipt["rpc_telemetry"] = transport.telemetry().to_dict()
    except BaseException as exc:
        receipt.update(
            status="ERROR",
            success=None,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        receipt["wall_s"] = time.perf_counter() - started
        try:
            engine.stop()
        except BaseException as stop_exc:
            receipt["stop_error"] = f"{type(stop_exc).__name__}: {stop_exc}"
        env.close()
        if telemetry_path.exists():
            receipt["engine_telemetry_sha256"] = _sha256(telemetry_path)
        (args.output / "episode_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--initial-state-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-control-steps", type=int, default=300)
    parser.add_argument("--expected-frequency-hz", type=float, default=20.0)
    parser.add_argument("--inference-timeout-s", type=float, default=5.0)
    parser.add_argument("--connect-timeout-s", type=float, default=3.0)
    parser.add_argument("--control-timeout-s", type=float, default=3.0)
    parser.add_argument("--action-wait-timeout-s", type=float, default=10.0)
    parser.add_argument("--action-poll-s", type=float, default=0.002)
    parser.add_argument("--bounded-hold-steps", type=int, default=2)
    parser.add_argument("--retry-backoff-s", type=float, default=0.05)
    parser.add_argument("--max-consecutive-failures", type=int, default=10)
    parser.add_argument("--join-timeout-s", type=float, default=3.0)
    parser.add_argument("--minimum-request-interval-steps", type=int, default=1)
    parser.add_argument(
        "--latest-only-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run_episode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
