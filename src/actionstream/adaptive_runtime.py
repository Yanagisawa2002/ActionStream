"""Risk-aware execution-mode selection for ActionStream-Adaptive v1."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np


AdaptiveExecutionMode = Literal["full_chunk", "age_aligned", "safe_hold"]
AdaptiveRegime = Literal["fresh", "moderate", "severe_but_bounded", "outage"]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkspaceBounds:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkspaceBounds":
        axes: dict[str, tuple[float, float]] = {}
        for axis in ("x", "y", "z"):
            raw = tuple(float(item) for item in value[axis])
            if len(raw) != 2 or not all(math.isfinite(item) for item in raw):
                raise ValueError(f"Adaptive workspace {axis} must contain two finite bounds")
            if raw[0] >= raw[1]:
                raise ValueError(f"Adaptive workspace {axis} lower bound must be smaller")
            axes[axis] = raw
        return cls(**axes)

    def contains(self, xyz: np.ndarray) -> bool:
        return bool(
            self.x[0] <= xyz[0] <= self.x[1]
            and self.y[0] <= xyz[1] <= self.y[1]
            and self.z[0] <= xyz[2] <= self.z[1]
        )


@dataclass(frozen=True)
class AdaptiveSelectorConfig:
    selector_id: str
    fresh_full_max_age_steps: int
    aligned_max_age_steps: int
    hard_max_age_steps: int
    minimum_aligned_horizon_steps: int
    age_ewma_alpha: float
    workspace: WorkspaceBounds
    maximum_position_disagreement_m: float
    maximum_rotation_geodesic_radians: float
    source_path: str
    source_sha256: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        source_path: Path,
        source_sha256: str,
    ) -> "AdaptiveSelectorConfig":
        if int(value.get("schema_version", -1)) != 1:
            raise ValueError("Expected ActionStream-Adaptive selector schema_version=1")
        decision = value["decision"]
        risk = value["risk_gate"]
        config = cls(
            selector_id=str(value["selector_id"]),
            fresh_full_max_age_steps=int(decision["fresh_full_max_age_steps"]),
            aligned_max_age_steps=int(decision["aligned_max_age_steps"]),
            hard_max_age_steps=int(decision["hard_max_age_steps"]),
            minimum_aligned_horizon_steps=int(
                decision["minimum_aligned_horizon_steps"]
            ),
            age_ewma_alpha=float(decision["age_ewma_alpha"]),
            workspace=WorkspaceBounds.from_mapping(
                risk["absolute_action_workspace"]
            ),
            maximum_position_disagreement_m=float(
                risk["maximum_position_disagreement_m"]
            ),
            maximum_rotation_geodesic_radians=float(
                risk["maximum_rotation_geodesic_radians"]
            ),
            source_path=str(source_path.resolve()),
            source_sha256=source_sha256,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.selector_id:
            raise ValueError("Adaptive selector_id must be nonempty")
        if not (
            0 <= self.fresh_full_max_age_steps
            < self.aligned_max_age_steps
            < self.hard_max_age_steps
        ):
            raise ValueError("Adaptive age thresholds must be strictly increasing")
        if self.minimum_aligned_horizon_steps <= 0:
            raise ValueError("Adaptive minimum aligned horizon must be positive")
        if not 0.0 < self.age_ewma_alpha <= 1.0:
            raise ValueError("Adaptive EWMA alpha must be in (0,1]")
        if not math.isfinite(self.maximum_position_disagreement_m) or (
            self.maximum_position_disagreement_m <= 0
        ):
            raise ValueError("Adaptive position disagreement threshold must be positive")
        if not math.isfinite(self.maximum_rotation_geodesic_radians) or (
            self.maximum_rotation_geodesic_radians <= 0
        ):
            raise ValueError("Adaptive rotation disagreement threshold must be positive")


def load_selector_config(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
) -> AdaptiveSelectorConfig:
    source_path = Path(path)
    actual_sha256 = sha256_file(source_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise ValueError(
            "Adaptive selector protocol hash mismatch: "
            f"expected={expected_sha256.lower()} actual={actual_sha256}"
        )
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    return AdaptiveSelectorConfig.from_mapping(
        raw,
        source_path=source_path,
        source_sha256=actual_sha256,
    )


def rotation_geodesic_radians(first: np.ndarray, second: np.ndarray) -> float:
    def quaternion(axis_angle: np.ndarray) -> np.ndarray:
        vector = np.asarray(axis_angle, dtype=np.float64)
        angle = float(np.linalg.norm(vector))
        if angle <= 1e-12:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        half_angle = angle / 2.0
        return np.concatenate(
            (
                np.asarray([math.cos(half_angle)], dtype=np.float64),
                vector / angle * math.sin(half_angle),
            )
        )

    cosine_half_angle = float(abs(np.dot(quaternion(first), quaternion(second))))
    return float(2.0 * math.acos(np.clip(cosine_half_angle, 0.0, 1.0)))


@dataclass(frozen=True)
class CandidateRisk:
    safe: bool
    reasons: tuple[str, ...]
    position_disagreement_m: float | None
    rotation_geodesic_radians: float | None
    gripper_switch: bool | None
    workspace_inside: bool | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveDecision:
    regime: AdaptiveRegime
    execution_mode: AdaptiveExecutionMode
    merge_mode: Literal["async_naive", "async_aligned"] | None
    reason: str
    result_age_steps: int
    age_ewma_steps: float
    queue_depth_steps: int
    incoming_chunk_steps: int
    selected_action_index: int | None
    preferred_action_index: int | None
    safe_alternate_override: bool
    selected_risk: CandidateRisk | None
    full_chunk_risk: CandidateRisk
    aligned_risk: CandidateRisk

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value


class AdaptiveSelector:
    """Select a merge rule from observed result age and command risk only."""

    def __init__(
        self,
        config: AdaptiveSelectorConfig,
        *,
        chunk_size: int,
        control_mode: str,
    ) -> None:
        if chunk_size <= config.hard_max_age_steps:
            raise ValueError(
                "Adaptive chunk must be longer than hard_max_age_steps so the "
                "safe-hold boundary is representable"
            )
        if control_mode != "absolute":
            raise ValueError(
                "ActionStream-Adaptive v1 risk calibration is frozen for absolute actions"
            )
        self.config = config
        self.chunk_size = int(chunk_size)
        self.control_mode = control_mode
        self._age_ewma_steps: float | None = None

    def reset(self) -> None:
        self._age_ewma_steps = None

    def _risk(
        self,
        candidate: np.ndarray,
        last_action: np.ndarray | None,
    ) -> CandidateRisk:
        command = np.asarray(candidate, dtype=np.float64)
        reasons: list[str] = []
        if command.shape != (7,) or not np.isfinite(command).all():
            return CandidateRisk(
                safe=False,
                reasons=("non_finite_or_wrong_shape",),
                position_disagreement_m=None,
                rotation_geodesic_radians=None,
                gripper_switch=None,
                workspace_inside=None,
            )

        workspace_inside = self.config.workspace.contains(command[:3])
        if not workspace_inside:
            reasons.append("outside_workspace")

        position_disagreement: float | None = None
        rotation_disagreement: float | None = None
        gripper_switch: bool | None = None
        if last_action is not None:
            previous = np.asarray(last_action, dtype=np.float64)
            if previous.shape != (7,) or not np.isfinite(previous).all():
                reasons.append("invalid_last_action")
            else:
                position_disagreement = float(
                    np.linalg.norm(command[:3] - previous[:3])
                )
                rotation_disagreement = rotation_geodesic_radians(
                    previous[3:6], command[3:6]
                )
                gripper_switch = bool(
                    (previous[6] >= 0.0) != (command[6] >= 0.0)
                )
                if (
                    position_disagreement
                    > self.config.maximum_position_disagreement_m
                ):
                    reasons.append("position_disagreement")
                if (
                    rotation_disagreement
                    > self.config.maximum_rotation_geodesic_radians
                ):
                    reasons.append("rotation_disagreement")

        return CandidateRisk(
            safe=not reasons,
            reasons=tuple(reasons),
            position_disagreement_m=position_disagreement,
            rotation_geodesic_radians=rotation_disagreement,
            gripper_switch=gripper_switch,
            workspace_inside=workspace_inside,
        )

    def decide(
        self,
        actions: np.ndarray,
        *,
        observation_control_step: int,
        current_control_step: int,
        queue_depth_steps: int,
        last_action: np.ndarray | None,
    ) -> AdaptiveDecision:
        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) != self.chunk_size:
            raise ValueError(
                f"Adaptive selector expected [{self.chunk_size},7], got {chunk.shape}"
            )
        if queue_depth_steps < 0:
            raise ValueError("Adaptive queue depth must be non-negative")
        age_steps = int(current_control_step) - int(observation_control_step)
        if age_steps < 0:
            raise ValueError("Adaptive result cannot be newer than the control step")
        if self._age_ewma_steps is None:
            self._age_ewma_steps = float(age_steps)
        else:
            alpha = self.config.age_ewma_alpha
            self._age_ewma_steps = (
                alpha * age_steps + (1.0 - alpha) * self._age_ewma_steps
            )

        aligned_index = min(age_steps, len(chunk) - 1)
        full_risk = self._risk(chunk[0], last_action)
        aligned_risk = self._risk(chunk[aligned_index], last_action)

        if age_steps <= self.config.fresh_full_max_age_steps:
            regime: AdaptiveRegime = "fresh"
            preferred_mode: AdaptiveExecutionMode = "full_chunk"
            preferred_index = 0
        elif (
            age_steps <= self.config.aligned_max_age_steps
            and len(chunk) - age_steps
            >= self.config.minimum_aligned_horizon_steps
        ):
            regime = "moderate"
            preferred_mode = "age_aligned"
            preferred_index = aligned_index
        elif age_steps <= self.config.hard_max_age_steps:
            regime = "severe_but_bounded"
            preferred_mode = "full_chunk"
            preferred_index = 0
        else:
            regime = "outage"
            return AdaptiveDecision(
                regime=regime,
                execution_mode="safe_hold",
                merge_mode=None,
                reason="result_age_exceeds_hard_limit",
                result_age_steps=age_steps,
                age_ewma_steps=self._age_ewma_steps,
                queue_depth_steps=queue_depth_steps,
                incoming_chunk_steps=len(chunk),
                selected_action_index=None,
                preferred_action_index=None,
                safe_alternate_override=False,
                selected_risk=None,
                full_chunk_risk=full_risk,
                aligned_risk=aligned_risk,
            )

        preferred_risk = full_risk if preferred_mode == "full_chunk" else aligned_risk
        if preferred_risk.safe:
            return AdaptiveDecision(
                regime=regime,
                execution_mode=preferred_mode,
                merge_mode=(
                    "async_naive" if preferred_mode == "full_chunk" else "async_aligned"
                ),
                reason=f"{regime}_preferred_candidate_safe",
                result_age_steps=age_steps,
                age_ewma_steps=self._age_ewma_steps,
                queue_depth_steps=queue_depth_steps,
                incoming_chunk_steps=len(chunk),
                selected_action_index=preferred_index,
                preferred_action_index=preferred_index,
                safe_alternate_override=False,
                selected_risk=preferred_risk,
                full_chunk_risk=full_risk,
                aligned_risk=aligned_risk,
            )

        alternate_mode: AdaptiveExecutionMode = (
            "age_aligned" if preferred_mode == "full_chunk" else "full_chunk"
        )
        alternate_index = aligned_index if alternate_mode == "age_aligned" else 0
        alternate_risk = aligned_risk if alternate_mode == "age_aligned" else full_risk
        if alternate_risk.safe:
            return AdaptiveDecision(
                regime=regime,
                execution_mode=alternate_mode,
                merge_mode=(
                    "async_naive" if alternate_mode == "full_chunk" else "async_aligned"
                ),
                reason=f"{regime}_safe_alternate_override",
                result_age_steps=age_steps,
                age_ewma_steps=self._age_ewma_steps,
                queue_depth_steps=queue_depth_steps,
                incoming_chunk_steps=len(chunk),
                selected_action_index=alternate_index,
                preferred_action_index=preferred_index,
                safe_alternate_override=True,
                selected_risk=alternate_risk,
                full_chunk_risk=full_risk,
                aligned_risk=aligned_risk,
            )

        return AdaptiveDecision(
            regime=regime,
            execution_mode="safe_hold",
            merge_mode=None,
            reason=f"{regime}_both_candidates_failed_risk_gate",
            result_age_steps=age_steps,
            age_ewma_steps=self._age_ewma_steps,
            queue_depth_steps=queue_depth_steps,
            incoming_chunk_steps=len(chunk),
            selected_action_index=None,
            preferred_action_index=preferred_index,
            safe_alternate_override=False,
            selected_risk=None,
            full_chunk_risk=full_risk,
            aligned_risk=aligned_risk,
        )
