from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from actionstream.adaptive_runtime import (
    AdaptiveSelector,
    load_selector_config,
    rotation_geodesic_radians,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "configs" / "actionstream_adaptive_v1_selector.json"
SELECTOR_SHA256 = "e5cd59bbe760a5d7c9b6e4b1addbb039af31f4962d622af8973944e2125006d5"


def _chunk() -> np.ndarray:
    actions = np.zeros((30, 7), dtype=np.float32)
    actions[:, 0] = np.linspace(-0.10, -0.071, 30)
    actions[:, 2] = 0.10
    actions[:, 6] = 1.0
    return actions


def _selector() -> AdaptiveSelector:
    config = load_selector_config(SELECTOR_PATH, expected_sha256=SELECTOR_SHA256)
    return AdaptiveSelector(config, chunk_size=30, control_mode="absolute")


@pytest.mark.parametrize(
    ("age", "regime", "mode", "index"),
    [
        (3, "fresh", "full_chunk", 0),
        (14, "moderate", "age_aligned", 14),
        (23, "severe_but_bounded", "full_chunk", 0),
        (29, "outage", "safe_hold", None),
    ],
)
def test_frozen_age_regimes_select_expected_execution(
    age: int,
    regime: str,
    mode: str,
    index: int | None,
) -> None:
    decision = _selector().decide(
        _chunk(),
        observation_control_step=10,
        current_control_step=10 + age,
        queue_depth_steps=12,
        last_action=np.asarray([-0.10, 0.0, 0.10, 0.0, 0.0, 0.0, 1.0]),
    )

    assert decision.regime == regime
    assert decision.execution_mode == mode
    assert decision.selected_action_index == index


def test_risk_gate_uses_safe_alternate_before_hold() -> None:
    actions = _chunk()
    actions[0, 0] = 0.80
    decision = _selector().decide(
        actions,
        observation_control_step=0,
        current_control_step=3,
        queue_depth_steps=5,
        last_action=np.asarray([-0.10, 0.0, 0.10, 0.0, 0.0, 0.0, 1.0]),
    )

    assert decision.regime == "fresh"
    assert decision.execution_mode == "age_aligned"
    assert decision.safe_alternate_override
    assert decision.preferred_action_index == 0
    assert decision.selected_action_index == 3
    assert decision.full_chunk_risk.reasons == ("outside_workspace", "position_disagreement")
    assert decision.aligned_risk.safe


def test_risk_gate_holds_when_both_candidates_are_unsafe() -> None:
    actions = _chunk()
    actions[:, 0] = 0.80
    decision = _selector().decide(
        actions,
        observation_control_step=0,
        current_control_step=14,
        queue_depth_steps=5,
        last_action=np.asarray([-0.10, 0.0, 0.10, 0.0, 0.0, 0.0, 1.0]),
    )

    assert decision.execution_mode == "safe_hold"
    assert decision.reason == "moderate_both_candidates_failed_risk_gate"
    assert not decision.full_chunk_risk.safe
    assert not decision.aligned_risk.safe


def test_selector_rejects_relative_action_contract() -> None:
    config = load_selector_config(SELECTOR_PATH, expected_sha256=SELECTOR_SHA256)
    with pytest.raises(ValueError, match="absolute actions"):
        AdaptiveSelector(config, chunk_size=30, control_mode="relative")


def test_selector_protocol_hash_is_mandatory() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_selector_config(SELECTOR_PATH, expected_sha256="0" * 64)


def test_rotation_disagreement_uses_so3_geodesic() -> None:
    assert rotation_geodesic_radians(
        np.zeros(3), np.asarray([0.0, 0.0, np.pi / 2.0])
    ) == pytest.approx(np.pi / 2.0)
