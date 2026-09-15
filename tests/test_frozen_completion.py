"""Regression checks for frozen decision semantics and camera orientation."""

import numpy as np
import pytest

from actionstream.llm_vla.completion import decision, native_rgb
from scripts.engineering.evaluate_visual_completion import metrics


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "incomplete"),
        (0.1, "incomplete"),
        (0.1001, "unknown"),
        (0.8999, "unknown"),
        (0.9, "complete"),
        (1, "complete"),
    ],
)
def test_frozen_boundaries(value, expected):
    assert decision(value) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probabilities(value):
    with pytest.raises(ValueError):
        decision(value)


def test_both_native_cameras_flip_vertically_without_swapping_horizontal_sides():
    pixels = {
        key: np.zeros((1, 360, 360, 3), dtype=np.uint8) for key in ("image", "image2")
    }
    pixels["image"][0, :180, :180] = [255, 0, 0]
    pixels["image2"][0, :180, :180] = [0, 255, 0]
    result = native_rgb(pixels)
    assert result[0, 100, 20].tolist() == [255, 0, 0]
    assert result[1, 100, 20].tolist() == [0, 255, 0]
    assert not result[:, 20, 20].any()
    assert not result[:, 100, 100].any()
    assert pixels["image"][0, 20, 20, 0] == 255


def test_abstentions_are_missed_completions_and_not_false_incomplete():
    result = metrics(
        [
            dict(truth=True, decision="unknown"),
            dict(truth=True, decision="incomplete"),
            dict(truth=False, decision="complete"),
        ]
    )
    assert result["missed_complete"] == 2
    assert result["false_incomplete"] == 1
    assert result["unknown"] == 1
    assert result["false_complete"] == 1
