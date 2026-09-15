"""Completion pilot data leakage and abstention accounting contracts."""

import numpy as np
import pytest

from scripts.engineering.train_visual_completion import (
    completion_metrics,
    episode_splits,
)


def test_split_is_episode_disjoint_and_order_independent():
    names = [f"demo_{i}" for i in range(50)]
    splits = episode_splits(names, 20260915)
    assert splits == episode_splits(list(reversed(names)), 20260915)
    assert list(map(len, splits.values())) == [30, 10, 10]
    assert set.union(*map(set, splits.values())) == set(names)
    assert sum(map(len, splits.values())) == len(names)


def test_unknown_does_not_count_as_success():
    metrics = completion_metrics([0, 0, 1, 1], [0.95, 0.5, 0.5, 0.05])
    assert metrics["false_complete_at_0_9"] == 1
    assert metrics["true_complete_at_0_9"] == 0
    assert metrics["false_incomplete_at_0_1"] == 1
    assert metrics["unknown_at_0_1_0_9"] == 2


def test_nonfinite_prediction_is_invalid():
    with pytest.raises(ValueError, match="Nonfinite"):
        completion_metrics([1], [np.nan])


def test_duplicate_episode_is_rejected():
    with pytest.raises(ValueError, match="distinct"):
        episode_splits(["same"] * 50, 20260915)
