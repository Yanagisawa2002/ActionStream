import pytest

from actionstream.llm_vla.confirmation import ContinuousConfirmation, episode_metrics


def test_confirmation_requires_interval_not_just_positive_count():
    gate = ContinuousConfirmation()
    assert not any(gate.update(i, "complete") for i in range(10))
    assert gate.update(10, "complete")


@pytest.mark.parametrize("interruption", ["unknown", "incomplete"])
def test_evidence_interruption_revokes_and_restarts(interruption):
    gate = ContinuousConfirmation()
    for i in range(11):
        gate.update(i, "complete")
    assert not gate.update(11, interruption)
    assert not any(gate.update(i, "complete") for i in range(12, 22))
    assert gate.update(22, "complete")


def test_missing_control_cannot_bridge_unobserved_time():
    gate = ContinuousConfirmation()
    for i in range(10):
        gate.update(i, "complete")
    assert not gate.update(11, "complete")
    assert not gate.update(12, "complete")
    with pytest.raises(ValueError):
        gate.update(12, "complete")


def test_later_success_does_not_erase_unsafe_first_stop():
    m = episode_metrics([False, False, True, True], [(1, True), (2, True)])
    assert m["premature_stop"] and not m["missed_completed_event"]
    assert m["confirmation_delay_s"] is None
    assert m["post_stop_stable"] is None


def test_delay_and_executed_post_stop_failure():
    m = episode_metrics([False, True, True], [(2, True)], post_stop=[True, False])
    assert m["confirmation_delay_s"] == 0.05
    assert m["post_stop_stable"] is False


def test_no_positive_truth_is_not_missed_completion():
    m = episode_metrics([False] * 5, [(2, False)], post_stop=[])
    assert not m["missed_completed_event"] and m["no_stop"]
    assert m["post_stop_stable"] is None


def test_entire_completed_event_can_be_missed():
    assert episode_metrics([False, True], [(1, False)])["missed_completed_event"]


def test_metric_rejects_unordered_or_unobserved_decisions():
    for decisions in ([(2, True)], [(1, False), (0, True)]):
        with pytest.raises(ValueError):
            episode_metrics([False, True], decisions)
