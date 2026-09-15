"""RGB-decision confirmation and independent episode-level completion metrics."""

from __future__ import annotations


class ContinuousConfirmation:
    """Require every control in a time interval to provide positive evidence.

    Unknown, negative evidence or a missing control clears the candidate. A
    confirmation is revocable; this class does not execute a robot stop or use
    simulator fields.
    """

    def __init__(self, window_controls: int = 10):
        if window_controls < 1:
            raise ValueError("Confirmation interval must be positive")
        self.window_controls = window_controls
        self.last_control = None
        self.started = None

    def update(self, control: int, decision: str) -> bool:
        if decision not in {"complete", "incomplete", "unknown"}:
            raise ValueError("Unknown decision")
        if control < 0 or (
            self.last_control is not None and control <= self.last_control
        ):
            raise ValueError("Controls must advance strictly")
        if self.last_control is not None and control != self.last_control + 1:
            self.started = None
        self.last_control = control
        if decision != "complete":
            self.started = None
            return False
        if self.started is None:
            self.started = control
        return control - self.started >= self.window_controls


def episode_metrics(truth, decisions, *, control_hz=20, post_stop=None):
    """Evaluate first-stop safety separately from eventual event detection.

    `truth` is private evaluator data, indexed by physical control. `decisions`
    contains (control, claims_complete) pairs. A later correct claim cannot
    erase an earlier unsafe first stop. Post-stop truth must be supplied by a
    separately executed simulator branch; absent data stays unavailable.
    """
    if control_hz <= 0 or not truth:
        raise ValueError("Need nonempty truth and a positive control frequency")
    controls = [step for step, _ in decisions]
    if any(b <= a for a, b in zip(controls, controls[1:])):
        raise ValueError("Decision controls must be unique and ordered")
    if any(step < 0 or step >= len(truth) for step in controls):
        raise ValueError("Decision outside observed truth")
    first_truth = next((i for i, ready in enumerate(truth) if ready), None)
    claims = [step for step, complete in decisions if complete]
    first_claim = claims[0] if claims else None
    first_correct = next((step for step in claims if truth[step]), None)
    premature = first_claim is not None and not truth[first_claim]
    return dict(
        first_truth_control=first_truth,
        first_claim_control=first_claim,
        first_correct_control=first_correct,
        premature_stop=premature,
        missed_completed_event=first_truth is not None and first_correct is None,
        no_stop=first_claim is None,
        confirmation_delay_s=(first_claim - first_truth) / control_hz
        if first_truth is not None and first_claim is not None and not premature
        else None,
        post_stop_stable=all(post_stop) if post_stop else None,
    )
