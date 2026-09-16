"""Post-execution audit of async timing, physical truth and bounded recovery.

Reads retained files only. Does not call the controller, learned verifier,
simulator or asynchronous engine. No value from this module is a control input.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def score_episode(directory, *, forced_open_until=0, fact_predicate=None):
    directory = Path(directory)

    def read(name):
        return json.loads((directory / name).read_text())

    outcome = read("outcome.json")
    config = outcome["config"]
    facts, bindings, actions = (
        read(name)
        for name in ("private_truth.json", "observation_bindings.json", "actions.json")
    )
    events = [
        json.loads(line)
        for line in (directory / "runtime.jsonl").read_text().splitlines()
    ]
    errors, truth = [], []
    window = deque(maxlen=11)
    for control, fact in enumerate(facts):
        speeds = (fact["linear_speed"], fact["angular_speed"])
        instantaneous = (
            fact_predicate(fact)
            if fact_predicate is not None
            else (
                fact["inside"]
                and not fact["finger_contact"]
                and fact["basket_contact"]
                and all(math.isfinite(v) and v >= 0 for v in speeds)
                and speeds[0] <= 0.03
                and speeds[1] <= 0.3
            )
        )
        window.append(instantaneous)
        strict = len(window) == 11 and all(window)
        truth.append(strict)
        if strict != fact["strict_complete"]:
            errors.append(f"physical_truth:{control}")
    dispatches = [e for e in events if e["event"] == "dispatch"]
    checks = [e for e in events if e["event"] in ("verification", "stale_observation")]
    expected = outcome["control_steps"] + outcome["post_stop_controls"]
    if not (
        len(actions) == len(dispatches) == expected
        and len(facts) == len(bindings) == expected + 1
    ):
        errors.append("physical_record_coverage")
    if [e["control"] for e in dispatches] != list(range(expected)):
        errors.append("dispatch_coverage")
    if [e["control"] for e in checks] != list(range(outcome["control_steps"] + 1)):
        errors.append("verification_coverage")
    first_claim, last_control, revision, history, streak = None, None, None, 0, 0
    for row in checks:
        control = row["control"]
        binding = row["observation"]
        if (
            control >= len(bindings)
            or binding != bindings[control]
            or binding["request_id"] != outcome["request_id"]
        ):
            errors.append(f"verification_binding:{control}")
        if revision != binding["revision"] or (
            last_control is not None and control != last_control + 1
        ):
            history = streak = 0
        revision, last_control = binding["revision"], control
        if row["event"] == "stale_observation":
            history = streak = 0
            continue
        history += 1
        ready = history >= 11
        values = row["probabilities"]
        decision = "unknown"
        if ready:
            if (
                values is None
                or len(values) != 3
                or not all(math.isfinite(v) and 0 <= v <= 1 for v in values)
                or abs(sum(values) - 1) > 1e-5
            ):
                errors.append(f"probabilities:{control}")
                continue
            decision = (
                "complete"
                if values[1] >= 0.95
                else "incomplete"
                if values[0] >= 0.9
                else "unknown"
            )
        elif values is not None:
            errors.append(f"unready_prediction:{control}")
        streak = streak + 1 if decision == "complete" else 0
        confirmed = streak >= 11
        if confirmed and first_claim is None:
            first_claim = control
        if (
            ready != row["history_ready"]
            or decision != row["decision"]
            or confirmed != row["confirmed"]
        ):
            errors.append(f"confirmation:{control}")
    if first_claim != outcome["first_claim_control"]:
        errors.append("first_claim")
    if (outcome["status"] == "complete") != (first_claim is not None):
        errors.append("outcome")
    chunks = {
        e["observation"]["observation_id"]: e
        for e in events
        if e["event"] == "async_chunk"
    }
    changed_commands = 0
    ages = []
    warmups = [e for e in events if e["event"] == "worker_warmup_complete"]
    epoch_offset = outcome.get("engine_epoch_offset", 0)
    if (
        epoch_offset != 1
        or len(warmups) != 1
        or not warmups[0]["discarded_all_warmup_actions"]
    ):
        errors.append("worker_warmup_boundary")
    for row, actual in zip(dispatches, actions):
        control = row["control"]
        requested = list(row["action"])
        expected_action = list(requested)
        if control < forced_open_until:
            changed_commands += int(expected_action[-1] != -1)
            expected_action[-1] = -1.0
        if expected_action != actual:
            errors.append(f"physical_action:{control}")
        source = row["source"]
        if source is None:
            continue
        binding = source["observation"]
        age = row["dispatched_monotonic"] - binding["captured_monotonic"]
        ages.append(age)
        chunk = chunks.get(binding["observation_id"])
        if (
            row["phase"] != "policy"
            or first_claim is not None
            and control >= first_claim
            or binding["request_id"] != outcome["request_id"]
            or binding["revision"] != row["revision"]
            or source["epoch"] != epoch_offset + row["revision"]
            or not 0 <= age <= config["action_source_max_age_s"]
            or chunk is None
            or chunk["observation"] != binding
            or chunk["source_control"] != source["control"]
            or not any(requested == value for value in chunk["actions"])
        ):
            errors.append(f"action_provenance:{control}")
    retries = [e for e in events if e["event"] == "recovery_started"]
    if (
        outcome["attempts"] != 1 + len(retries)
        or outcome["attempts"] > config["max_attempts"]
    ):
        errors.append("recovery_budget")
    cap = config["max_attempts"] * (
        config["policy_controls"] + config["settle_controls"]
    )
    cap += (config["max_attempts"] - 1) * config["recovery_open_controls"]
    if outcome["control_steps"] > cap:
        errors.append("total_control_budget")
    first_truth = next(
        (c for c, value in enumerate(truth[: outcome["control_steps"] + 1]) if value),
        None,
    )
    premature = first_claim is not None and (
        first_claim >= len(truth) or not truth[first_claim]
    )
    post = truth[first_claim + 1 :] if first_claim is not None else []
    stable = (
        len(post) == config["post_stop_controls"] and all(post)
        if first_claim is not None
        else None
    )
    if (
        first_claim is not None
        and outcome["post_stop_controls"] != config["post_stop_controls"]
    ):
        errors.append("post_stop_coverage")
    if outcome["status"] == "ERROR":
        errors.append("execution_error")
    if not dispatches:
        errors.append("empty_execution")
    work = [row["work_wall_s"] for row in dispatches]
    periods = [
        b["started_monotonic"] - a["started_monotonic"]
        for a, b in zip(dispatches, dispatches[1:])
    ]
    if any(not math.isfinite(value) or value < 0 for value in work + periods):
        errors.append("invalid_clock")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as trajectory:
        for field in ("states", "rgb"):
            if len(trajectory[field]) != expected + 1:
                errors.append("trajectory_coverage:" + field)
        rgb = trajectory["rgb"]
        if rgb.shape != (expected + 1, 2, 192, 192, 3) or rgb.dtype != np.uint8:
            errors.append("trajectory_rgb_contract")
        else:
            for row in checks:
                if row["event"] == "verification" and row["history_ready"]:
                    control = row["control"]
                    clip = np.stack([rgb[control - offset] for offset in (10, 5, 0)])
                    if hashlib.sha256(clip.tobytes()).hexdigest() != row["clip_sha256"]:
                        errors.append(f"clip_identity:{control}")
    safe = (
        outcome["status"] == "complete"
        and not errors
        and not premature
        and stable is True
    )
    retry_control = retries[0]["control"] if retries else None
    stops = [e for e in events if e["event"] == "confirmed_stop"]
    wall_delay = (
        stops[0]["confirmed_monotonic"] - bindings[first_truth]["captured_monotonic"]
        if first_claim is not None and first_truth is not None and not premature
        else None
    )
    attempt_end = config["policy_controls"] + config["settle_controls"]
    eligible = (
        forced_open_until > 0
        and changed_commands > 0
        and len(truth) > attempt_end
        and not any(truth[: attempt_end + 1])
    )
    return dict(
        integrity_passed=not errors,
        integrity_errors=errors,
        first_truth_control=first_truth,
        first_claim_control=first_claim,
        safely_completed=safe,
        premature_stop=premature,
        missed_completed_event=first_truth is not None and first_claim is None,
        post_stop_stable=stable,
        confirmation_delay_s=(first_claim - first_truth) / 20
        if first_claim is not None and first_truth is not None and not premature
        else None,
        confirmation_wall_delay_s=wall_delay,
        timing=dict(
            controls=len(work),
            deadline_misses=sum(v > config["period_s"] for v in work),
            work_p95_s=float(np.quantile(work, 0.95)) if work else None,
            period_p95_s=float(np.quantile(periods, 0.95)) if periods else None,
            period_max_s=max(periods) if periods else None,
            action_source_max_age_s=max(ages) if ages else None,
            stale_observations=sum(e["event"] == "stale_observation" for e in events),
        ),
        recovery=dict(
            forced_open_until=forced_open_until,
            modified_gripper_commands=changed_commands,
            retries=len(retries),
            physical_failure_eligible=eligible,
            recovered_from_failure=bool(
                eligible
                and safe
                and retry_control is not None
                and first_claim > retry_control
            ),
        ),
    )
