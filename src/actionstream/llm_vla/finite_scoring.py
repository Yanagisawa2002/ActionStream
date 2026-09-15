"""Post-execution private scoring; this module is never a controller input."""

from __future__ import annotations

import json
from pathlib import Path

from actionstream.completion_labels import StableTruth
from .confirmation import ContinuousConfirmation, episode_metrics
from .temporal_completion import classify


def score_episode(directory):
    directory = Path(directory)
    outcome = json.loads((directory / "outcome.json").read_text())
    facts = json.loads((directory / "private_truth.json").read_text())
    bindings = json.loads((directory / "observation_bindings.json").read_text())
    actions = json.loads((directory / "actions.json").read_text())
    runtime = [
        json.loads(line)
        for line in (directory / "runtime.jsonl").read_text().splitlines()
    ]
    checks = [r for r in runtime if r["event"] == "verification"]
    truth, confirmation = StableTruth(), ContinuousConfirmation(10)
    errors, strict, decisions = [], [], []
    for i, row in enumerate(facts):
        value = truth.update(i, row)
        strict.append(value)
        if value != row["strict_complete"]:
            errors.append(f"truth:{i}")
    for row in checks:
        c = row["control"]
        decision = classify(row["probabilities"]) if c >= 10 else "unknown"
        yes = confirmation.update(c, decision)
        decisions.append((c, yes))
        if (
            row["decision"] != decision
            or row["confirmed"] != yes
            or c >= len(bindings)
            or row["observation"] != bindings[c]
            or row["observation"]["request_id"] != outcome["request_id"]
            or row["observation"]["revision"] != 0
            or row["history_ready"] != (c >= 10)
        ):
            errors.append(f"verification:{c}")
    expected = outcome["control_steps"] + outcome["post_stop_controls"]
    if (
        len(facts) != expected + 1
        or len(actions) != expected
        or len(bindings) != len(facts)
    ):
        errors.append("physical_record_coverage")
    if [c for c, _ in decisions] != list(range(outcome["control_steps"] + 1)):
        errors.append("control_coverage")
    first = next((c for c, yes in decisions if yes), None)
    if first != outcome["first_claim_control"]:
        errors.append("first_claim")
    if (outcome["status"] == "complete") != (first is not None):
        errors.append("outcome")
    post = strict[first + 1 :] if first is not None else None
    if first is not None and (len(post) != 40 or outcome["post_stop_controls"] != 40):
        errors.append("post_stop_coverage")
    if outcome["status"] == "ERROR":
        errors.append("execution_error")
    chunks = {
        r["source_control"]: r["actions"] for r in runtime if r["event"] == "chunk"
    }
    expected_chunk_controls = list(range(0, min(outcome["control_steps"], 300), 30))
    if sorted(chunks) != expected_chunk_controls:
        errors.append("chunk_coverage")
    for index, action in enumerate(actions):
        if index >= outcome["control_steps"] or index >= 300:
            expected_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        else:
            chunk = chunks.get(index // 30 * 30, [])
            expected_action = chunk[index % 30] if len(chunk) == 30 else None
        if action != expected_action:
            errors.append(f"physical_action:{index}")
    metrics = episode_metrics(
        strict[: outcome["control_steps"] + 1], decisions, post_stop=post
    )
    return dict(**metrics, integrity_passed=not errors, integrity_errors=errors)


def score_run(protocol, language, episodes):
    """Count all supported requests, including rejects, in the task denominator."""
    labels = protocol["labels"]
    expected = {r["id"] for r in protocol["requests"]}
    actual = [r["id"] for r in language]
    accepted = {r["id"] for r in language if r["decision"] == "accept"}
    positive = {key for key, value in labels.items() if value == "supported"}
    executable = {r["id"] for r in protocol["requests"] if r.get("seed") is not None}
    seeds = {r["id"]: r["seed"] for r in protocol["requests"] if "seed" in r}
    completed = [e for e in episodes if e["score"]["first_truth_control"] is not None]
    safe = [
        e
        for e in episodes
        if e["outcome"]["status"] == "complete"
        and not e["score"]["premature_stop"]
        and e["score"]["post_stop_stable"] is True
        and e["score"]["integrity_passed"]
    ]
    delays = [
        e["score"]["confirmation_delay_s"]
        for e in episodes
        if e["score"]["confirmation_delay_s"] is not None
    ]
    language_metrics = dict(
        cases=len(language),
        supported=len(positive),
        supported_accepted=len(accepted & positive),
        unsupported=len(labels) - len(positive),
        unsupported_accepted=len(accepted - positive),
        model_or_schema_errors=sum(
            r["model_status"] != "COMPLETED" or r["schema_status"] != "VALID"
            for r in language
        ),
        exact_request_coverage=set(actual) == expected and len(actual) == len(expected),
    )
    runtime = dict(
        requested_tasks=len(executable),
        executed_tasks=len(episodes),
        language_blocked_tasks=len(executable - accepted),
        physically_completed=len(completed),
        safely_completed_tasks=len(safe),
        premature_stops=sum(e["score"]["premature_stop"] for e in episodes),
        missed_completed_events=sum(
            e["score"]["missed_completed_event"] for e in episodes
        ),
        post_stop_failures=sum(
            e["score"]["post_stop_stable"] is False for e in episodes
        ),
        integrity_failures=sum(not e["score"]["integrity_passed"] for e in episodes),
        maximum_confirmation_delay_s=max(delays) if delays else None,
        exact_execution_coverage={e["id"] for e in episodes} == (executable & accepted)
        and len(episodes) == len(executable & accepted)
        and all(e.get("seed") == seeds.get(e["id"]) for e in episodes),
    )
    gates = protocol["gates"]
    passed = bool(
        language_metrics["exact_request_coverage"]
        and runtime["exact_execution_coverage"]
        and language_metrics["supported_accepted"] >= gates["supported_accepted_min"]
        and language_metrics["unsupported_accepted"] == 0
        and language_metrics["model_or_schema_errors"] == 0
        and runtime["safely_completed_tasks"] >= gates["safely_completed_tasks_min"]
        and runtime["premature_stops"]
        == runtime["missed_completed_events"]
        == runtime["post_stop_failures"]
        == runtime["integrity_failures"]
        == 0
        and delays
        and max(delays) <= gates["max_confirmation_delay_s"]
    )
    return dict(
        status="GO" if passed else "NO-GO", language=language_metrics, runtime=runtime
    )
