"""Post-run task, RGB, physics and timing audit. No online feedback channel."""

import json
from pathlib import Path

from .async_scoring import score_episode as score_async
from .grounding import OriginalRequest
from .multitask_language import authorize_task
from .task_registry import development_task
from .task_completion import TaskCondition
from .task_truth import PROFILES, instantaneous


def score_episode(directory):
    directory = Path(directory)

    def read(name):
        return json.loads((directory / name).read_text())

    identity, outcome = read("task.json"), read("outcome.json")
    task = development_task(identity["task"]["key"])
    from dataclasses import asdict

    errors = []
    condition = TaskCondition(**read("condition.json"))
    try:
        condition.validate(task)
        if condition.sha256 != outcome["condition_sha256"]:
            errors.append("condition_identity")
    except ValueError:
        errors.append("condition_identity")
    if (
        identity["task"] != asdict(task)
        or identity["sha256"] != task.sha256
        or identity["bddl_sha256"] != PROFILES[task.key]["bddl_sha256"]
    ):
        errors.append("task_definition")
    parsed = read("parser_call.json")
    try:
        if parsed["call"]["status"] != "COMPLETED":
            raise ValueError("No completed model parser call")
        permit = authorize_task(
            OriginalRequest(**parsed["original"]), parsed["call"]["raw_output"]
        )
        if (
            permit.task_key != task.key
            or permit.original.request_id != outcome["request_id"]
        ):
            errors.append("language_task")
    except (ValueError, KeyError, TypeError):
        errors.append("language_authorization")
    for fact in read("private_truth.json"):
        if fact["task_key"] != task.key or fact["task_sha256"] != task.sha256:
            errors.append("physical_task_binding")
            break
    events = [
        json.loads(line)
        for line in (directory / "runtime.jsonl").read_text().splitlines()
    ]
    authorized = [row for row in events if row["event"] == "authorized"]
    if not errors and (
        len(authorized) != 1
        or authorized[0]["original"] != parsed["original"]
        or authorized[0]["permit"] != asdict(permit)
        or authorized[0]["task"] != asdict(task)
    ):
        errors.append("authorization_journal")
    if any(
        row.get("task_key") != task.key
        or row.get("task_sha256") != task.sha256
        or row.get("condition_sha256") != outcome["condition_sha256"]
        for row in [outcome, *events]
    ):
        errors.append("online_task_binding")
    score = score_async(
        directory,
        forced_open_until=read("intervention.json")["forced_open_until"],
        fact_predicate=instantaneous,
    )
    score["integrity_errors"].extend(errors)
    score["integrity_passed"] = not score["integrity_errors"]
    score["safely_completed"] &= score["integrity_passed"]
    score["recovery"]["recovered_from_failure"] &= score["integrity_passed"]
    return dict(score, task_key=task.key, task_sha256=task.sha256)
