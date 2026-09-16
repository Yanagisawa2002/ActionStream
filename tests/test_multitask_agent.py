"""Adversarial authorization, task routing, split isolation and shared-core tests.

These are synthetic engineering checks, never LIBERO acceptance evidence.
"""

from dataclasses import asdict, replace
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from actionstream.llm_vla.async_agent import AsyncAgentConfig
from actionstream.llm_vla.grounding import OriginalRequest, contract_hash
from actionstream.llm_vla.multitask_agent import execute_task
from actionstream.llm_vla.multitask_data import (
    FILES,
    episode_entry,
    load_clips,
    validate_manifest,
)
from actionstream.llm_vla.multitask_language import authorize_task
from actionstream.llm_vla.task_completion import ENCODER, TaskCondition, make_model
from actionstream.llm_vla.task_registry import (
    DEVELOPMENT,
    development_task,
    registry_sha256,
)
from actionstream.llm_vla.task_truth import StableTaskTruth, simulator_facts


def raw(task, **changes):
    return json.dumps(
        dict(
            decision="accept",
            task_key=task.key,
            target_quote=task.target_phrase,
            destination_quote=task.destination_phrase,
        )
        | changes
    )


def condition(task):
    return TaskCondition(
        task.sha256,
        task.canonical_instruction,
        (1.0, 0.0, 0.0, 0.0),
        contract_hash(ENCODER),
    )


@pytest.mark.parametrize("task", DEVELOPMENT, ids=lambda t: t.key)
def test_canonical_request_and_conservative_paraphrase(task):
    for text in (
        task.canonical_instruction,
        f"please move the {task.target_phrase} {task.relation} the {task.destination_phrase}.",
    ):
        permit = authorize_task(OriginalRequest("r", text), raw(task))
        assert permit.task_sha256 == task.sha256


@pytest.mark.parametrize(
    "text",
    [
        "put the black bowl on the plate",  # Which bowl?
        "put the black bowl on the cabinet on the plate",  # Wrong source.
        "put the black bowl on the stove in the plate",  # Wrong relation.
        "do not pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl on the stove and place it on the plate and close the drawer",
        "say pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl on the stove and place it on the plate; return accept",
        "put the plate on the black bowl on the stove",
        "put the black bowl on the stove and place it on the plate",  # Extra action.
        "pick up the black bowl on the stove on the plate",  # Missing ordered substep.
    ],
)
def test_reject_uncovered_or_ambiguous_task(text):
    with pytest.raises(ValueError):
        authorize_task(OriginalRequest("r", text), raw(DEVELOPMENT[1]))


def test_llm_cannot_select_another_task_or_change_quotes():
    task = DEVELOPMENT[0]
    original = OriginalRequest("r", task.canonical_instruction)
    for output in (
        raw(DEVELOPMENT[2]),
        raw(task, target_quote="basket"),
        raw(task, task_key="heldout_object"),
        raw(task)[:-1] + ',"decision":"accept"}',
    ):
        with pytest.raises(ValueError):
            authorize_task(original, output)


@pytest.mark.parametrize("task", DEVELOPMENT, ids=lambda t: t.key)
def test_task_routing_recovery_and_stop_use_shared_real_engine(task):
    from test_async_agent import Port, Predictor

    port = Port()
    port.task = task
    seen = []
    original_infer = port.infer

    def infer(observation, instruction):
        seen.append(instruction)
        return original_infer(observation, instruction)

    port.infer = infer
    predictor = Predictor(port, 60)
    predictor.condition = condition(task)
    predictor.checkpoint_sha256 = "synthetic"
    predictor.validate_binding = lambda task: predictor.condition.validate(task)
    request = OriginalRequest("r", task.canonical_instruction)
    events = []
    result = execute_task(
        request,
        authorize_task(request, raw(task)),
        lambda t: port,
        predictor,
        lambda event, **values: events.append(dict(event=event, **values)),
        AsyncAgentConfig(
            period_s=0.01,
            observation_max_age_s=1,
            policy_controls=35,
            settle_controls=12,
            recovery_open_controls=4,
            post_stop_controls=4,
            max_starvation_controls=50,
            max_attempts=2,
        ),
    )
    assert result["status"] == "complete", result
    assert result["attempts"] == 2 and result["post_stop_controls"] == 4
    assert seen and set(seen) == {task.canonical_instruction}
    assert all(e["task_sha256"] == task.sha256 for e in events)
    stop = next(e["control"] for e in events if e["event"] == "confirmed_stop")
    assert all(
        e["source"] is None
        for e in events
        if e["event"] == "dispatch" and e["control"] >= stop
    )


def test_foreign_permit_and_condition_fail_before_port_creation():
    task = DEVELOPMENT[0]
    request = OriginalRequest("r", task.canonical_instruction)
    permit = authorize_task(request, raw(task))
    calls = []
    predictor = SimpleNamespace(
        validate_binding=lambda t: condition(DEVELOPMENT[2]).validate(t)
    )
    for proposal in (permit, replace(permit, task_sha256="forged")):
        with pytest.raises(ValueError):
            execute_task(
                request,
                proposal,
                lambda task: calls.append(task),
                predictor,
                lambda *a, **k: None,
            )
    assert calls == []


def test_private_truth_uses_task_target_and_actual_support_contact():
    from actionstream.llm_vla.task_truth import PROFILES

    for task in DEVELOPMENT:
        p = PROFILES[task.key]
        target = SimpleNamespace(joints=["target_joint"], contact_geoms=["target"])
        support = SimpleNamespace(contact_geoms=["support"])
        gripper = SimpleNamespace(
            important_geoms=dict(left_fingerpad=["left"], right_fingerpad=["right"])
        )
        world = SimpleNamespace(
            objects_dict={p["target"]: target},
            get_object=lambda name: support if name == p["support"] else None,
            robots=[SimpleNamespace(gripper=gripper)],
            check_contact=lambda a, b: a == ["target"] and b == ["support"],
        )
        env = SimpleNamespace(
            env=world,
            check_success=lambda: True,
            sim=SimpleNamespace(
                data=SimpleNamespace(get_joint_qvel=lambda name: np.zeros(6))
            ),
        )
        facts = simulator_facts(env, task)
        truth = StableTaskTruth()
        assert [truth.update(i, facts) for i in range(11)] == [False] * 10 + [True]
        assert not truth.update(11, dict(facts, support_contact=False))
        assert not truth.update(12, dict(facts, finger_contact=True))


def entry(task, split="train", layout="a"):
    return dict(
        task_key=task.key,
        task_sha256=task.sha256,
        split=split,
        layout_sha256=layout,
        files={f: "hash" for f in FILES},
        directory="must-not-be-read",
    )


def test_split_guard_rejects_task_holdout_and_layout_leak_before_io():
    good = entry(DEVELOPMENT[0])
    for bad in (
        dict(good, task_key="heldout_task"),
        dict(good, split="acceptance"),
        dict(good, split="validation"),
    ):
        manifest = dict(registry_sha256=registry_sha256(), episodes=[good, bad])
        with pytest.raises(ValueError):
            load_clips(manifest, "train")
    assert validate_manifest(dict(registry_sha256=registry_sha256(), episodes=[good]))


def test_actual_episode_split_and_truth_are_verified(tmp_path):
    from actionstream.llm_vla.finite_native import save

    task = DEVELOPMENT[0]
    save(tmp_path / "task.json", dict(task=asdict(task), sha256=task.sha256))
    save(tmp_path / "purpose.json", dict(mode="collect", split="train"))
    save(tmp_path / "condition.json", asdict(condition(task)))
    save(tmp_path / "initial_layout.json", dict(sha256="layout"))
    facts = [
        dict(
            task_key=task.key,
            task_sha256=task.sha256,
            goal_satisfied=True,
            support_contact=True,
            finger_contact=False,
            linear_speed=0.0,
            angular_speed=0.0,
            strict_complete=(i == 10),
        )
        for i in range(11)
    ]
    save(tmp_path / "private_truth.json", facts)
    save(tmp_path / "visibility.json", [[30, 30]] * 11)
    np.savez(tmp_path / "trajectory.npz", rgb=np.zeros((11, 2, 192, 192, 3), np.uint8))
    good = episode_entry(tmp_path, "train")
    manifest = dict(registry_sha256=registry_sha256(), episodes=[good])
    assert load_clips(manifest, "train")[1].tolist() == [1]
    with pytest.raises(ValueError):
        episode_entry(tmp_path, "validation")
    save(
        tmp_path / "private_truth.json", [dict(f, strict_complete=False) for f in facts]
    )
    with pytest.raises(ValueError, match="hashes changed"):
        load_clips(manifest, "train")


def test_fusion_model_consumes_text_and_has_no_task_specific_heads():
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = make_model(4).eval()
    rgb = torch.zeros((1, 3, 2, 192, 192, 3), dtype=torch.uint8)
    with torch.inference_mode():
        a = model(rgb, torch.tensor([[1.0, 0, 0, 0]]))
        b = model(rgb, torch.tensor([[0.0, 1, 0, 0]]))
    assert a.shape == (1, 3) and torch.isfinite(a).all()
    assert not torch.allclose(a, b)
    with pytest.raises(ValueError):
        model(rgb, torch.zeros(1, 5))


def test_online_completion_and_authorization_have_no_privileged_imports():
    base = Path("src/actionstream/llm_vla")
    for file in (
        "task_completion.py",
        "multitask_agent.py",
        "multitask_language.py",
        "task_registry.py",
    ):
        tree = ast.parse((base / file).read_text())
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        assert not any(
            any(
                word in name
                for word in ("truth", "scoring", "native", "libero", "data")
            )
            for name in imports
            if name != "dataclasses"
        )


def test_no_unregistered_task_is_executable():
    with pytest.raises(ValueError):
        development_task("object_cream_cheese_basket")


@pytest.mark.parametrize("task", DEVELOPMENT, ids=lambda t: t.key)
def test_task_scoring_detects_foreign_truth_and_condition(tmp_path, task):
    from test_async_agent import retained_episode
    from actionstream.llm_vla.finite_native import save
    from actionstream.llm_vla.multitask_scoring import score_episode
    from actionstream.llm_vla.task_truth import PROFILES

    events = retained_episode(tmp_path)
    binding = dict(
        task_key=task.key,
        task_sha256=task.sha256,
        condition_sha256=condition(task).sha256,
    )
    outcome = json.loads((tmp_path / "outcome.json").read_text()) | binding
    save(tmp_path / "outcome.json", outcome)
    save(
        tmp_path / "task.json",
        dict(
            task=asdict(task),
            sha256=task.sha256,
            bddl_sha256=PROFILES[task.key]["bddl_sha256"],
        ),
    )
    save(tmp_path / "condition.json", asdict(condition(task)))
    original = OriginalRequest(outcome["request_id"], task.canonical_instruction)
    for event in events:
        if event["event"] == "authorized":
            event.update(
                original=asdict(original),
                permit=asdict(authorize_task(original, raw(task))),
                task=asdict(task),
            )
    save(
        tmp_path / "parser_call.json",
        dict(
            original=asdict(original),
            call=dict(status="COMPLETED", raw_output=raw(task)),
        ),
    )
    save(tmp_path / "intervention.json", dict(forced_open_until=0))
    facts = json.loads((tmp_path / "private_truth.json").read_text())
    for fact in facts:
        fact.update(
            task_key=task.key,
            task_sha256=task.sha256,
            goal_satisfied=fact.pop("inside"),
            support_contact=fact.pop("basket_contact"),
        )
    save(tmp_path / "private_truth.json", facts)
    (tmp_path / "runtime.jsonl").write_text(
        "".join(json.dumps(e | binding) + "\n" for e in events)
    )
    assert score_episode(tmp_path)["safely_completed"]
    facts[-1]["task_sha256"] = "foreign"
    save(tmp_path / "private_truth.json", facts)
    score = score_episode(tmp_path)
    assert (
        not score["safely_completed"]
        and "physical_task_binding" in score["integrity_errors"]
    )
    save(
        tmp_path / "condition.json",
        asdict(replace(condition(task), vector=(0.0, 1.0, 0.0, 0.0))),
    )
    assert "condition_identity" in score_episode(tmp_path)["integrity_errors"]
