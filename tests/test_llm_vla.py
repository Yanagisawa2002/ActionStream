"""CPU boundary tests only, never evidence of real language/vision/native ability."""

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest

from actionstream.llm_vla.contracts import (
    CAPABILITY,
    CheckDecision,
    Execution,
    Observation,
    TaskSpec,
)
from actionstream.llm_vla.qwen import checker_messages, parser_messages
from actionstream.llm_vla.scoring import score
from actionstream.llm_vla.integration import execute


def task(decision="accept"):
    return TaskSpec.parse(
        json.dumps(
            dict(
                version="1.0",
                request_id="opaque",
                decision=decision,
                reason="test",
                **(CAPABILITY if decision == "accept" else dict.fromkeys(CAPABILITY)),
            )
        ),
        "opaque",
    )


def observation(revision=0, identity="o1"):
    raw = dict(
        pixels={
            key: np.zeros((1, 360, 360, 3), np.uint8) for key in ("image", "image2")
        },
        robot_state={
            "eef": {
                "pos": np.zeros((1, 3)),
                "quat": np.zeros((1, 4)),
                "mat": np.zeros((1, 3, 3)),
            },
            "gripper": {key: np.zeros((1, 2)) for key in ("qpos", "qvel")},
            "joints": {key: np.zeros((1, 7)) for key in ("pos", "vel")},
        },
    )
    return raw, Observation.project(raw, "opaque", revision, identity)


def test_oracle_and_mutable_inputs_cannot_cross_public_projection():
    raw, clean = observation()
    raw.update(is_success=True, privileged_state="SECRET")
    raw["robot_state"]["object_position"] = "SECRET"
    public = Observation.project(raw, "opaque", 0, "o1", clean.captured_monotonic)
    assert public.binding() == clean.binding()
    assert set(public.native_input()) == {"pixels", "robot_state"}
    assert "SECRET" not in repr(public)
    raw["pixels"]["image"].fill(255)
    assert public.pixels["image"].max() == 0
    with pytest.raises(ValueError):
        public.pixels["image"].setflags(write=True)
    messages = checker_messages(
        "prompt", public.observation_id, public.checker_images()
    )
    assert "SECRET" not in repr(messages) and "is_success" not in repr(messages)
    text = parser_messages("prompt", "opaque", "user request")
    assert set(json.loads(text[1]["content"][0]["text"])) == {"request_id", "utterance"}


def test_completion_rgb_cache_preserves_pixels_and_cannot_cross_observations():
    from actionstream.llm_vla.temporal_completion import camera_rgb

    raw, _ = observation()
    rng = np.random.default_rng(42)
    for value in raw["pixels"].values():
        value[:] = rng.integers(0, 256, value.shape, dtype=np.uint8)
    public = Observation.project(raw, "opaque", 0, "random")
    expected = camera_rgb(
        {
            "agentview_image": raw["pixels"]["image"][0],
            "robot0_eye_in_hand_image": raw["pixels"]["image2"][0],
        }
    )
    assert np.array_equal(public.completion_rgb, expected)
    assert public.completion_rgb is public.completion_rgb
    with pytest.raises(ValueError):
        public.completion_rgb.setflags(write=True)
    raw["pixels"]["image"].fill(0)
    new = Observation.project(raw, "opaque", 1, "new")
    assert np.array_equal(public.completion_rgb, expected)
    assert not np.array_equal(new.completion_rgb, expected)
    assert new.completion_rgb is not public.completion_rgb


@pytest.mark.parametrize("decision", ["reject", "unknown"])
def test_rejection_prevents_execution_construction(decision):
    with pytest.raises(ValueError):
        Execution(task(decision))


def test_invalid_json_and_forged_accept_fail_closed():
    for raw in ("```json\n{}\n```", "{}", '{"version":"1.0","version":"1.0"}'):
        with pytest.raises(ValueError):
            TaskSpec.parse(raw, "opaque")
    with pytest.raises(ValueError):
        Execution(replace(task(), target="orange_juice"))
    with pytest.raises(ValueError):
        CheckDecision.parse(
            '{"observation_id":"o2","status":"complete","evidence":"visible"}', "o1"
        )


def test_unknown_and_native_boundary_do_not_become_completion():
    _, obs = observation()
    for status, boundary, expected in (
        ("unknown", False, "unknown_stop"),
        ("incomplete", True, "native_boundary_stop"),
    ):
        run = Execution(task())
        assert (
            run.apply_check(CheckDecision("o1", status, "visible"), obs, obs, boundary)
            == expected
        )
        with pytest.raises(ValueError):
            run.queue(np.zeros((30, 7)), obs)


def test_old_identity_hash_timestamp_revision_cannot_stop_a_task():
    _, obs = observation()
    run = Execution(task())
    decision = CheckDecision("o1", "complete", "visible")
    for other in (
        replace(obs, observation_id="old"),
        replace(obs, frame_hashes=("old", "old")),
        replace(obs, revision=1),
        replace(obs, captured_monotonic=time.monotonic() - 31),
    ):
        with pytest.raises(ValueError):
            run.apply_check(decision, obs, other)
    assert run.status == "accepted"


def test_interruption_discards_chunk_and_does_not_reset_total_budget():
    _, obs = observation()
    run = Execution(task(), max_steps=68)
    for _ in range(2):
        run.queue(np.zeros((30, 7)), obs)
        for _ in range(30):
            run.take_action()
    run.queue(np.ones((30, 7)), obs)
    for _ in range(7):
        run.take_action()
    assert run.steps == 67 and run.interrupt() == 23
    with pytest.raises(ValueError):
        run.take_action()
    with pytest.raises(ValueError):
        run.queue(np.zeros((30, 7)), obs)
    _, fresh = observation(1, "fresh")
    run.apply_check(CheckDecision("fresh", "incomplete", "outside"), fresh, fresh)
    run.queue(np.zeros((30, 7)), fresh)
    assert np.array_equal(run.take_action(), np.zeros(7)) and run.steps == 68
    with pytest.raises(ValueError):
        run.take_action()
    with pytest.raises(ValueError):
        run.interrupt()


def test_invalid_actions_never_consume_or_enter_queue():
    run = Execution(task())
    _, obs = observation()
    for actions in (np.zeros((30, 20)), np.full((30, 7), np.nan), np.full((30, 7), 2)):
        with pytest.raises(ValueError):
            run.queue(actions, obs)
    assert run.steps == 0 and not run.pending


def test_not_run_or_model_error_cannot_pass_a_gate():
    config = json.loads(
        (Path(__file__).parents[1] / "configs/llm_vla_dispatch003.json").read_text()
    )
    answers = [{"request_id": "opaque", "expected_accept": True}]
    for receipt in (
        {"status": "NOT_RUN"},
        {
            "status": "ERROR",
            "cases": [
                {
                    "identity": "opaque",
                    "status": "ERROR",
                    "raw_output": None,
                    "error": "timeout",
                }
            ],
        },
    ):
        result = score("language", receipt, answers, config)
        assert result["status"] == "FAIL" and result["correct_accepts"] == 0


def test_rejected_request_cannot_touch_native_port():
    class ForbiddenPort:
        def start(self, *args):
            pytest.fail("Rejected request reached native reset")

    with pytest.raises(ValueError):
        execute(task("reject"), ForbiddenPort(), None, lambda *args, **kwargs: None)


def test_public_loop_unknown_at_native_boundary_never_reads_oracle():
    class Port:
        def start(self, request_id, revision):
            return observation()[1]

        def infer(self, obs, instruction):
            assert instruction == CAPABILITY["canonical_instruction"]
            assert not hasattr(obs, "is_success")
            return np.zeros((30, 7)), {}

        def step(self, action, request_id, revision, control_step):
            assert control_step == 1
            return observation(identity="terminal_view")[1], True

    def check(obs):
        return CheckDecision(obs.observation_id, "unknown", "occluded"), {}

    events = []
    result = execute(
        task(), Port(), check, lambda event, **data: events.append((event, data))
    )
    assert result["status"] == "unknown_stop" and not result["checker_complete"]
    assert result["control_steps"] == 1 and result["native_boundary"]
    assert len([e for e, _ in events if e == "checker"]) == 1


def test_public_loop_error_keeps_spent_control_call_without_retry():
    class Port:
        def start(self, request_id, revision):
            return observation()[1]

        def infer(self, obs, instruction):
            return np.zeros((30, 7)), {}

        def step(self, *args):
            raise RuntimeError("native step failed after dispatch")

    result = execute(task(), Port(), None, lambda *args, **kwargs: None)
    assert result["status"] == "ERROR" and result["control_steps"] == 1
    assert result["checks"] == 0 and not result["checker_complete"]


def test_failed_language_gate_blocks_native_import_and_model_load(
    tmp_path, monkeypatch
):
    from actionstream.llm_vla import native_runner

    config = tmp_path / "config.json"
    config.write_text("{}")
    (tmp_path / "language_score.json").write_text('{"status":"FAIL"}')
    output = tmp_path / "native"
    monkeypatch.setattr(
        "sys.argv",
        [
            "native_runner",
            "--config",
            str(config),
            "--model",
            str(tmp_path),
            "--output",
            str(output),
            "--base-store",
            str(tmp_path),
            "--evidence",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        native_runner,
        "LocalQwen",
        lambda *args: pytest.fail("Model loaded after failed gate"),
    )
    with pytest.raises(ValueError, match="Both real gates"):
        native_runner.main()
    assert not output.exists()
