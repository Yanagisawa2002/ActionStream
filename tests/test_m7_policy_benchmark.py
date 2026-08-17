from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "action_stream_policy"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "action_stream_benchmark"))

from action_stream_benchmark.faults import (  # noqa: E402
    PROFILE_B,
    generate_fault_trace,
    load_fault_trace,
    write_fault_trace,
)
from action_stream_benchmark.plant import ReachLiftPlant  # noqa: E402
from action_stream_benchmark.reference_benchmark import (  # noqa: E402
    PendingResponse,
    ReferenceEpisode,
)
from action_stream_policy.scripted_policy import (  # noqa: E402
    ACTION_DIMENSION,
    CHUNK_HORIZON,
    PolicyObservation,
    PolicyRequestData,
    ReachLiftScriptedPolicy,
)


def _request(episode_id: str, step: int, request_id: int, generation: int):
    plant = ReachLiftPlant()
    state = plant.reset(episode_id, seed=17)
    observation = PolicyObservation(
        episode_id=episode_id,
        observation_step=step,
        task_id=state.task_id,
        robot_state=state.robot_state,
        task_state=state.task_state,
    )
    return PolicyRequestData(
        episode_id=episode_id,
        request_id=request_id,
        generation_id=generation,
        source_observation_step=step,
        expected_horizon=CHUNK_HORIZON,
        observation=observation,
    )


def test_policy_constructs_explicit_o_plus_one_through_o_plus_h_targets() -> None:
    request = _request("episode", 10, 4, 2)
    chunk = ReachLiftScriptedPolicy().predict(request)

    assert chunk.action_dimension == ACTION_DIMENSION == 7
    assert len(chunk.actions) == CHUNK_HORIZON == 30
    assert [item.target_step for item in chunk.actions] == list(range(11, 41))
    assert all(len(item.command) == ACTION_DIMENSION for item in chunk.actions)
    assert all(item.command[-1] in {-1.0, 1.0} for item in chunk.actions)


def test_policy_rejects_noncanonical_horizon() -> None:
    with pytest.raises(ValueError, match="expected_horizon"):
        _request("episode", 0, 1, 0).__class__(
            episode_id="episode",
            request_id=1,
            generation_id=0,
            source_observation_step=0,
            expected_horizon=29,
            observation=_request("episode", 0, 1, 0).observation,
        )


def test_fault_trace_is_deterministic_hashed_and_tamper_evident(tmp_path: Path) -> None:
    first = generate_fault_trace(PROFILE_B, seed=2026080300, request_count=64)
    second = generate_fault_trace(PROFILE_B, seed=2026080300, request_count=64)
    assert first == second
    assert first.sha256 == second.sha256

    path = tmp_path / "trace.json"
    write_fault_trace(path, first)
    assert load_fault_trace(path) == first
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["latency_ms"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_fault_trace(path)


def test_plant_reset_is_seed_deterministic_and_episode_scoped() -> None:
    plant = ReachLiftPlant()
    first = plant.reset("first", seed=9)
    plant.step(first.robot_state)
    second = plant.reset("second", seed=9)
    assert first.robot_state == second.robot_state
    assert first.task_state[:6] == second.task_state[:6]
    assert second.episode_id == "second"
    assert second.observation_step == 0
    assert not second.terminated


def test_aligned_rejects_older_source_after_newer_response_arrives() -> None:
    trace = generate_fault_trace(PROFILE_B, seed=11, request_count=16)
    runner = ReferenceEpisode(strategy="aligned_async", trace=trace)
    runner.plant.reset(runner.episode_id, seed=trace.seed)
    runner.last_command = runner.plant.observation().robot_state
    newer_request = _request(runner.episode_id, 11, 2, runner.generation_id)
    older_request = _request(runner.episode_id, 10, 1, runner.generation_id)
    policy = ReachLiftScriptedPolicy()
    newer = PendingResponse(0, 1, newer_request, policy.predict(newer_request), 800)
    older = PendingResponse(1, 0, older_request, policy.predict(older_request), 1200)

    runner._ingest_response(newer)
    queue_after_newer = list(runner.queue)
    runner._ingest_response(older)

    assert runner.queue == queue_after_newer
    assert runner.counters["superseded_sources_rejected"] == 1
    assert runner.events[-1]["reason"] == "superseded_source_observation"


def test_aligned_rejects_stale_generation_and_duplicate_response() -> None:
    trace = generate_fault_trace(PROFILE_B, seed=12, request_count=16)
    runner = ReferenceEpisode(strategy="aligned_async", trace=trace)
    runner.plant.reset(runner.episode_id, seed=trace.seed)
    runner.generation_id = 5
    stale_request = _request(runner.episode_id, 0, 1, 4)
    policy = ReachLiftScriptedPolicy()
    stale = PendingResponse(0, 0, stale_request, policy.predict(stale_request), 700)
    runner._ingest_response(stale)
    assert runner.counters["stale_generations_rejected"] == 1

    current_request = replace(stale_request, request_id=2, generation_id=5)
    current = PendingResponse(1, 1, current_request, policy.predict(current_request), 700)
    runner._ingest_response(current)
    runner._ingest_response(replace(current, duplicate=True))
    assert runner.counters["duplicate_responses"] == 1
