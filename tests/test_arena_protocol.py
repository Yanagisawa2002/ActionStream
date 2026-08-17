from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from actionstream.arena_protocol import (
    ArenaProtocolError,
    load_arena_protocol,
    validate_arena_result,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "isaaclab_arena_frozen_v1.json"


def test_frozen_arena_matrix_is_vectorized_disjoint_and_paired() -> None:
    protocol = load_arena_protocol(PROTOCOL_PATH)

    development = protocol.cells("development")
    holdout = protocol.cells("holdout")
    assert len(development) == 180
    assert len(holdout) == 270
    assert {cell.task_family for cell in holdout} == {
        "pick_and_place",
        "press_button",
        "put_and_close_door",
    }
    assert all(cell.num_envs == 8 for cell in development + holdout)

    paired: dict[str, set[str]] = {}
    for cell in holdout:
        paired.setdefault(cell.pair_id, set()).add(cell.runtime)
    expected = {
        "sync",
        "latest_only",
        "rtc",
        "actionstream_aligned",
        "actionstream_guarded",
    }
    assert paired
    assert all(runtimes == expected for runtimes in paired.values())

    raw = protocol.raw
    dev = raw["splits"]["development"]
    sealed = raw["splits"]["holdout"]
    assert set(dev["reset_seeds"]).isdisjoint(sealed["reset_seeds"])
    assert set(dev["network_trace_ids"]).isdisjoint(sealed["network_trace_ids"])


def test_job_renderer_uses_one_seed_per_official_arena_process() -> None:
    protocol = load_arena_protocol(PROTOCOL_PATH)
    cell = protocol.cells("holdout")[0]
    job = protocol.arena_job_for(cell)

    assert len(job["jobs"]) == 1
    row = job["jobs"][0]
    assert row["arena_env_args"]["num_envs"] == 8
    assert row["num_episodes"] == 24
    assert row["policy_config_dict"]["reset_seed"] == cell.reset_seed
    assert row["policy_config_dict"]["protocol_sha256"] == protocol.sha256


def test_protocol_rejects_split_overlap(tmp_path: Path) -> None:
    raw = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw["splits"]["holdout"]["reset_seeds"][0] = raw["splits"]["development"][
        "reset_seeds"
    ][0]
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ArenaProtocolError, match="reset seeds must be disjoint"):
        load_arena_protocol(path)


def test_protocol_rejects_task_family_relabel(tmp_path: Path) -> None:
    raw = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw["tasks"][1]["arena_env_args"]["environment"] = raw["tasks"][0][
        "arena_env_args"
    ]["environment"]
    path = tmp_path / "duplicate_env.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        ArenaProtocolError, match="Arena environments contains duplicates"
    ):
        load_arena_protocol(path)


def test_structural_smoke_cannot_be_reported_as_success() -> None:
    protocol = load_arena_protocol(PROTOCOL_PATH)
    record = {
        "protocol_sha256": protocol.sha256,
        "cell_id": protocol.cells("development")[0].cell_id,
        "arena_commit": protocol.raw["arena_source"]["commit"],
        "lerobot_commit": protocol.raw["lerobot_source"]["commit"],
        "policy_revision": protocol.raw["models"][0]["revision"],
        "status": "success",
        "evidence_level": "simulator_smoke",
    }

    with pytest.raises(ArenaProtocolError, match="success requires"):
        validate_arena_result(record, protocol)


def test_learned_result_requires_runtime_metrics_and_replay_evidence() -> None:
    protocol = load_arena_protocol(PROTOCOL_PATH)
    record = {
        "protocol_sha256": protocol.sha256,
        "cell_id": protocol.cells("development")[0].cell_id,
        "arena_commit": protocol.raw["arena_source"]["commit"],
        "lerobot_commit": protocol.raw["lerobot_source"]["commit"],
        "policy_revision": protocol.raw["models"][0]["revision"],
        "status": "success",
        "evidence_level": "learned_policy_rollout",
        "metrics": {"success": True},
    }

    with pytest.raises(
        ArenaProtocolError, match="lacks required Arena/runtime metrics"
    ):
        validate_arena_result(record, protocol)

    complete = copy.deepcopy(record)
    complete["metrics"] = {
        "success": True,
        "gpu_parallel_env_steps_per_second": 123.0,
        "queue_age_steps": 2.0,
        "fallback_count": 0,
    }
    complete["video_paths"] = ["videos/example.mp4"]
    complete["episode_provenance_path"] = "episodes.jsonl"
    validate_arena_result(complete, protocol)
