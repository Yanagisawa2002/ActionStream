from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from actionstream.current_baselines import load_protocol


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/actionstream_transport_h1_final_registry.json"
ORCHESTRATOR_PATH = ROOT / "scripts/experiments/run_transport_h1_final.py"
RESUME_ORCHESTRATOR_PATH = (
    ROOT / "scripts/experiments/resume_transport_h1_final.py"
)
RESUME_CONFIG_PATH = ROOT / "configs/actionstream_transport_h1_final_resume.json"
RESUME_V2_ORCHESTRATOR_PATH = (
    ROOT / "scripts/experiments/resume_transport_h1_final_v2.py"
)
RESUME_V2_CONFIG_PATH = ROOT / "configs/actionstream_transport_h1_final_resume_v2.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_object_sha256(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _protocols() -> list[tuple[str, Path, dict]]:
    rows = []
    for family, entry in _registry()["protocols"].items():
        path = ROOT / entry["path"]
        rows.append((family, path, json.loads(path.read_text(encoding="utf-8"))))
    return rows


def test_transport_h1_final_candidate_and_protocol_hashes_are_frozen() -> None:
    registry = _registry()
    candidate = registry["candidate"]
    base_commit = candidate["base_commit"]
    for key, relative in {
        "current_baselines_sha256": "src/actionstream/current_baselines.py",
        "lerobot_inference_sha256": "src/actionstream/lerobot_inference.py",
        "gpu_measurement_sha256": "src/actionstream/gpu_measurement.py",
    }.items():
        assert candidate[key] == _git_object_sha256(base_commit, relative)
    assert candidate["orchestrator_sha256"] == _sha256(ORCHESTRATOR_PATH)

    for family, path, raw in _protocols():
        assert registry["protocols"][family]["sha256"] == _sha256(path)
        assert raw["candidate"] == candidate


def test_transport_h1_final_is_one_fixed_delay_hypothesis() -> None:
    expected_runtimes = {
        "actionstream_backend_aligned",
        "actionstream_backend_pipelined_aligned",
        "lerobot_latest_only",
    }
    execution_orders = []
    for family, path, raw in _protocols():
        protocol = load_protocol(path)
        assert protocol.raw == raw
        assert raw["protocol_status"] == "frozen_holdout"
        assert raw["benchmark_version"] == 3
        assert raw["hypothesis_id"] == "transport_h1_final_serialized_delivery"
        assert raw["task_family"] == family
        assert set(raw["runtimes"]) == expected_runtimes
        assert set(raw["runtime_execution_order"]) == expected_runtimes
        execution_orders.append(raw["runtime_execution_order"])
        assert raw["delay_profiles"] == [
            {
                "key": "fixed_0950_transport_h1_final",
                "kind": "fixed",
                "milliseconds": 950,
            }
        ]
        assert raw["compact_matrix"] == {
            "episodes_per_task": 5,
            "paired": True,
            "reuse_delay_trace_across_runtimes": True,
        }
        serialized = dict(
            raw["actionstream_backend"]["actionstream_backend_aligned"]
        )
        pipelined = dict(
            raw["actionstream_backend"][
                "actionstream_backend_pipelined_aligned"
            ]
        )
        assert serialized.pop("delivery_scheduler_enabled") is False
        assert pipelined.pop("delivery_scheduler_enabled") is True
        assert serialized == pipelined == {"latest_only_fallback": False}

    for position in range(3):
        assert {order[position] for order in execution_orders} == expected_runtimes


def test_transport_h1_final_scored_identities_are_disjoint_and_bounded() -> None:
    registry = _registry()
    expected_states = {25, 26, 27, 28, 29}
    seeds = set()
    for family, _path, raw in _protocols():
        assert set(raw["environment"]["initial_state_indices"]) == expected_states
        assert set(raw["evidence_boundary"]["scored_states"]) == expected_states
        prior_key = f"{raw['environment']['suite']}_task{raw['environment']['task_ids'][0]}"
        prior = set(registry["identity_audit"]["prior_exact_task_scored_states"][prior_key])
        assert expected_states.isdisjoint(prior)
        seeds.add(raw["environment"]["base_seed"])
        assert raw["measurement_v2"]["worker_warmup"]["initial_state_index"] == 46
        assert raw["measurement_v2"]["worker_warmup"]["inference_calls"] == 2

    assert len(seeds) == 3
    assert registry["formal_matrix"]["paired_identities"] == 15
    assert registry["formal_matrix"]["scored_episodes"] == 45
    assert registry["formal_matrix"]["fresh_runtime_family_processes"] == 9
    assert registry["formal_matrix"]["gpu_session_budget"] == 1


def test_transport_h1_final_decision_boundary_cannot_be_broadened() -> None:
    registry = _registry()
    gates = registry["decision_gates"]
    assert gates["minimum_aggregate_request_rate_ratio"] == 4.0
    assert gates["minimum_aggregate_depletion_reduction_fraction"] == 0.8
    assert gates["maximum_pipelined_aggregate_depletion_fraction"] == 0.05
    assert gates["minimum_task_families_with_lower_depletion"] == 3
    assert gates["maximum_out_of_order_rejections"] == 0
    assert gates["maximum_fallback_activations"] == 0
    assert gates["maximum_pipelined_success_count_deficit_vs_latest"] == 1
    guard = registry["single_hypothesis_guard"]
    assert all(guard.values())
    assert registry["analysis_plan"]["bootstrap_resamples"] == 10000
    assert "real remote RPC" in registry["claim_boundary"]


def test_transport_h1_final_orchestrator_accepts_only_the_frozen_matrix() -> None:
    spec = importlib.util.spec_from_file_location(
        "run_transport_h1_final", ORCHESTRATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ALLOWED_RUNTIMES == {
        "actionstream_backend_aligned",
        "actionstream_backend_pipelined_aligned",
        "lerobot_latest_only",
    }
    candidate = _registry()["candidate"]
    current_candidate_matches = all(
        candidate[key] == _sha256(ROOT / relative)
        for key, relative in {
            "current_baselines_sha256": "src/actionstream/current_baselines.py",
            "lerobot_inference_sha256": "src/actionstream/lerobot_inference.py",
            "gpu_measurement_sha256": "src/actionstream/gpu_measurement.py",
        }.items()
    )
    for _family, path, raw in _protocols():
        if current_candidate_matches:
            assert module._validate_protocol(path).raw == raw
        else:
            with pytest.raises(RuntimeError, match="candidate hash mismatch"):
                module._validate_protocol(path)


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transport_h1_final_structural_resume_is_hash_bounded() -> None:
    resume = json.loads(RESUME_CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = resume["frozen_artifacts"]
    assert resume["status"] == "frozen_structural_resume_before_paired_result"
    assert resume["original_registry_sha256"] == _sha256(REGISTRY_PATH)
    assert frozen["original_orchestrator_sha256"] == _sha256(ORCHESTRATOR_PATH)
    assert frozen["resume_orchestrator_sha256"] == _sha256(
        RESUME_ORCHESTRATOR_PATH
    )
    assert resume["structural_failure"]["paired_contrast_available_before_fix"] is False
    assert resume["resume_contract"]["remaining_cells"] == 8
    assert resume["resume_contract"]["duplicated_scored_episodes_allowed"] == 0
    assert all(resume["unchanged_scientific_contract"].values())
    module = _load_script("resume_transport_h1_final", RESUME_ORCHESTRATOR_PATH)
    assert module._validate_resume_config(RESUME_CONFIG_PATH) == resume


def test_transport_h1_final_resume_validates_actual_profiles_key(tmp_path: Path) -> None:
    module = _load_script("resume_transport_h1_final_fixture", RESUME_ORCHESTRATOR_PATH)
    cell = tmp_path / "cell"
    (cell / "traces").mkdir(parents=True)
    (cell / "videos").mkdir()
    receipt = {
        "selection": {
            "runtimes": ["actionstream_backend_aligned"],
            "profiles": ["fixed_0950_transport_h1_final"],
        },
        "system_gpu": {
            "sample_count": 2,
            "phases": {
                "steady_state": {
                    "resident_sample_count": 1,
                    "process_gpu_memory_mib_max": 100.0,
                }
            },
        },
    }
    (cell / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    records = [
        {
            "worker_warmup": {},
            "inference_requests_per_second": 1.0,
            "system_gpu": {},
        }
        for _ in range(5)
    ]
    (cell / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    for index in range(5):
        (cell / "traces" / f"{index}.json").write_text("{}", encoding="utf-8")
    (cell / "videos" / "representative.mp4").write_bytes(b"video")
    assert len(module._validate_cell(cell, "actionstream_backend_aligned")["records"]) == 5

    receipt["selection"]["delay_profiles"] = receipt["selection"].pop("profiles")
    (cell / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="wrong profile"):
        module._validate_cell(cell, "actionstream_backend_aligned")


def test_transport_h1_final_structural_resume_v2_is_hash_bounded() -> None:
    resume = json.loads(RESUME_V2_CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = resume["frozen_artifacts"]
    assert resume["status"] == "frozen_structural_resume_v2_before_paired_result"
    assert frozen["resume_v1_config_sha256"] == _sha256(RESUME_CONFIG_PATH)
    assert frozen["resume_v1_orchestrator_sha256"] == _sha256(RESUME_ORCHESTRATOR_PATH)
    assert frozen["resume_v2_orchestrator_sha256"] == _sha256(
        RESUME_V2_ORCHESTRATOR_PATH
    )
    assert resume["structural_failure_v2"]["new_gpu_child_process_started"] is False
    assert resume["structural_failure_v2"]["new_scored_episode_started"] is False
    assert resume["resume_contract"]["remaining_cells"] == 8
    assert resume["resume_contract"]["duplicated_scored_episodes_allowed"] == 0
    assert all(resume["unchanged_scientific_contract"].values())


def test_transport_h1_final_resume_v2_accepts_receipt_level_gpu_only(
    tmp_path: Path,
) -> None:
    v1 = _load_script("resume_transport_h1_final_v1_fixture", RESUME_ORCHESTRATOR_PATH)
    v2 = _load_script(
        "resume_transport_h1_final_v2_fixture", RESUME_V2_ORCHESTRATOR_PATH
    )
    cell = tmp_path / "cell"
    (cell / "traces").mkdir(parents=True)
    (cell / "videos").mkdir()
    receipt = {
        "selection": {
            "runtimes": ["actionstream_backend_aligned"],
            "profiles": ["fixed_0950_transport_h1_final"],
        },
        "system_gpu": {
            "sample_count": 2,
            "phases": {
                "steady_state": {
                    "resident_sample_count": 1,
                    "process_gpu_memory_mib_max": 100.0,
                }
            },
        },
    }
    (cell / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    records = [
        {"worker_warmup": {}, "inference_requests_per_second": 1.0} for _ in range(5)
    ]
    (cell / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    for index in range(5):
        (cell / "traces" / f"{index}.json").write_text("{}", encoding="utf-8")
        (cell / "traces" / f"{index}.telemetry.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
    (cell / "videos" / "representative.mp4").write_bytes(b"video")

    with pytest.raises(RuntimeError, match="Missing GPU measurement"):
        v1._validate_cell(cell, "actionstream_backend_aligned")
    assert len(v2._validate_cell(cell, "actionstream_backend_aligned")["records"]) == 5
