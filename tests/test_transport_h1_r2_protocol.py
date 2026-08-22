from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/actionstream_transport_h1_r2_registry.json"
ORCHESTRATOR_PATH = ROOT / "scripts/experiments/run_transport_h1_r2.py"
V2_ORCHESTRATOR_PATH = ROOT / "scripts/experiments/run_transport_h1_r2_v2.py"
V2_CONFIG_PATH = ROOT / "configs/actionstream_transport_h1_r2_resume_v2.json"
FAILURE_RECEIPT_PATH = (
    ROOT / "reports/actionstream_transport_h1_r2/preflight_failure_v1.json"
)


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


def _orchestrator():
    spec = importlib.util.spec_from_file_location("run_transport_h1_r2", ORCHESTRATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v2_orchestrator():
    spec = importlib.util.spec_from_file_location(
        "run_transport_h1_r2_v2", V2_ORCHESTRATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h1_r2_candidate_and_protocol_hashes_are_frozen() -> None:
    registry = _registry()
    candidate = registry["candidate"]
    base_commit = candidate["base_commit"]
    for key, relative in {
        "current_baselines_sha256": "src/actionstream/current_baselines.py",
        "lerobot_inference_sha256": "src/actionstream/lerobot_inference.py",
        "gpu_measurement_sha256": "src/actionstream/gpu_measurement.py",
    }.items():
        assert candidate[key] == _sha256(ROOT / relative)
        assert candidate[key] == _git_object_sha256(base_commit, relative)
    assert candidate["orchestrator_sha256"] == _sha256(ORCHESTRATOR_PATH)
    for family, path, raw in _protocols():
        assert registry["protocols"][family]["sha256"] == _sha256(path)
        assert raw["candidate"] == candidate


def test_h1_r2_is_only_one_two_runtime_fixed_delay_hypothesis() -> None:
    expected_runtimes = {
        "actionstream_backend_aligned",
        "actionstream_backend_pipelined_aligned",
    }
    orders = []
    for family, _path, raw in _protocols():
        assert raw["task_family"] == family
        assert raw["protocol_status"] == "frozen_holdout"
        assert raw["hypothesis_id"] == "transport_h1_r2_delivery_pipeline"
        assert set(raw["runtimes"]) == expected_runtimes
        assert set(raw["runtime_execution_order"]) == expected_runtimes
        orders.append(raw["runtime_execution_order"])
        assert raw["delay_profiles"] == [
            {
                "key": "fixed_0950_transport_h1_r2",
                "kind": "fixed",
                "milliseconds": 950,
            }
        ]
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
    assert sum(order[0] == "actionstream_backend_aligned" for order in orders) == 2
    assert (
        sum(
            order[0] == "actionstream_backend_pipelined_aligned"
            for order in orders
        )
        == 1
    )


def test_h1_r2_scored_identities_are_new_and_bounded() -> None:
    registry = _registry()
    expected_states = {10, 11, 12, 13, 14}
    seeds = set()
    for _family, _path, raw in _protocols():
        assert set(raw["environment"]["initial_state_indices"]) == expected_states
        assert set(raw["evidence_boundary"]["scored_states"]) == expected_states
        prior_key = (
            f"{raw['environment']['suite']}_task"
            f"{raw['environment']['task_ids'][0]}"
        )
        prior = set(
            registry["identity_audit"]["prior_exact_task_reserved_or_scored_states"][
                prior_key
            ]
        )
        assert expected_states.isdisjoint(prior)
        assert raw["measurement_v2"]["worker_warmup"]["initial_state_index"] == 45
        assert raw["measurement_v2"]["worker_warmup"]["inference_calls"] == 2
        seeds.add(raw["environment"]["base_seed"])
    assert len(seeds) == 3
    matrix = registry["formal_matrix"]
    assert matrix["paired_identities"] == 15
    assert matrix["scored_episodes"] == 30
    assert matrix["fresh_runtime_family_processes"] == 6
    assert matrix["gpu_session_budget"] == 1
    assert registry["recovery_semantics"] == {
        "old_h1_status": "incomplete_unpaired",
        "old_h1_outputs_reused": False,
        "old_h1_scored_identities_reused": False,
        "r2_is_new_experiment_not_resume": True,
    }


def test_h1_r2_decision_boundary_cannot_be_broadened() -> None:
    registry = _registry()
    gates = registry["decision_gates"]
    assert gates["minimum_aggregate_request_rate_ratio"] == 4.0
    assert gates["minimum_aggregate_depletion_reduction_fraction"] == 0.8
    assert gates["maximum_pipelined_aggregate_depletion_fraction"] == 0.05
    assert gates["minimum_task_families_with_lower_depletion"] == 3
    assert gates["maximum_out_of_order_rejections"] == 0
    assert gates["maximum_fallback_activations"] == 0
    assert gates["pipelined_success_count_must_be_at_least_serialized"] is True
    assert registry["analysis_plan"]["bootstrap_resamples"] == 10000
    assert all(registry["single_hypothesis_guard"].values())
    assert "real remote RPC" in registry["claim_boundary"]


def test_h1_r2_orchestrator_accepts_only_the_frozen_matrix() -> None:
    module = _orchestrator()
    assert module.ALLOWED_RUNTIMES == {
        "actionstream_backend_aligned",
        "actionstream_backend_pipelined_aligned",
    }
    assert module.EXPECTED_STATES == [10, 11, 12, 13, 14]
    assert (
        module.EXPECTED_OUTPUT_BASENAME
        == "actionstream_transport_h1_r2_holdout_20260822"
    )
    for _family, path, raw in _protocols():
        assert module._validate_protocol(path) == raw


def test_h1_r2_cell_validator_uses_profiles_and_exact_identities(
    tmp_path: Path,
) -> None:
    module = _orchestrator()
    _family, _path, protocol = _protocols()[0]
    runtime = "actionstream_backend_aligned"
    cell = tmp_path / "cell"
    (cell / "traces").mkdir(parents=True)
    (cell / "videos").mkdir()
    receipt = {
        "selection": {
            "runtimes": [runtime],
            "profiles": ["fixed_0950_transport_h1_r2"],
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
    records = []
    for index, state in enumerate(module.EXPECTED_STATES):
        records.append(
            {
                "suite": protocol["environment"]["suite"],
                "task_id": protocol["environment"]["task_ids"][0],
                "initial_state_index": state,
                "seed": protocol["environment"]["base_seed"] + index,
                "runtime": runtime,
                "worker_warmup": {},
                "inference_requests_per_second": 1.0,
                "depletion_safe_hold_steps": 1,
                "responses_rejected_out_of_order": 0,
                "fallback_activations": 0,
                "engine_config": {"latest_only_fallback": False},
            }
        )
        (cell / "traces" / f"{index}.json").write_text("{}", encoding="utf-8")
        (cell / "traces" / f"{index}.telemetry.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
    (cell / "videos" / "representative.mp4").write_bytes(b"video")
    (cell / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    assert len(module._validate_cell(cell, runtime, protocol)["records"]) == 5

    receipt["selection"]["delay_profiles"] = receipt["selection"].pop("profiles")
    (cell / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    try:
        module._validate_cell(cell, runtime, protocol)
    except RuntimeError as error:
        assert "wrong profile" in str(error)
    else:
        raise AssertionError("H1-R2 accepted the old receipt key")


def test_h1_r2_v2_is_a_hash_bounded_preflight_only_resume() -> None:
    module = _v2_orchestrator()
    resume = json.loads(V2_CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = resume["frozen_artifacts"]
    assert resume["status"] == "frozen_preflight_resume_before_formal_root"
    assert frozen["registry_sha256"] == _sha256(REGISTRY_PATH)
    assert frozen["v1_orchestrator_sha256"] == _sha256(ORCHESTRATOR_PATH)
    assert frozen["v2_orchestrator_sha256"] == _sha256(V2_ORCHESTRATOR_PATH)
    assert frozen["preflight_failure_receipt_sha256"] == _sha256(
        FAILURE_RECEIPT_PATH
    )
    assert resume["preflight_failure"]["formal_output_root_created"] is False
    assert resume["preflight_failure"]["gpu_child_processes_started"] == 0
    assert resume["preflight_failure"]["scored_episodes_started"] == 0
    assert all(resume["unchanged_scientific_contract"].values())
    assert module._validate_resume_config(V2_CONFIG_PATH) == resume


def test_h1_r2_v2_preserves_the_active_venv_interpreter() -> None:
    module = _v2_orchestrator()
    captured = {}
    original = module._V1_PREFLIGHT

    def fake_preflight(**kwargs):
        captured.update(kwargs)
        return {"status": "fixture"}

    try:
        module._V1_PREFLIGHT = fake_preflight
        assert module._venv_preflight(
            python=Path("/resolved/base/python"),
            lerobot_root=Path("/lerobot"),
            environment={"PYTHONPATH": "fixture"},
        ) == {"status": "fixture"}
    finally:
        module._V1_PREFLIGHT = original
    assert captured["python"] == Path(sys.executable)
