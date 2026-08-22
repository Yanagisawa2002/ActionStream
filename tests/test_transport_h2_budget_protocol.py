from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/actionstream_transport_h2_budget_registry.json"
ORCHESTRATOR_PATH = ROOT / "scripts/experiments/run_transport_h2_budget.py"
RUNTIMES = {
    "actionstream_backend_pipelined_aligned": 1,
    "actionstream_backend_budgeted_pipelined_aligned": 5,
}


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
    spec = importlib.util.spec_from_file_location(
        "run_transport_h2_budget", ORCHESTRATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h2_candidate_and_protocol_hashes_are_frozen() -> None:
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


def test_h2_changes_only_the_frozen_request_budget() -> None:
    orders = []
    for family, _path, raw in _protocols():
        assert raw["task_family"] == family
        assert raw["protocol_status"] == "frozen_holdout"
        assert raw["hypothesis_id"] == "transport_h2_compute_budget"
        assert set(raw["runtimes"]) == set(RUNTIMES)
        assert set(raw["runtime_execution_order"]) == set(RUNTIMES)
        orders.append(raw["runtime_execution_order"])
        assert raw["delay_profiles"] == [
            {
                "key": "fixed_0950_transport_h2_budget",
                "kind": "fixed",
                "milliseconds": 950,
            }
        ]
        configs = {
            runtime: dict(raw["actionstream_backend"][runtime])
            for runtime in RUNTIMES
        }
        for runtime, interval in RUNTIMES.items():
            assert configs[runtime].pop("minimum_request_interval_steps") == interval
        assert all(
            config
            == {
                "latest_only_fallback": False,
                "delivery_scheduler_enabled": True,
            }
            for config in configs.values()
        )
    assert sum(order[0].startswith("actionstream_backend_budgeted") for order in orders) == 2
    assert sum(order[0] == "actionstream_backend_pipelined_aligned" for order in orders) == 1


def test_h2_identities_are_new_and_matrix_is_bounded() -> None:
    registry = _registry()
    expected_states = {15, 16, 17, 18, 19}
    seeds = set()
    for _family, _path, raw in _protocols():
        assert set(raw["environment"]["initial_state_indices"]) == expected_states
        assert set(raw["evidence_boundary"]["scored_states"]) == expected_states
        prior_key = (
            f"{raw['environment']['suite']}_task"
            f"{raw['environment']['task_ids'][0]}"
        )
        prior = set(
            registry["identity_audit"][
                "prior_exact_task_reserved_or_scored_states"
            ][prior_key]
        )
        assert expected_states.isdisjoint(prior)
        assert raw["measurement_v2"]["worker_warmup"]["initial_state_index"] == 20
        assert raw["measurement_v2"]["worker_warmup"]["inference_calls"] == 2
        seeds.add(raw["environment"]["base_seed"])
    assert len(seeds) == 3
    matrix = registry["formal_matrix"]
    assert matrix["paired_identities"] == 15
    assert matrix["scored_episodes"] == 30
    assert matrix["fresh_runtime_family_processes"] == 6
    assert matrix["gpu_session_budget"] == 1


def test_h2_decision_boundary_cannot_be_broadened() -> None:
    registry = _registry()
    gates = registry["decision_gates"]
    assert gates["maximum_aggregate_inference_call_ratio"] == 0.5
    assert gates["minimum_budgeted_median_request_rate_per_second"] == 2.5
    assert gates["maximum_budgeted_aggregate_depletion_fraction"] == 0.05
    assert gates["maximum_budgeted_depletion_fraction_per_task_family"] == 0.05
    assert gates["budgeted_success_count_must_be_at_least_unbounded"] is True
    assert gates["maximum_out_of_order_rejections"] == 0
    assert gates["maximum_fallback_activations"] == 0
    assert gates["maximum_inference_timeouts"] == 0
    assert gates["maximum_inference_errors"] == 0
    assert registry["budget_choice"]["no_interval_sweep"] is True
    assert registry["analysis_plan"]["bootstrap_resamples"] == 10000
    assert all(registry["single_hypothesis_guard"].values())
    assert "real remote RPC" in registry["claim_boundary"]


def test_h2_orchestrator_accepts_only_the_frozen_matrix() -> None:
    module = _orchestrator()
    assert module.ALLOWED_RUNTIMES == set(RUNTIMES)
    assert module.EXPECTED_INTERVALS == RUNTIMES
    assert module.EXPECTED_STATES == [15, 16, 17, 18, 19]
    assert (
        module.EXPECTED_OUTPUT_BASENAME
        == "actionstream_transport_h2_budget_holdout_20260822"
    )
    for _family, path, raw in _protocols():
        assert module._validate_protocol(path) == raw


def test_h2_cell_validator_checks_budget_telemetry_and_exact_identities(
    tmp_path: Path,
) -> None:
    module = _orchestrator()
    _family, _path, protocol = _protocols()[0]
    runtime = "actionstream_backend_budgeted_pipelined_aligned"
    cell = tmp_path / "cell"
    (cell / "traces").mkdir(parents=True)
    (cell / "videos").mkdir()
    receipt = {
        "selection": {
            "runtimes": [runtime],
            "profiles": ["fixed_0950_transport_h2_budget"],
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
                "inference_requests_per_second": 3.5,
                "inference_calls": 12,
                "inference_latency_p50_seconds": 0.08,
                "inference_latency_p95_seconds": 0.09,
                "depletion_safe_hold_steps": 0,
                "queue_age_p50_steps": 12.0,
                "queue_age_p95_steps": 18.0,
                "observations_skipped_by_budget": 40,
                "responses_rejected_out_of_order": 0,
                "fallback_activations": 0,
                "inference_timeouts": 0,
                "inference_errors": 0,
                "request_interval_steps": 5,
                "engine_config": {
                    "latest_only_fallback": False,
                    "delivery_scheduler_enabled": True,
                    "minimum_request_interval_steps": 5,
                },
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
