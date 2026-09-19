"""Freeze outcome-informed v2 without running any environment or GPU code."""

import hashlib
import json
from pathlib import Path

from actionstream.libero_rpc_client import build_parser
from actionstream.rpc_transport import TcpInferenceTransport

ROOT = Path(__file__).resolve().parents[1]


def test_v2_keeps_exact_v1_cohort_order_faults_and_evidence_identity():
    v1_path = ROOT / "configs/rpc_remote_transport_replication_v1.json"
    v1 = json.loads(v1_path.read_text())
    v2 = json.loads(
        (ROOT / "configs/rpc_remote_transport_replication_v2.json").read_text()
    )
    assert (
        hashlib.sha256(v1_path.read_bytes()).hexdigest()
        == "b2fc8f8a0f7f66eb60cddbd79645647b7313435deaafc84831dd396616749019"
    )
    for field in (
        "model_id",
        "model_revision",
        "lerobot_commit",
        "task_families",
        "stages",
        "run_matrix",
        "run_order",
        "seed_mapping",
        "coverage",
        "hard_gates",
        "attempt_policy",
        "controller_frequency_hz",
        "max_control_steps",
    ):
        assert v2[field] == v1[field], field
    assert len(v2["run_matrix"]) == 51
    assert v2["protocol_id"] == "rpc_remote_transport_replication_v2"
    assert v2["status"] == "PRE_REGISTERED_NOT_EXECUTED"
    assert v2["claim_boundary"]["outcome_informed_by_v1"] is True
    assert v2["claim_boundary"]["independent_replication_of_v1"] is False
    assert not v2["provenance"]["no_remote_rpc_outcome_observed_before_protocol_change"]
    assert v2["provenance"]["v1"]["run_1_classification"] == "VALID_NEGATIVE"
    assert v2["provenance"]["v1"]["permanent_completed_runs"] == 1
    assert (
        v2["provenance"]["v1"]["evidence_manifest_sha256"]
        == "9b0e223c00b92f907a7d0c0803c94f41a07e01a46fe052d0ac28bc4f9609329d"
    )
    assert (
        v2["provenance"]["v1"]["postmortem_manifest_sha256"]
        == "f515e1f6324c34d7c0cd4933df3c437971bfe1c34e8ac399fff504aa9171a0be"
    )
    assert not v2["identity_rule"]["freshness_required"]
    assert v2["execution_requirements"]["fresh_server_process_each_episode_condition"]
    assert v2["execution_requirements"]["distinct_physical_hosts_required"]
    assert v2["execution_requirements"]["no_unreported_inference_warmup"]
    assert v2["execution_requirements"][
        "protocol_commit_must_exist_before_gpu_execution"
    ]
    assert v2["transport"]["max_concurrent_policy_calls"] == 1


def test_v2_parameters_are_explicit_cli_overrides_with_consistent_outer_wait(tmp_path):
    v2 = json.loads(
        (ROOT / "configs/rpc_remote_transport_replication_v2.json").read_text()
    )
    params = v2["client_runtime_parameters"]
    argv = [
        "--suite",
        "libero_object",
        "--task-id",
        "5",
        "--initial-state-index",
        "25",
        "--seed",
        "2026091725",
        "--host",
        "127.0.0.1",
        "--output",
        str(tmp_path),
    ]
    for name, value in params.items():
        flag = "--" + name.replace("_", "-")
        argv.extend([flag] if value is True else [flag, str(value)])
    args = build_parser().parse_args(argv)
    for name, value in params.items():
        assert getattr(args, name) == value
    assert args.startup_inference_timeout_s == 15.0
    assert args.steady_inference_timeout_s == args.inference_timeout_s == 5.0
    assert args.connect_timeout_s == args.control_timeout_s == 3.0
    assert args.action_wait_timeout_s == 20.0
    assert (
        args.action_wait_timeout_s
        > args.startup_inference_timeout_s + args.control_timeout_s
    )
    client = TcpInferenceTransport(
        args.host,
        args.port,
        startup_inference_timeout_s=args.startup_inference_timeout_s,
        steady_inference_timeout_s=args.steady_inference_timeout_s,
    )
    assert client.next_inference_timeout_s(args.inference_timeout_s) == 15.0
    client.close()
    assert build_parser().get_default("action_wait_timeout_s") == 10.0
    assert build_parser().get_default("startup_inference_timeout_s") is None


def test_v2_claim_limits_and_timing_contract_are_explicit():
    v2 = json.loads(
        (ROOT / "configs/rpc_remote_transport_replication_v2.json").read_text()
    )
    assert {
        "unseen_state_external_validity",
        "unseen_task_generalization",
        "physical_robot_safety",
        "remote_cuda_kernel_preemption",
        "independent_replication_of_v1",
        "15_second_measured_SLA",
    } <= set(v2["claim_boundary"]["unsupported_claims"])
    assert {
        "server_queue_wait_s",
        "server_compute_s",
        "server_service_s",
        "server_inference_s",
        "connection_id",
        "connection_request_ordinal",
        "global_request_ordinal",
    } <= v2["response_telemetry"].keys()
    assert "Omitted" in v2["response_telemetry"]["server_lock_wait_s"]
    assert "not a measured SLA" in v2["deadline_semantics"]["startup_budget_rationale"]
