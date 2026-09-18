"""Guard the preregistered cohort and claims without creating an environment."""

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "configs/rpc_external_validity_v1.json"
CONFIG = ROOT / "configs/rpc_remote_transport_replication_v1.json"


def test_replication_preserves_aborted_protocol_and_mechanism_design():
    old = json.loads(V1.read_text(encoding="utf-8"))
    new = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert hashlib.sha256(V1.read_bytes()).hexdigest() == (
        "7d663780396e28c0bc4b867a363e087a40a09cd8bc84e1fab9d24122bc614887"
    )
    for field in (
        "model_id",
        "model_revision",
        "lerobot_commit",
        "controller_frequency_hz",
        "max_control_steps",
        "transport",
        "task_families",
        "stages",
        "required_metrics",
        "hard_gates",
    ):
        assert new[field] == old[field], field
    assert new["protocol_id"] == "rpc_remote_transport_replication_v1"
    assert new["status"] == "PRE_REGISTERED_NOT_EXECUTED"
    assert new["identity_rule"]["historically_executed_identities_permitted"]
    assert not new["identity_rule"]["freshness_required"]
    assert new["identity_rule"]["unknown_is_not_reclassified_as_clean"]
    assert not new["identity_rule"]["training_finetuning_or_parameter_updates_allowed"]
    assert not new["identity_rule"]["outcome_conditioned_identity_selection_allowed"]
    assert new["claim_boundary"][
        "historical_h1_h2_task_success_is_not_a_clean_causal_baseline"
    ]
    assert not new["claim_boundary"]["external_task_generalization_established"]


def test_exact_paired_51_run_coverage_and_single_canary():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = []
    for stage in cfg["stages"]:
        for family in cfg["task_families"]:
            for position in stage["episode_positions"]:
                episode = family["candidate_episodes"][position]
                for condition in stage["conditions"]:
                    expected.append(
                        {
                            "run_id": len(expected) + 1,
                            "stage": stage["name"],
                            "family": family["family"],
                            "suite": family["suite"],
                            "task_id": family["task_id"],
                            **episode,
                            "condition": condition["name"],
                        }
                    )
    rows = cfg["run_matrix"]
    assert rows == expected
    assert len(rows) == 51
    assert (
        len(
            {
                (r["suite"], r["task_id"], r["initial_state_index"], r["condition"])
                for r in rows
            }
        )
        == 51
    )
    assert Counter(r["stage"] for r in rows) == {
        "remote_no_fault": 15,
        "remote_fault_matrix": 36,
    }
    assert Counter(r["suite"] for r in rows) == {
        "libero_object": 17,
        "libero_spatial": 17,
        "libero_goal": 17,
    }
    assert (
        rows[0]["suite"],
        rows[0]["task_id"],
        rows[0]["initial_state_index"],
        rows[0]["seed"],
        rows[0]["condition"],
    ) == (
        "libero_object",
        5,
        25,
        2026091725,
        "tcp_no_injected_fault",
    )
    for row in rows:
        assert row["seed"] == 2026091700 + row["initial_state_index"]
        assert row["seed"] == cfg["seed_mapping"][str(row["initial_state_index"])]
    assert cfg["formal_canary"]["run_id"] == 1
    assert cfg["formal_canary"]["counts_toward_51"]
    assert not cfg["formal_canary"]["rerun_merely_because_called_smoke"]
    assert not cfg["attempt_policy"]["valid_outcome_rerun_allowed"]
    assert cfg["attempt_policy"]["valid_task_failure_counts"]
    table = (ROOT / "docs/rpc-remote-transport-replication-v1-matrix.md").read_text(
        encoding="utf-8"
    )
    aliases = dict(
        zip(
            [c["name"] for s in cfg["stages"] for c in s["conditions"]],
            ["N", "J50", "J250", "J950", "D7"],
            strict=True,
        )
    )
    for row in rows:
        assert (
            f"| {row['run_id']} | {row['suite']} | {row['task_id']} | "
            f"{row['initial_state_index']} | {row['seed']} | "
            f"{aliases[row['condition']]} |"
        ) in table


def test_frozen_client_defaults_match_actual_parser_without_importing_libero():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = ast.parse(
        (ROOT / "src/actionstream/libero_rpc_client.py").read_text(encoding="utf-8")
    )
    defaults = {}
    for node in ast.walk(source):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
        ):
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                defaults[node.args[0].value.removeprefix("--").replace("-", "_")] = (
                    ast.literal_eval(keyword.value)
                )
    for name, value in cfg["client_runtime_parameters"].items():
        assert defaults[name] == value, name
    assert defaults["max_control_steps"] == cfg["max_control_steps"]
    assert defaults["expected_frequency_hz"] == cfg["controller_frequency_hz"]
    assert defaults["inference_timeout_s"] == cfg["transport"]["client_deadline_s"]
    assert defaults["minimum_request_interval_steps"] == 1
