"""Read-only replay of sealed v2 evidence; no policy, simulator or network imports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST_SHA = "9c8e376f83ca4dce162b452defd2d9c42ea6bc8e44478133951846c8068fbe2f"
CONFIG_SHA = "661ee2fbcd565ec2a04deda57d8021cc088f05d2a3cb915bc59e379486fbb4f1"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify_manifest(path: Path, base: Path | None = None) -> int:
    base = base or path.parent
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = (base / relative).resolve()
        assert target.is_relative_to(base.resolve()), relative
        assert sha(target) == digest, target
        count += 1
    return count


def recompute(root: Path) -> tuple[dict, list[dict]]:
    manifest_path = root / "experiment_manifest.json"
    assert sha(manifest_path) == MANIFEST_SHA
    manifest = read(manifest_path)
    config_path = REPO / "configs/rpc_remote_transport_replication_v2.json"
    assert sha(config_path) == CONFIG_SHA
    config = read(config_path)
    assert len(manifest["runs"]) == len(config["run_matrix"]) == 51
    for item in manifest["artifacts"]:
        assert sha(root / item["path"]) == item["sha256"]
    events, sources, failures = [], [], []
    successes = actual_reconnects = connections = verified_entries = 0
    for identity, entry in zip(config["run_matrix"], manifest["runs"], strict=True):
        assert identity["run_id"] == entry["run_id"]
        run_manifest = root / entry["path"]
        assert sha(run_manifest) == entry["sha256"]
        verified_entries += verify_manifest(run_manifest)
        run = run_manifest.parent
        for host in ("a", "b"):
            verify_manifest(
                run / f"host_{host}/raw/hashes/raw_manifest.sha256",
                run / f"host_{host}/raw",
            )
        client = run / "host_b/raw/client"
        receipt = read(client / "episode_receipt.json")
        for key in ("suite", "task_id", "initial_state_index", "seed"):
            assert receipt[key] == identity[key]
        successes += int(receipt["success"])
        if not receipt["success"]:
            failures.append({**identity, "end_reason": receipt["end_reason"]})
        actual_reconnects += receipt["rpc_telemetry"]["reconnections"]
        connections += receipt["rpc_telemetry"]["connections_established"]
        paths = [client / "rpc_telemetry.jsonl", client / "engine_telemetry.jsonl"]
        assert sha(paths[0]) == receipt["rpc_telemetry_sha256"]
        assert sha(paths[1]) == receipt["engine_telemetry_sha256"]
        if identity["condition"] != "periodic_pre_inference_disconnect":
            continue
        sources.extend(
            {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}
            for path in paths + [client / "episode_receipt.json"]
        )
        rpc, engine = rows(paths[0]), rows(paths[1])
        server = rows(run / "host_a/raw/server/executor_telemetry.jsonl")
        admitted = {x["request_id"]: x for x in server if x["event"] == "enqueued"}
        stop = next(x for x in engine if x["event"] == "engine_stop_requested")
        fault_count = 0
        for ordinal, fault in enumerate(rpc, 1):
            assert fault.get("global_request_ordinal", ordinal) == ordinal
            if ordinal % 7:
                continue
            assert fault["outcome"] == "transport_error"
            fault_count += 1
            later = rpc[ordinal:]
            next_attempt = later[0] if later else None
            # This rule admits successes AND observed errors/deadlines. It does
            # not select opportunities solely because recovery succeeded.
            observable = [
                x
                for x in later
                if x["finished_utc_unix_ns"] <= stop["utc_unix_ns"]
                and x["outcome"] != "cancelled"
            ]
            recovered = [x for x in observable if x["outcome"] == "success"]
            cancelled = next(
                (
                    x
                    for x in engine
                    if x["event"] == "inference_cancelled"
                    and x["request_ordinal"] == ordinal
                    and x["reset_or_stop"]
                ),
                None,
            )
            terminal = not observable
            if terminal:
                assert not later or (
                    len(later) == 1
                    and next_attempt["outcome"] == "cancelled"
                    and next_attempt["started_utc_unix_ns"]
                    < stop["utc_unix_ns"]
                    <= next_attempt["finished_utc_unix_ns"]
                    and cancelled
                )
                assert cancelled is not None  # Includes pre-RPC provider reset.
            before = next(
                x for x in reversed(rpc[: ordinal - 1]) if x["outcome"] == "success"
            )
            metadata = admitted.get(next_attempt["request_id"], {}) if later else {}
            events.append(
                {
                    "run_id": identity["run_id"],
                    "family": identity["family"],
                    "state": identity["initial_state_index"],
                    "fault_global_ordinal": ordinal,
                    "fault_request_id": fault["request_id"],
                    "fault_finished_utc_ns": fault["finished_utc_unix_ns"],
                    "episode_stop_utc_ns": stop["utc_unix_ns"],
                    "later_rpc_attempt_exists": bool(later),
                    "next_request_id": next_attempt["request_id"] if later else None,
                    "next_attempt_outcome": next_attempt["outcome"] if later else None,
                    "next_start_utc_ns": next_attempt["started_utc_unix_ns"]
                    if later
                    else None,
                    "next_finish_utc_ns": next_attempt["finished_utc_unix_ns"]
                    if later
                    else None,
                    "next_budget_s": next_attempt["deadline_s"] if later else None,
                    "connection_before": before["connection_id"],
                    "connection_after": metadata.get("connection_id"),
                    "usable_observed_response_opportunity": bool(observable),
                    "terminal_edge": terminal,
                    "later_success": bool(recovered),
                    "cancellation_phase": cancelled["phase"] if cancelled else None,
                    "classification": "terminal_edge"
                    if terminal
                    else "recovery_observable",
                    "terminal_reason": (
                        "stop_cancelled_after_rpc_admission"
                        if terminal and later
                        else "stop_cancelled_provider_reset_before_rpc_admission"
                        if terminal
                        else None
                    ),
                    "source_run_manifest": entry["path"],
                    "source_run_manifest_sha256": entry["sha256"],
                }
            )
        assert fault_count == receipt["rpc_telemetry"]["transport_errors"]
    opportunities = sum(x["usable_observed_response_opportunity"] for x in events)
    recovered = sum(x["later_success"] for x in events)
    terminal = [x for x in events if x["terminal_edge"]]
    return {
        "analysis_type": "POST_HOC_DESCRIPTIVE_ANALYSIS",
        "experiment_manifest_sha256": MANIFEST_SHA,
        "protocol_config_sha256": CONFIG_SHA,
        "formal_runs": len(manifest["runs"]),
        "task_successes": successes,
        "injected_disconnect_events": len(events),
        "connections_established_all_runs": connections,
        "actual_reconnections": actual_reconnects,
        "later_rpc_attempt_present_events": sum(
            x["later_rpc_attempt_exists"] for x in events
        ),
        "genuine_observed_response_opportunities": opportunities,
        "successful_recoveries_among_opportunities": recovered,
        "terminal_edge_events": len(terminal),
        "terminal_events": terminal,
        "task_failures": failures,
        "original_all_fault_event_recovery": {
            "numerator": recovered,
            "denominator": len(events),
            "gate": "PASS" if recovered == len(events) else "FAIL",
        },
        "opportunity_conditioned_recovery": {
            "numerator": recovered,
            "denominator": opportunities,
        },
        "definition": {
            "fault": "Each seventh admitted single-client inference ends in transport_error; all available global ordinals must match attempted order. Pre-inference faults precede server enqueue.",
            "later_request": "A later admitted TCP inference attempt in rpc_telemetry.jsonl. A provider-reset cancellation before TCP admission is not counted as an RPC request.",
            "opportunity": "At least one later non-cancelled RPC attempt finishes no later than engine_stop_requested. A completed error or deadline would remain eligible, not be excluded as failure.",
            "terminal_edge": "No later uncancelled attempt has an observed endpoint before stop: either no RPC admission, or the only later attempt spans stop and is cancelled with reset_or_stop=true.",
            "two_axes": "Later-request presence and terminal-edge censoring overlap for two events. Literal later-request presence is 99, not the 97 usable observed-response opportunities.",
        },
        "limitations": [
            "Post-hoc censoring is termination-dependent and may be informative; 97/97 is descriptive, not an unbiased recovery probability or a new gate.",
            "A cancelled attempt's hypothetical outcome if the episode had continued is unknown; no counterfactual guarantee is inferred.",
            "Client and engine timestamps are from the same host. No cross-host clock subtraction is needed for opportunity classification.",
        ],
        "formal_verdict_unchanged": "COMPLETED_PARTIAL_SUPPORT_HARD_GATE_FAIL",
        "statement": "The post-hoc opportunity-conditioned analysis helps explain the preregistered failure but does not revise the frozen gate or its NO-GO verdict.",
        "source_files": sources,
        "run_manifest_file_entries_verified": verified_entries,
    }, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=REPO / "artifacts/rpc_remote_transport_replication_v2",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.resolve().is_relative_to(args.evidence_root.resolve())
    result, events = recompute(args.evidence_root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "posthoc_recovery.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    with (args.output / "posthoc_events.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(events)
    print(json.dumps({k: v for k, v in result.items() if isinstance(v, int)}))


if __name__ == "__main__":
    main()
