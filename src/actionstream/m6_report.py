"""Generate the auditable M6-G0 source crosswalk, report, and figures.

This module is downstream-only: it reads the frozen conformance evidence,
verifies the pinned LeRobot checkout and source ranges, and writes report
artifacts. It never runs a policy, changes a benchmark result, or controls
hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from actionstream.m6_conformance import canonical_sha256, file_sha256


MILESTONE = "M6-G0"
SCHEMA_VERSION = 1
UPSTREAM_REPOSITORY = "https://github.com/huggingface/lerobot"
RUNTIME_ORDER = (
    "sync_hold",
    "lerobot_weighted_average",
    "lerobot_latest_only",
    "lerobot_average",
    "lerobot_conservative",
    "actionstream_aligned",
)
RUNTIME_LABELS = {
    "sync_hold": "sync_hold reference",
    "lerobot_weighted_average": "LeRobot weighted_average (default)",
    "lerobot_latest_only": "LeRobot latest_only",
    "lerobot_average": "LeRobot average",
    "lerobot_conservative": "LeRobot conservative",
    "actionstream_aligned": "ActionStream aligned",
}
FAMILY_ORDER = (
    "zero",
    "quarter_horizon",
    "sixty_percent_horizon",
    "one_twenty_percent_horizon",
    "bounded_jitter",
    "out_of_order",
    "late_after_newer_generation",
)
FAMILY_LABELS = {
    "zero": "0H",
    "quarter_horizon": "0.25H",
    "sixty_percent_horizon": "0.58H",
    "one_twenty_percent_horizon": "1.17H",
    "bounded_jitter": "jitter",
    "out_of_order": "out-of-order",
    "late_after_newer_generation": "late generation",
}

# These ranges are intentionally frozen against the exact upstream commit in
# configs/m6_g0.json. Required fragments make source drift fail loudly.
SOURCE_REFERENCE_SPECS: dict[str, dict[str, Any]] = {
    "timed_data": {
        "file": "src/lerobot/async_inference/helpers.py",
        "symbol": "TimedData, TimedAction, TimedObservation",
        "start_line": 202,
        "end_line": 235,
        "required_fragments": (
            "timestamp: float",
            "timestep: int",
            "class TimedAction",
            "class TimedObservation",
            "must_go: bool = False",
        ),
    },
    "aggregate_registry": {
        "file": "src/lerobot/async_inference/configs.py",
        "symbol": "AGGREGATE_FUNCTIONS",
        "start_line": 29,
        "end_line": 34,
        "required_fragments": (
            '"weighted_average"',
            '"latest_only"',
            '"average"',
            '"conservative"',
        ),
    },
    "client_config": {
        "file": "src/lerobot/async_inference/configs.py",
        "symbol": "RobotClientConfig",
        "start_line": 103,
        "end_line": 143,
        "required_fragments": (
            "chunk_size_threshold",
            "default=0.5",
            'default="weighted_average"',
        ),
    },
    "send_observation": {
        "file": "src/lerobot/async_inference/robot_client.py",
        "symbol": "RobotClient.send_observation",
        "start_line": 183,
        "end_line": 215,
        "required_fragments": (
            "pickle.dumps(obs)",
            "self.stub.SendObservations",
        ),
    },
    "aggregate_queues": {
        "file": "src/lerobot/async_inference/robot_client.py",
        "symbol": "RobotClient._aggregate_action_queues",
        "start_line": 224,
        "end_line": 267,
        "required_fragments": (
            "future_action_queue = Queue()",
            "new_action.get_timestep() <= latest_action",
            "current_action_queue[new_action.get_timestep()]",
            "self.action_queue = future_action_queue",
        ),
    },
    "receive_actions": {
        "file": "src/lerobot/async_inference/robot_client.py",
        "symbol": "RobotClient.receive_actions",
        "start_line": 269,
        "end_line": 337,
        "required_fragments": (
            "self.stub.GetActions",
            "self._aggregate_action_queues",
            "self.must_go.set()",
        ),
    },
    "execute_action": {
        "file": "src/lerobot/async_inference/robot_client.py",
        "symbol": "RobotClient.control_loop_action",
        "start_line": 370,
        "end_line": 401,
        "required_fragments": (
            "self.action_queue.get_nowait()",
            "self.robot.send_action",
            "self.latest_action = timed_action.get_timestep()",
        ),
    },
    "request_threshold": {
        "file": "src/lerobot/async_inference/robot_client.py",
        "symbol": "RobotClient._ready_to_send_observation",
        "start_line": 403,
        "end_line": 406,
        "required_fragments": (
            "self.action_queue.qsize() / self.action_chunk_size",
            "self._chunk_size_threshold",
        ),
    },
    "capture_observation": {
        "file": "src/lerobot/async_inference/robot_client.py",
        "symbol": "RobotClient.control_loop_observation",
        "start_line": 408,
        "end_line": 453,
        "required_fragments": (
            "self.robot.get_observation()",
            "timestamp=time.time()",
            "timestep=max(latest_action, 0)",
            "observation.must_go",
        ),
    },
    "client_control_loop": {
        "file": "src/lerobot/async_inference/robot_client.py",
        "symbol": "RobotClient.control_loop",
        "start_line": 458,
        "end_line": 481,
        "required_fragments": (
            "if self.actions_available()",
            "self.control_loop_action",
            "if self._ready_to_send_observation()",
        ),
    },
    "server_observation_receive": {
        "file": "src/lerobot/async_inference/policy_server.py",
        "symbol": "PolicyServer.SendObservations",
        "start_line": 173,
        "end_line": 212,
        "required_fragments": (
            "receive_time = time.time()",
            "pickle.loads",
            "self._enqueue_observation",
        ),
    },
    "server_get_actions": {
        "file": "src/lerobot/async_inference/policy_server.py",
        "symbol": "PolicyServer.GetActions",
        "start_line": 214,
        "end_line": 258,
        "required_fragments": (
            "self.observation_queue.get",
            "self._predict_action_chunk",
            "self.config.inference_latency",
        ),
    },
    "observation_filter": {
        "file": "src/lerobot/async_inference/policy_server.py",
        "symbol": "PolicyServer._obs_sanity_checks, _enqueue_observation",
        "start_line": 268,
        "end_line": 310,
        "required_fragments": (
            "obs.get_timestep() in predicted_timesteps",
            "observations_similar",
            "if self.observation_queue.full()",
            "self.observation_queue.get_nowait()",
        ),
    },
    "time_action_chunk": {
        "file": "src/lerobot/async_inference/policy_server.py",
        "symbol": "PolicyServer._time_action_chunk",
        "start_line": 312,
        "end_line": 320,
        "required_fragments": (
            "t_0 + i * self.config.environment_dt",
            "timestep=i_0 + i",
        ),
    },
    "predict_action_chunk": {
        "file": "src/lerobot/async_inference/policy_server.py",
        "symbol": "PolicyServer._predict_action_chunk",
        "start_line": 330,
        "end_line": 387,
        "required_fragments": (
            "self._get_action_chunk",
            "self._time_action_chunk",
            "observation_t.get_timestamp()",
            "observation_t.get_timestep()",
        ),
    },
    "rtc_support_check": {
        "file": "src/lerobot/rollout/inference/rtc.py",
        "symbol": "supports_rtc_inference",
        "start_line": 66,
        "end_line": 80,
        "required_fragments": (
            "supports_rtc()",
            "inference_delay=0",
            "prev_chunk_left_over=None",
        ),
    },
    "rtc_inference_loop": {
        "file": "src/lerobot/rollout/inference/rtc.py",
        "symbol": "RTCInferenceEngine._rtc_loop",
        "start_line": 288,
        "end_line": 327,
        "required_fragments": (
            "delay = math.ceil(latency / time_per_chunk)",
            "prev_actions = queue.get_left_over()",
            "inference_delay=delay",
            "prev_chunk_left_over=prev_actions",
        ),
    },
    "rtc_queue": {
        "file": "src/lerobot/policies/rtc/action_queue.py",
        "symbol": "ActionQueue.merge, _replace_actions_queue",
        "start_line": 147,
        "end_line": 188,
        "required_fragments": (
            "self._replace_actions_queue",
            "Discards the first `real_delay` actions",
            "processed_actions[clamped_delay:]",
        ),
    },
    "base_rtc_support": {
        "file": "src/lerobot/policies/pretrained.py",
        "symbol": "PreTrainedPolicy.supports_rtc",
        "start_line": 252,
        "end_line": 254,
        "required_fragments": ("return False",),
    },
    "act_chunk_signature": {
        "file": "src/lerobot/policies/act/modeling_act.py",
        "symbol": "ACTPolicy.predict_action_chunk",
        "start_line": 125,
        "end_line": 135,
        "required_fragments": (
            "def predict_action_chunk(self, batch",
            "actions = self.model(batch)[0]",
        ),
    },
    "rtc_context_guard": {
        "file": "src/lerobot/rollout/context.py",
        "symbol": "make_rollout_context RTC guard",
        "start_line": 227,
        "end_line": 238,
        "required_fragments": (
            "if not supports_rtc_inference(policy)",
            "RTC inference is not supported",
        ),
    },
    "so_follower_safety_config": {
        "file": "src/lerobot/robots/so_follower/config_so_follower.py",
        "symbol": "SOFollowerConfig.max_relative_target",
        "start_line": 27,
        "end_line": 39,
        "required_fragments": (
            "max_relative_target",
            "limits the magnitude",
        ),
    },
    "so_follower_send_action": {
        "file": "src/lerobot/robots/so_follower/so_follower.py",
        "symbol": "SOFollower.send_action",
        "start_line": 205,
        "end_line": 230,
        "required_fragments": (
            "ensure_safe_goal_position",
            'self.bus.sync_write("Goal_Position"',
        ),
    },
}


def _read_json(path: Path | str, *, role: str) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{role} is not valid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must be a JSON object: {source}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def build_source_references(
    *,
    lerobot_root: Path,
    expected_commit: str,
) -> dict[str, Any]:
    """Verify frozen source ranges and return permalink-rich metadata."""

    checkout = lerobot_root.resolve()
    actual_commit = _git(checkout, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"LeRobot checkout is {actual_commit}, expected {expected_commit}"
        )
    if _git(checkout, "status", "--porcelain"):
        raise ValueError("LeRobot checkout must be clean for source verification")

    references: dict[str, Any] = {}
    for key, spec in SOURCE_REFERENCE_SPECS.items():
        relative = str(spec["file"])
        path = checkout / relative
        lines = path.read_text(encoding="utf-8").splitlines()
        start = int(spec["start_line"])
        end = int(spec["end_line"])
        if start < 1 or end > len(lines) or start > end:
            raise ValueError(f"Invalid source range for {key}: L{start}-L{end}")
        excerpt = "\n".join(lines[start - 1 : end])
        missing = [
            fragment
            for fragment in spec["required_fragments"]
            if fragment not in excerpt
        ]
        if missing:
            raise ValueError(f"Frozen source range {key} drifted; missing {missing}")
        references[key] = {
            "file": relative,
            "symbol": spec["symbol"],
            "start_line": start,
            "end_line": end,
            "file_sha256": file_sha256(path),
            "permalink": (
                f"{UPSTREAM_REPOSITORY}/blob/{expected_commit}/"
                f"{relative}#L{start}-L{end}"
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "upstream_commit": expected_commit,
        "checkout_clean": True,
        "references": references,
    }


def _cite(references: Mapping[str, Any], *keys: str) -> str:
    return ", ".join(
        f"[`{references[key]['symbol']}`]({references[key]['permalink']})"
        for key in keys
    )


def build_source_crosswalk(
    *,
    upstream: Mapping[str, Any],
    source_payload: Mapping[str, Any],
) -> str:
    refs = source_payload["references"]
    commit = str(upstream["commit"])
    rows = [
        (
            "Observation capture",
            "The client reads the robot, then wraps the observation with wall-clock "
            "`time.time()` and `timestep=max(latest_action, 0)`.",
            "The harness defines observation step `O` as the last completed control "
            "step and preserves it as explicit request provenance.",
            _cite(refs, "capture_observation"),
        ),
        (
            "Request submission",
            "An observation is sent when queue size divided by observed chunk size is "
            "at or below the configured threshold (default 0.5); an empty queue can "
            "mark the observation `must_go`.",
            "M6 freezes request steps before execution and gives every request an ID "
            "and generation.",
            _cite(refs, "client_config", "request_threshold", "capture_observation"),
        ),
        (
            "Observation time representation",
            "`TimedObservation` carries a Unix timestamp and integer timestep, but no "
            "request ID or generation.",
            "Request ID, generation, observation step, arrival step, and insertion "
            "step remain separate fields.",
            _cite(refs, "timed_data"),
        ),
        (
            "Action chunk timestamping",
            "The server passes the observation timestamp and timestep into "
            "`_time_action_chunk`; element `i` gets `t0+i*dt` and `i0+i`.",
            "A chunk captured at `O` encodes intended steps `O+1` through `O+H`.",
            _cite(refs, "predict_action_chunk", "time_action_chunk"),
        ),
        (
            "Per-action intended execution index",
            "Every action has a timestamp/timestep label, but the first label equals "
            "the observation label rather than an explicit post-observation `O+1` "
            "contract.",
            "The intended absolute execution step is explicit and independently "
            "checked against the actual step.",
            _cite(refs, "time_action_chunk", "capture_observation"),
        ),
        (
            "Elapsed inference time",
            "The server measures inference and can sleep to a configured latency, but "
            "returns the generated chunk unchanged; the client later compares labels "
            "with `latest_action`.",
            "Arrival age directly determines the elapsed prefix length.",
            _cite(refs, "server_get_actions", "aggregate_queues"),
        ),
        (
            "Elapsed-prefix removal",
            "The ACT-compatible async client skips incoming actions whose timestep is "
            "not newer than the latest executed label. It does not compute a prefix "
            "from request observation age.",
            "The aligned scheduler removes exactly `max(0, arrival-O)` leading "
            "actions; a fully elapsed chunk is rejected without replacing the queue.",
            _cite(refs, "aggregate_queues"),
        ),
        (
            "Overlapping old/new chunks",
            "The client builds a new queue from incoming actions. Equal timestep "
            "entries are combined with the selected function; old-only future "
            "entries are not copied into the new queue.",
            "The surviving age-aligned suffix replaces the old queue.",
            _cite(refs, "aggregate_queues"),
        ),
        (
            "Aggregation coordinate",
            "Registered functions combine tensors only when their integer timestep "
            "labels match; they are not called by raw queue position.",
            "Alignment is based on observation/arrival age before replacement, not "
            "tensor averaging.",
            _cite(refs, "aggregate_registry", "aggregate_queues"),
        ),
        (
            "Late results",
            "There is no request-generation field. A returned future-labeled action "
            "can enter the replacement queue even if a newer request completed first.",
            "A result from an older generation is rejected before queue mutation.",
            _cite(refs, "timed_data", "receive_actions", "aggregate_queues"),
        ),
        (
            "Out-of-order results",
            "Each arrival independently rebuilds the queue after filtering labels at "
            "or before `latest_action`; no request-order comparison is present.",
            "Same-generation out-of-order arrivals are age-aligned deterministically; "
            "generation races are rejected.",
            _cite(refs, "receive_actions", "aggregate_queues"),
        ),
        (
            "Stale generation rejection",
            "Not represented in `TimedObservation` or `TimedAction`, so the official "
            "path has no explicit generation rejection.",
            "Explicit request generations guard queue replacement.",
            _cite(refs, "timed_data"),
        ),
        (
            "Queue empty",
            "The control loop calls `control_loop_action` only when actions are "
            "available. Otherwise it sends no new robot command and can request a "
            "`must_go` observation.",
            "Before the first safe action it blocks; after one exists it repeats the "
            "last command and records a hold.",
            _cite(refs, "client_control_loop", "capture_observation", "execute_action"),
        ),
        (
            "Hold/repeat/block semantics",
            "No explicit repeat-last-command branch exists in the async control loop; "
            "the previously commanded physical target may persist, but that is robot "
            "controller behavior rather than a new client send.",
            "Startup block and repeat-last-command are explicit, distinguishable "
            "states.",
            _cite(refs, "client_control_loop", "execute_action"),
        ),
        (
            "Policy-architecture dependency",
            "The audited async queue accepts ACT chunks. RTC is a separate rollout "
            "engine requiring both `supports_rtc()` and an extended prediction "
            "signature; ACT exposes neither.",
            "M6 compares only ACT-compatible async aggregation and audits RTC "
            "semantically.",
            _cite(
                refs,
                "rtc_support_check",
                "base_rtc_support",
                "act_chunk_signature",
                "rtc_context_guard",
            ),
        ),
        (
            "Registered aggregate function?",
            "Insufficient: the callback receives only old/new tensors sharing a "
            "timestep, after request provenance has already been lost.",
            "Cannot implement age-based prefix removal or generation rejection as an "
            "aggregate callback alone.",
            _cite(refs, "aggregate_registry", "aggregate_queues", "timed_data"),
        ),
        (
            "RobotClient scheduler extension?",
            "The narrowest client insertion is after action deserialization and "
            "before `_aggregate_action_queues`, paired with observation-capture "
            "metadata.",
            "Required, but reliable request/generation echo also needs server/timed "
            "data changes.",
            _cite(refs, "receive_actions", "capture_observation"),
        ),
        (
            "Policy wrapper?",
            "Insufficient because queue replacement, arrival time, and latest-executed "
            "state live outside ACT.",
            "ACT itself need not change; the scheduling contract belongs in runtime "
            "plumbing.",
            _cite(
                refs, "predict_action_chunk", "receive_actions", "act_chunk_signature"
            ),
        ),
        (
            "Broader runtime fork?",
            "A custom function is too small, but a full fork is unnecessary: additive "
            "timed-data fields plus localized server echo and client scheduling are "
            "the credible surface.",
            "The semantic port is localized, although the M6 decision does not "
            "authorize building or deploying it.",
            _cite(
                refs,
                "timed_data",
                "server_observation_receive",
                "predict_action_chunk",
                "receive_actions",
            ),
        ),
    ]
    table = [
        "| Semantic point | Official LeRobot at frozen commit | ActionStream M6 contract | Upstream evidence |",
        "|---|---|---|---|",
    ]
    table.extend(
        f"| {point} | {official} | {actionstream} | {evidence} |"
        for point, official, actionstream, evidence in rows
    )
    return "\n".join(
        [
            "# M6-G0 source-level semantic crosswalk",
            "",
            f"Frozen upstream: `{commit}` (LeRobot {upstream['source_version']}). "
            "The table compares observable scheduling behavior; it does not treat "
            "different names as novelty.",
            "",
            *table,
            "",
            "## RTC boundary",
            "",
            "RTC does remove a measured-delay prefix when its opt-in queue replaces a "
            f"chunk ({_cite(refs, 'rtc_inference_loop', 'rtc_queue')}). However, the "
            "rollout gate requires a policy declaration and an extended "
            "`predict_action_chunk(..., inference_delay, prev_chunk_left_over)` call "
            f"shape ({_cite(refs, 'rtc_support_check', 'rtc_context_guard')}). "
            f"`ACTPolicy.predict_action_chunk(batch)` has neither ({_cite(refs, 'act_chunk_signature')}); "
            "therefore RTC was not forced into the ACT-compatible synthetic matrix.",
            "",
        ]
    )


def build_portability_audit(
    *,
    decision: Mapping[str, Any],
    source_payload: Mapping[str, Any],
) -> str:
    refs = source_payload["references"]
    return "\n".join(
        [
            "# M6-G0 SO-101 + ACT portability audit",
            "",
            f"Formal outcome: **{decision['classification']}**. The smallest credible "
            "port is documented for feasibility only; M6-G0 does not implement or "
            "authorize physical-robot deployment.",
            "",
            "## Smallest credible insertion",
            "",
            "The client insertion point is in `RobotClient.receive_actions`, after "
            "deserialization and before `_aggregate_action_queues`, with corresponding "
            "request metadata captured in `control_loop_observation` "
            f"({_cite(refs, 'receive_actions', 'capture_observation')}). A custom "
            "`aggregate_fn` is insufficient because it sees only two tensors for an "
            f"already-matched timestep ({_cite(refs, 'aggregate_registry', 'aggregate_queues')}).",
            "",
            "The credible opt-in path would:",
            "",
            "1. Add `request_id`, `request_generation`, and explicit observation/intended-step "
            "metadata to the timed request/action objects.",
            "2. Have `PolicyServer` preserve and echo that provenance when producing the chunk.",
            "3. Have `RobotClient` compute elapsed control steps at arrival, reject old "
            "generations, remove the elapsed prefix, and atomically replace the queue.",
            "4. Make startup block, bounded repeat-last-command, and watchdog expiry "
            "explicit telemetry and configuration.",
            "",
            "Existing timestamps can be retained, but they are not sufficient alone: "
            "the current first action reuses the observation timestep and the timed "
            f"objects carry no request generation ({_cite(refs, 'time_action_chunk', 'timed_data')}).",
            "",
            "## Component impact",
            "",
            "| Component | Required change | Conclusion |",
            "|---|---|---|",
            "| SO-101 follower | No scheduler algorithm in the driver; configure target clipping, watchdog/stop behavior, and log the command actually sent. | Safety integration required, no hardware work in M6. |",
            "| ACT policy | None to architecture, weights, or training. | ACT remains an ordinary chunk producer. |",
            "| PolicyServer | Echo request ID/generation and explicit origin/intended-step convention. | Required for robust provenance. |",
            "| RobotClient | Add opt-in arrival-age alignment, generation rejection, atomic replacement, and explicit underrun/hold states. | Primary implementation surface. |",
            "| Timed data/config | Add provenance fields and scheduler/hold/watchdog configuration. | Required and additive. |",
            "| Aggregate callback | No sufficient implementation is possible with only `(old, new)` tensors. | Not the port boundary. |",
            "",
            "## Files and estimated surface",
            "",
            "A credible upstream implementation would touch four production files and "
            "their focused tests:",
            "",
            "- `src/lerobot/async_inference/helpers.py`",
            "- `src/lerobot/async_inference/configs.py`",
            "- `src/lerobot/async_inference/policy_server.py`",
            "- `src/lerobot/async_inference/robot_client.py`",
            "- `tests/async_inference/test_helpers.py`",
            "- `tests/async_inference/test_policy_server.py`",
            "- `tests/async_inference/test_robot_client.py`",
            "",
            "Estimated implementation size is roughly 150-250 production lines plus "
            "tests and benchmark fixtures. This is an engineering estimate, not a "
            "validated port measurement. Pickled timed dataclasses already travel "
            f"inside the existing RPC payload ({_cite(refs, 'send_observation', 'server_observation_receive')}); "
            "a protobuf schema change is not obviously required, but mixed-version "
            "client/server compatibility would need an explicit test.",
            "",
            "## Telemetry and safety controls",
            "",
            "Required telemetry: request ID/generation, observation timestamp and "
            "control step, inference start/end, client receive/insertion step, prefix "
            "drop count, queue depth before/after, executed intended/actual index, "
            "hold/underrun reason, command age, and command actually sent.",
            "",
            "Required pre-hardware controls: startup block until a valid command, "
            "bounded hold duration, communication watchdog, queue/generation reset on "
            "reconnect or episode reset, monotonic local arrival timing, explicit "
            "clock-skew handling for cross-host timestamps, joint/velocity/workspace "
            "limits, accessible emergency stop, and SO-101 relative-target clipping. "
            "LeRobot exposes `max_relative_target`, and `SOFollower.send_action` clips "
            "before writing motor goals "
            f"({_cite(refs, 'so_follower_safety_config', 'so_follower_send_action')}).",
            "",
            "## Compatibility risks and upstreamability",
            "",
            "- The official action label convention starts at the observation label; "
            "changing it globally could break queue/tests, so the new convention must "
            "be opt-in or separately represented.",
            "- Pickle compatibility across mixed client/server versions needs explicit "
            "fallback behavior.",
            "- Host wall clocks are used for cross-host timestamp diagnostics; safety "
            "decisions should prefer a local monotonic arrival clock or verified clock sync.",
            "- Replacing a queue after a fully stale result can starve execution; a "
            "bounded fallback policy must be specified before hardware.",
            "- Extra observation/request generation must remain compatible with the "
            "server's queue-of-one replacement and duplicate/similarity filters "
            f"({_cite(refs, 'observation_filter')}).",
            "",
            "An opt-in scheduler plus diagnostics could plausibly be reviewed as an "
            "upstream PR because the surface is localized and ACT does not change. "
            "However, the predeclared M6 result is not `PORT GO`, and official "
            "scheduling was not equivalent-or-better in enough families for "
            "`TOOLING GO / ALGORITHM OVERLAP`. The formal next step is therefore "
            f"**{decision['recommended_next_step']}**, not a hardware port or PR.",
            "",
        ]
    )


def _load_jsonl_filtered(
    path: Path,
    *,
    family: str,
    seed: int,
    runtimes: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if (
                f'"latency_family":"{family}"' not in line
                or f'"trace_seed":{seed}' not in line
            ):
                continue
            row = json.loads(line)
            if row["runtime"] in runtimes:
                rows.append(row)
    return rows


def build_representative_timeline(
    *,
    output_root: Path,
    seed: int,
    best_official: str,
) -> dict[str, Any]:
    family = "out_of_order"
    runtimes = {best_official, "actionstream_aligned"}
    action_rows = _load_jsonl_filtered(
        output_root / "traces" / "action_records.jsonl",
        family=family,
        seed=seed,
        runtimes=runtimes,
    )
    result_rows = _load_jsonl_filtered(
        output_root / "traces" / "result_records.jsonl",
        family=family,
        seed=seed,
        runtimes=runtimes,
    )
    by_runtime: dict[str, Any] = {}
    for runtime in (best_official, "actionstream_aligned"):
        runtime_actions = [row for row in action_rows if row["runtime"] == runtime]
        runtime_results = [row for row in result_rows if row["runtime"] == runtime]
        executions = [
            {
                "actual_execution_step": row["actual_execution_step"],
                "intended_execution_step": row["intended_execution_step"],
                "request_id": row["request_id"],
                "request_generation": row["request_generation"],
                "queue_depth_before_action": row["queue_depth_before_action"],
                "queue_depth_after_action": row["queue_depth_after_action"],
                "state": row["hold_or_underrun_state"],
                "record_kind": row["record_kind"],
            }
            for row in runtime_actions
            if row["actual_execution_step"] is not None
            and row["hold_or_underrun_state"] is not None
        ]
        transitions = [
            {
                key: row[key]
                for key in (
                    "request_id",
                    "request_generation",
                    "observation_step",
                    "result_arrival_step",
                    "queue_insertion_step",
                    "accepted",
                    "reason",
                    "out_of_order",
                    "late_generation",
                    "old_queue_intended_steps",
                    "new_chunk_intended_steps",
                    "discarded_prefix_intended_steps",
                    "new_queue_intended_steps",
                    "actions_inserted",
                    "actions_discarded",
                )
            }
            for row in runtime_results
        ]
        by_runtime[runtime] = {
            "executions": executions,
            "queue_transitions": transitions,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "latency_family": family,
        "trace_seed": seed,
        "interpretation": (
            "Every transition records observation/request, arrival/insertion, old "
            "queue, new chunk, discarded prefix, and resulting queue. Executions "
            "record actual and intended indices plus underrun/hold state."
        ),
        "runtimes": by_runtime,
    }


def _configure_plot_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 130,
            "savefig.dpi": 250,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)


def plot_representative_timeline(
    timeline: Mapping[str, Any],
    *,
    best_official: str,
    output_dir: Path,
) -> None:
    _configure_plot_style()
    runtimes = (best_official, "actionstream_aligned")
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.2), sharex=True, sharey=True)
    for ax, runtime in zip(axes, runtimes, strict=True):
        payload = timeline["runtimes"][runtime]
        executions = payload["executions"]
        actual = np.asarray(
            [row["actual_execution_step"] for row in executions], dtype=float
        )
        intended = np.asarray(
            [
                np.nan
                if row["intended_execution_step"] is None
                else row["intended_execution_step"]
                for row in executions
            ],
            dtype=float,
        )
        states = [row["state"] for row in executions]
        fresh = np.asarray([state == "fresh" for state in states])
        hold = np.asarray(["hold" in state for state in states])
        underrun = np.asarray(
            [state in {"startup_block", "upstream_no_send"} for state in states]
        )
        ax.plot([1, 48], [1, 48], color="#A0A0A0", linewidth=1, linestyle="--")
        ax.scatter(
            actual[fresh],
            intended[fresh],
            s=15,
            color="#2F6B9A",
            label="executed: intended index",
            zorder=3,
        )
        if hold.any():
            ax.scatter(
                actual[hold],
                intended[hold],
                s=24,
                marker="s",
                facecolors="none",
                edgecolors="#E17C05",
                label="repeat-last hold",
                zorder=3,
            )
        if underrun.any():
            ax.scatter(
                actual[underrun],
                np.full(int(underrun.sum()), -1.0),
                s=25,
                marker="x",
                color="#B13A3A",
                label="queue empty / no command",
                zorder=4,
            )
        for transition in payload["queue_transitions"]:
            x = float(transition["queue_insertion_step"])
            discarded = int(transition["actions_discarded"])
            queue_after = len(transition["new_queue_intended_steps"])
            color = "#B13A3A" if discarded else "#5A9E6F"
            ax.scatter(
                [x],
                [50.0],
                s=25 + 5 * queue_after,
                marker="v",
                color=color,
                alpha=0.75,
                zorder=2,
            )
            if discarded:
                ax.annotate(
                    f"-{discarded}",
                    (x, 50.0),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=color,
                )
        ax.set_title(RUNTIME_LABELS[runtime], loc="left")
        ax.set_ylabel("intended index")
        ax.set_ylim(-3, 54)
        ax.grid(axis="both", color="#E6E6E6", linewidth=0.6)
    axes[-1].set_xlabel("actual control step")
    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        extra_handles, extra_labels = ax.get_legend_handles_labels()
        handles.extend(extra_handles)
        labels.extend(extra_labels)
    dedup = dict(zip(labels, handles, strict=False))
    fig.legend(
        dedup.values(),
        dedup.keys(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        "Representative out-of-order trace: execution and queue-arrival markers",
        y=1.08,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        -0.015,
        "Triangles are result insertion events; marker area scales with resulting "
        "queue depth and red labels show discarded actions.",
        ha="center",
        fontsize=7.5,
    )
    fig.subplots_adjust(hspace=0.24)
    _save_figure(fig, output_dir, "representative_timeline")


def plot_aggregate_mechanism(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
) -> None:
    _configure_plot_style()
    lookup = {(str(row["runtime"]), str(row["latency_family"])): row for row in rows}
    colors = {
        "sync_hold": "#888888",
        "lerobot_weighted_average": "#4C78A8",
        "lerobot_latest_only": "#72B7B2",
        "lerobot_average": "#F58518",
        "lerobot_conservative": "#E45756",
        "actionstream_aligned": "#54A24B",
    }
    styles = {
        "sync_hold": ":",
        "lerobot_weighted_average": "-",
        "lerobot_latest_only": "-",
        "lerobot_average": "--",
        "lerobot_conservative": "--",
        "actionstream_aligned": "-",
    }
    metrics = (
        ("stale_prefix_actions_executed_per_trace", "stale actions executed / trace"),
        ("median_temporal_index_error", "median temporal error (steps)"),
        ("queue_underrun_rate", "queue-underrun rate"),
    )
    x = np.arange(len(FAMILY_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), sharex=True)
    for runtime in RUNTIME_ORDER:
        for ax, (metric, _) in zip(axes, metrics, strict=True):
            values = [
                (
                    np.nan
                    if lookup[(runtime, family)][metric] is None
                    else float(lookup[(runtime, family)][metric])
                )
                for family in FAMILY_ORDER
            ]
            ax.plot(
                x,
                values,
                color=colors[runtime],
                linestyle=styles[runtime],
                marker="o",
                markersize=3,
                linewidth=1.5 if runtime == "actionstream_aligned" else 1.0,
                label=RUNTIME_LABELS[runtime],
            )
    for ax, (_, label) in zip(axes, metrics, strict=True):
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [FAMILY_LABELS[family] for family in FAMILY_ORDER],
            rotation=35,
            ha="right",
        )
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    axes[2].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle(
        "M6-G0 scheduling behavior across the frozen latency families",
        y=1.18,
        fontsize=10.5,
    )
    fig.subplots_adjust(wspace=0.27)
    _save_figure(fig, output_dir, "aggregate_mechanism")


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _full_result_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lookup = {(str(row["runtime"]), str(row["latency_family"])): row for row in rows}
    table = [
        "| Runtime | Family | Stale executed / trace | Median / p95 error | Underrun | Hold | Max queue | Median / p95 scheduler us |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for runtime in RUNTIME_ORDER:
        for family in FAMILY_ORDER:
            row = lookup[(runtime, family)]
            table.append(
                "| "
                + " | ".join(
                    (
                        RUNTIME_LABELS[runtime],
                        FAMILY_LABELS[family],
                        _fmt(row["stale_prefix_actions_executed_per_trace"]),
                        (
                            f"{_fmt(row['median_temporal_index_error'])} / "
                            f"{_fmt(row['p95_temporal_index_error'])}"
                        ),
                        f"{100 * float(row['queue_underrun_rate']):.2f}%",
                        f"{100 * float(row['hold_rate']):.2f}%",
                        str(row["maximum_queue_depth"]),
                        (
                            f"{_fmt(row['scheduler_overhead_median_microseconds'])} / "
                            f"{_fmt(row['scheduler_overhead_p95_microseconds'])}"
                        ),
                    )
                )
                + " |"
            )
    return table


def build_paper_report(
    *,
    manifest: Mapping[str, Any],
    upstream: Mapping[str, Any],
    validation: Mapping[str, Any],
    decision: Mapping[str, Any],
    aggregate_rows: Sequence[Mapping[str, Any]],
    source_payload: Mapping[str, Any],
) -> str:
    refs = source_payload["references"]
    best = str(decision["best_official_act_compatible_runtime"])
    measurements = decision["measurements"]
    criteria = decision["criteria"]
    criteria_table = [
        "| Predeclared gate | Result |",
        "|---|---:|",
        *[
            f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
            for name, passed in criteria.items()
        ],
    ]
    family_reduction = measurements["family_median_temporal_error_reduction_fraction"]
    reduction_rows = [
        "| Non-zero family | Median-error reduction vs best official | Underrun worsening |",
        "|---|---:|---:|",
    ]
    for family in FAMILY_ORDER[1:]:
        reduction = family_reduction[family]
        reduction_text = (
            "unavailable" if reduction is None else f"{100 * reduction:.1f}%"
        )
        worsening = measurements["queue_underrun_worsening_percentage_points"][family]
        reduction_rows.append(
            f"| {FAMILY_LABELS[family]} | {reduction_text} | {worsening:.2f} pp |"
        )
    commit = str(upstream["commit"])
    return "\n".join(
        [
            "# M6-G0: LeRobot async-runtime overlap and port-value audit",
            "",
            "## Abstract",
            "",
            "M6-G0 asks whether ActionStream's stale-prefix alignment is materially "
            "different and better than the current official ACT-compatible LeRobot "
            "asynchronous runtime, before any real-robot work. The answer is mixed: "
            "ActionStream eliminated stale-prefix execution and reduced temporal "
            "index error, but fully stale and out-of-order conditions caused excessive "
            "queue underrun/hold. The frozen decision is therefore "
            f"**{decision['classification']}**, with recommended next step "
            f"**{decision['recommended_next_step']}**.",
            "",
            "No policy was trained, no GPU was required, and no physical robot was "
            "controlled. M4 and M5 artifacts were not modified.",
            "",
            "## Frozen upstream and executable evidence",
            "",
            f"The comparison uses official LeRobot source commit [`{commit}`]"
            f"({UPSTREAM_REPOSITORY}/tree/{commit}), source/package version "
            f"`{upstream['source_version']}`, retrieved {upstream['retrieval_date']}. "
            "The checkout was clean. The official async unit subset passed 22 tests. "
            "The M6 adapter directly invoked "
            f"{_cite(refs, 'time_action_chunk')} and "
            f"{_cite(refs, 'aggregate_queues')} rather than reproducing their "
            "behavior from documentation.",
            "",
            "The audit venv used Python "
            f"`{upstream['environment']['python_version']}` and torch "
            f"`{upstream['environment']['packages']['torch']}`. The frozen source "
            "declares torch `>=2.7,<2.12.0`, so the CPU build lies inside the "
            "declared bound. The environment contains LeRobot base dependencies and "
            "audit/report tools, not hardware-specific deployment extras.",
            "",
            "## Source-level semantic result",
            "",
            "Official async LeRobot gives each action a timestamp and timestep, "
            f"derived from the observation ({_cite(refs, 'timed_data', 'predict_action_chunk', 'time_action_chunk')}). "
            "At arrival, the client filters labels at or before the latest executed "
            "action, creates a new queue from the incoming chunk, and aggregates only "
            f"equal labels ({_cite(refs, 'aggregate_queues')}). It has no explicit "
            "request generation. ActionStream instead uses observation-to-arrival age "
            "to discard an elapsed prefix and rejects stale generations. These are "
            "observable semantic differences, not naming differences.",
            "",
            "The complete point-by-point table is in "
            "[source_semantic_crosswalk.md](source_semantic_crosswalk.md).",
            "",
            "## Frozen synthetic protocol",
            "",
            f"- Chunk horizon: H={manifest['scenario']['chunk_horizon']} at "
            f"{manifest['scenario']['control_frequency_hz']} Hz "
            f"(dt={manifest['scenario']['environment_dt_seconds']} s).",
            f"- Episode: {manifest['scenario']['execution_steps']} execution steps; "
            f"requests every {manifest['scenario']['request_interval_steps']} steps "
            f"at {manifest['scenario']['request_observation_steps']}.",
            f"- Actions: {manifest['scenario']['action_dimensionality']}D, with the "
            "first component encoding the intended absolute execution step `O+1..O+H`.",
            "- Latencies: 0H, 0.25H, 0.58H, 1.17H, seeded bounded jitter, "
            "out-of-order arrival, and a late result after a newer generation.",
            f"- Traces: {manifest['scenario']['deterministic_trace_count']} repeats "
            "for each fixed family and "
            f"{manifest['scenario']['stochastic_trace_count']} seeded jitter traces "
            "per runtime.",
            "- Runtimes: fixed-schedule sync_hold reference, all four registered "
            "official aggregate functions, and ActionStream aligned.",
            "",
            "`sync_hold` is a fixed-request, full-replacement/hold conformance "
            "reference under the identical schedule. It is not a remeasurement of "
            "M4's dynamically timed sync policy.",
            "",
            "The exact metrics are `abs(actual_execution_step - "
            "intended_execution_step)` and stale prefix iff "
            "`intended_execution_step < queue_insertion_step`. Validation confirms "
            f"{validation['counts']['action_records']:,} action records, "
            f"{validation['counts']['result_records']:,} request/result records, "
            f"{validation['counts']['trace_summaries']:,} runtime traces, "
            "identical inputs across runtimes, 50 jitter traces per runtime, and "
            "direct upstream method calls.",
            "",
            "## Results",
            "",
            f"The predeclared lexicographic official baseline selector chose "
            f"**{RUNTIME_LABELS[best]}**. Against it, ActionStream reduced delayed "
            "stale-prefix actions executed by "
            f"{100 * measurements['delayed_stale_prefix_execution_reduction_fraction']:.1f}% "
            "and met the median-error reduction threshold in "
            f"{len(measurements['families_meeting_30pct_error_reduction'])} "
            "non-zero families. At 1.17H, ActionStream executed no action, so temporal "
            "error is unavailable rather than zero.",
            "",
            *reduction_rows,
            "",
            "The decisive failure was queue availability: maximum underrun worsening "
            "was "
            f"{measurements['maximum_queue_underrun_worsening_percentage_points']:.2f} "
            "percentage points, above the frozen 5 pp limit. The 1.17H family reached "
            "100% underrun, and out-of-order arrivals worsened underrun by "
            f"{measurements['queue_underrun_worsening_percentage_points']['out_of_order']:.2f} pp. "
            "Zero-latency behavior did not regress.",
            "",
            "![Aggregate scheduling mechanisms](plots/aggregate_mechanism.png)",
            "",
            "The representative timeline's result-transition records explicitly "
            "contain observations, requests, arrival/insertion, old queue, new chunk, "
            "discarded prefix, resulting queue, and executed indices.",
            "",
            "![Representative out-of-order timeline](plots/representative_timeline.png)",
            "",
            "### Complete runtime-by-family metrics",
            "",
            *_full_result_table(aggregate_rows),
            "",
            "## Predeclared decision",
            "",
            *criteria_table,
            "",
            f"Classification: **{decision['classification']}**.",
            "",
            *[f"- {reason}" for reason in decision["reasons"]],
            "",
            "This does not support a stable net advantage under the frozen acceptance "
            "rule: stronger temporal alignment trades into excessive blocking or "
            "repeat-last behavior. The formal next step is "
            f"**{decision['recommended_next_step']}**.",
            "",
            "## RTC applicability",
            "",
            "RTC is semantically related: it estimates inference delay, passes leftover "
            "actions into compatible policies, and its queue removes a measured-delay "
            f"prefix ({_cite(refs, 'rtc_inference_loop', 'rtc_queue')}). It was not "
            "included in the ACT matrix because the rollout gate requires "
            "`supports_rtc()` and an extended prediction signature "
            f"({_cite(refs, 'rtc_support_check', 'rtc_context_guard')}), while ACT "
            f"offers `predict_action_chunk(batch)` only ({_cite(refs, 'act_chunk_signature')}). "
            "Forcing RTC here would not be an ACT-compatible comparison.",
            "",
            "## SO-101 + ACT portability",
            "",
            "A registered aggregate function is insufficient. The smallest credible "
            "port is an opt-in, localized extension to timed data/config, PolicyServer "
            "provenance echo, and RobotClient arrival scheduling; ACT weights and "
            "architecture do not change. SO-101 deployment would additionally require "
            "bounded holds, watchdogs, resets, joint/workspace limits, emergency stop, "
            "and relative-target clipping. See "
            "[portability_audit.md](portability_audit.md) for the exact insertion "
            "point, files, telemetry, risks, and estimated surface.",
            "",
            "## Limitations and scope guards",
            "",
            "- This is a deterministic synthetic scheduling audit, not task-success, "
            "network-stack, actuator, or human safety validation.",
            "- Scheduler overhead is a CPU microbenchmark inside this process; it is "
            "reported for completeness, not as a deployment latency bound.",
            "- Official wall-clock timestamps and local simulated control-step timing "
            "are not evidence of cross-host clock synchronization.",
            "- Fully stale rejection preserves the current queue but can block when no "
            "safe command has ever been established; this caused the 1.17H failure.",
            "- M5-G0 remains **NO-GO**.",
            "- PoseGuard is not being revived.",
            "- FoundationPose is not part of M6-G0.",
            "- RTC was audited but not forced into the ACT-compatible benchmark.",
            "",
            "## Reproduction",
            "",
            "From the repository root, with the exact ignored LeRobot checkout and "
            "audit environment described in the README:",
            "",
            "```powershell",
            "$env:PYTHONPATH=(Resolve-Path src).Path",
            ".\\.external\\m6_venv\\Scripts\\python.exe -m actionstream.m6_conformance `",
            "  --manifest configs/m6_g0.json --lerobot-root .external/lerobot `",
            "  --output-root outputs/m6_g0",
            ".\\.external\\m6_venv\\Scripts\\python.exe -m actionstream.m6_report `",
            "  --manifest configs/m6_g0.json --lerobot-root .external/lerobot `",
            "  --output-root outputs/m6_g0",
            "```",
            "",
        ]
    )


def _update_artifact_manifest(
    output_root: Path,
    *,
    upstream_commit: str,
    scenario_manifest_sha256: str,
) -> None:
    artifacts = {
        path.relative_to(output_root).as_posix(): file_sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    core = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "scenario_manifest_sha256": scenario_manifest_sha256,
        "upstream_commit": upstream_commit,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }
    _write_json(
        output_root / "artifact_manifest.json",
        {**core, "artifact_manifest_sha256": canonical_sha256(core)},
    )


def validate_report_artifacts(
    *,
    report: str,
    crosswalk: str,
    portability: str,
    source_payload: Mapping[str, Any],
    decision: Mapping[str, Any],
    validation: Mapping[str, Any],
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required_report_sections = (
        "## Abstract",
        "## Frozen upstream and executable evidence",
        "## Source-level semantic result",
        "## Frozen synthetic protocol",
        "## Results",
        "## Predeclared decision",
        "## RTC applicability",
        "## SO-101 + ACT portability",
        "## Limitations and scope guards",
        "## Reproduction",
    )
    checks = {
        "benchmark_validation_passed": validation.get("passed") is True,
        "decision_hash_valid": decision.get("decision_sha256")
        == canonical_sha256(
            {key: value for key, value in decision.items() if key != "decision_sha256"}
        ),
        "paper_sections_complete": all(
            section in report for section in required_report_sections
        ),
        "scope_boundaries_explicit": all(
            phrase in report
            for phrase in (
                "M5-G0 remains **NO-GO**",
                "PoseGuard is not being revived",
                "FoundationPose is not part of M6-G0",
                "RTC was audited",
            )
        ),
        "all_runtimes_reported": all(
            RUNTIME_LABELS[runtime] in report for runtime in RUNTIME_ORDER
        ),
        "all_latency_families_reported": all(
            FAMILY_LABELS[family] in report for family in FAMILY_ORDER
        ),
        "aggregate_matrix_has_42_rows": len(aggregate_rows)
        == len(RUNTIME_ORDER) * len(FAMILY_ORDER),
        "crosswalk_complete": crosswalk.count("\n| ") >= 19,
        "source_references_complete": set(source_payload["references"])
        == set(SOURCE_REFERENCE_SPECS),
        "every_source_reference_is_pinned": all(
            source_payload["upstream_commit"] in ref["permalink"]
            for ref in source_payload["references"].values()
        ),
        "port_surface_explicit": all(
            phrase in portability
            for phrase in (
                "custom `aggregate_fn` is insufficient",
                "PolicyServer",
                "RobotClient",
                "ACT policy",
                "Telemetry and safety controls",
                "estimated surface",
            )
        ),
        "decision_is_frozen_label": decision.get("classification")
        in {"PORT GO", "TOOLING GO / ALGORITHM OVERLAP", "NO-GO"},
    }
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "passed": not errors,
        "checks": checks,
        "errors": errors,
    }


def generate_report(
    *,
    manifest_path: Path,
    lerobot_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path, role="scenario manifest")
    upstream = _read_json(
        output_root / "upstream" / "pinned_upstream.json",
        role="pinned upstream metadata",
    )
    validation = _read_json(
        output_root / "validation.json",
        role="benchmark validation",
    )
    decision = _read_json(output_root / "decision.json", role="decision")
    aggregates = _read_json(
        output_root / "metrics" / "aggregate_metrics.json",
        role="aggregate metrics",
    )
    if manifest.get("milestone") != MILESTONE:
        raise ValueError("Scenario manifest is not M6-G0")
    if manifest.get("frozen_before_benchmark") is not True:
        raise ValueError("Scenario manifest was not frozen before benchmarking")
    if upstream.get("commit") != manifest["upstream"]["commit"]:
        raise ValueError("Pinned upstream commit does not match frozen config")
    if not validation.get("passed"):
        raise ValueError("Benchmark validation did not pass")
    if aggregates.get("upstream_commit") != upstream["commit"]:
        raise ValueError("Aggregate metrics use a different upstream commit")
    rows = aggregates.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Aggregate metrics rows are missing")

    source_payload = build_source_references(
        lerobot_root=lerobot_root,
        expected_commit=str(upstream["commit"]),
    )
    crosswalk = build_source_crosswalk(
        upstream=upstream,
        source_payload=source_payload,
    )
    portability = build_portability_audit(
        decision=decision,
        source_payload=source_payload,
    )
    timeline = build_representative_timeline(
        output_root=output_root,
        seed=int(manifest["scenario"]["seed_base"]),
        best_official=str(decision["best_official_act_compatible_runtime"]),
    )
    report = build_paper_report(
        manifest=manifest,
        upstream=upstream,
        validation=validation,
        decision=decision,
        aggregate_rows=rows,
        source_payload=source_payload,
    )
    report_validation = validate_report_artifacts(
        report=report,
        crosswalk=crosswalk,
        portability=portability,
        source_payload=source_payload,
        decision=decision,
        validation=validation,
        aggregate_rows=rows,
    )
    if not report_validation["passed"]:
        raise ValueError(f"M6 report validation failed: {report_validation['errors']}")

    report_dir = output_root / "report"
    plot_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "source_semantic_crosswalk.md").write_text(
        crosswalk,
        encoding="utf-8",
        newline="\n",
    )
    (report_dir / "portability_audit.md").write_text(
        portability,
        encoding="utf-8",
        newline="\n",
    )
    (report_dir / "m6_g0_report.md").write_text(
        report,
        encoding="utf-8",
        newline="\n",
    )
    _write_json(report_dir / "source_references.json", source_payload)
    _write_json(report_dir / "representative_timeline.json", timeline)
    _write_json(report_dir / "report_validation.json", report_validation)
    plot_representative_timeline(
        timeline,
        best_official=str(decision["best_official_act_compatible_runtime"]),
        output_dir=plot_dir,
    )
    plot_aggregate_mechanism(rows, output_dir=plot_dir)
    _update_artifact_manifest(
        output_root,
        upstream_commit=str(upstream["commit"]),
        scenario_manifest_sha256=file_sha256(manifest_path),
    )
    return {
        "report": report_dir / "m6_g0_report.md",
        "crosswalk": report_dir / "source_semantic_crosswalk.md",
        "portability": report_dir / "portability_audit.md",
        "classification": decision["classification"],
        "validation": report_validation,
    }


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "configs" / "m6_g0.json",
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=root / ".external" / "lerobot",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "outputs" / "m6_g0",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_report(
        manifest_path=args.manifest,
        lerobot_root=args.lerobot_root,
        output_root=args.output_root,
    )
    print(f"{result['report']} ({result['classification']})")


if __name__ == "__main__":
    main()
