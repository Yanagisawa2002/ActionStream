"""Independent stdlib replay of retained async journals and acceptance gates.

Does not import the controller, engine, learned verifier or production scorer.
Full RGB/state bytes are covered by the separately hashed simulator audit;
--bundle additionally verifies every external file in the downloaded bundle.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from record_io import archive_bytes, open_archive

HERE = Path(__file__).resolve().parent
RUN = "budget-acceptance-v2/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def quantile(values, q):
    ordered = sorted(values)
    x = (len(ordered) - 1) * q
    low = int(x)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (
        x - low
    )


def same(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (
        actual,
        expected,
    )


def replay_episode(read, lines, prefix, protocol, fault):
    outcome = read(prefix + "outcome.json")
    score = read(prefix + "independent_score.json")
    events = lines(prefix + "runtime.jsonl")
    dispatch = [e for e in events if e["event"] == "dispatch"]
    checks = [e for e in events if e["event"] in ("verification", "stale_observation")]
    facts, bindings, actions = [
        read(prefix + n)
        for n in ("private_truth.json", "observation_bindings.json", "actions.json")
    ]
    owners = [e for e in events if e["event"] == "async_chunk"]
    assert owners and all(e["inference_owner"] == "spawned_process" for e in owners)
    pids = {e["inference_pid"] for e in owners}
    assert len(pids) == 1 and all(type(pid) is int and pid > 0 for pid in pids)
    config = outcome["config"]
    for key, value in protocol["config"].items():
        assert config[key] == value, key
    assert config["max_attempts"] == (2 if fault else 1)
    n = outcome["control_steps"] + outcome["post_stop_controls"]
    assert len(dispatch) == len(actions) == n
    assert len(facts) == len(bindings) == n + 1
    assert len({b["observation_id"] for b in bindings}) == len(bindings)
    assert all(
        a["captured_monotonic"] < b["captured_monotonic"]
        for a, b in zip(bindings, bindings[1:])
    )
    engine = lines(prefix + "engine.jsonl")
    assert all(
        e["thread"] == "ActionStreamInference"
        for e in engine
        if e["event"] == "inference_started"
    )
    assert all(
        e["thread"] == "MainThread" for e in engine if e["event"] == "action_dequeued"
    )
    assert [e["control"] for e in dispatch] == list(range(n))
    assert [e["control"] for e in checks] == list(range(outcome["control_steps"] + 1))
    instant, truth = [], []
    for fact in facts:
        speeds = [fact["linear_speed"], fact["angular_speed"]]
        good = fact["inside"] and fact["basket_contact"] and not fact["finger_contact"]
        good = (
            good
            and all(math.isfinite(v) and v >= 0 for v in speeds)
            and speeds[0] <= 0.03
            and speeds[1] <= 0.3
        )
        instant.append(good)
        strict = len(instant) >= 11 and all(instant[-11:])
        assert strict == fact["strict_complete"]
        truth.append(strict)
    history = streak = 0
    revision, claim = None, None
    for row in checks:
        binding = row["observation"]
        assert binding == bindings[row["control"]]
        assert binding["request_id"] == outcome["request_id"]
        if revision != binding["revision"]:
            history = streak = 0
        revision = binding["revision"]
        if row["event"] == "stale_observation":
            history = streak = 0
            continue
        history += 1
        ready, decision = history >= 11, "unknown"
        p = row["probabilities"]
        if ready:
            assert len(p) == 3 and all(math.isfinite(v) and 0 <= v <= 1 for v in p)
            assert abs(sum(p) - 1) <= 1e-5
            decision = (
                "complete"
                if p[1] >= 0.95
                else "incomplete"
                if p[0] >= 0.9
                else "unknown"
            )
        else:
            assert p is None
        streak = streak + 1 if decision == "complete" else 0
        assert row["history_ready"] == ready and row["decision"] == decision
        assert row["confirmed"] == (streak >= 11)
        if streak >= 11 and claim is None:
            claim = row["control"]
    assert claim == outcome["first_claim_control"] == score["first_claim_control"]
    chunks = {
        e["observation"]["observation_id"]: e
        for e in events
        if e["event"] == "async_chunk"
    }
    warm = [e for e in events if e["event"] == "worker_warmup_complete"]
    assert (
        len(warm) == 1
        and warm[0]["engine_epoch_offset"] == outcome["engine_epoch_offset"] == 1
    )
    assert warm[0]["discarded_all_warmup_actions"]
    assert events.index(warm[0]) < events.index(dispatch[0])
    modified = 0
    for row, actual in zip(dispatch, actions):
        physical = list(row["action"])
        if row["control"] < fault:
            modified += int(physical[-1] != -1)
            physical[-1] = -1.0
        assert physical == actual
        source = row["source"]
        if source is None:
            continue
        binding = source["observation"]
        assert row["phase"] == "policy" and (claim is None or row["control"] < claim)
        assert binding["request_id"] == outcome["request_id"]
        assert source["epoch"] == row["revision"] + 1 == binding["revision"] + 1
        assert binding == bindings[source["control"]]
        assert (
            0
            <= row["dispatched_monotonic"] - binding["captured_monotonic"]
            <= config["action_source_max_age_s"]
        )
        chunk = chunks[binding["observation_id"]]
        assert (
            chunk["observation"] == binding
            and chunk["source_control"] == source["control"]
        )
        assert row["action"] in chunk["actions"]
    first = next(
        (i for i, v in enumerate(truth[: outcome["control_steps"] + 1]) if v), None
    )
    early = claim is not None and not truth[claim]
    missed = first is not None and claim is None
    stable = (
        len(truth[claim + 1 :]) == config["post_stop_controls"]
        and all(truth[claim + 1 :])
        if claim is not None
        else None
    )
    safe = (
        outcome["status"] == "complete"
        and not early
        and stable is True
        and score["integrity_passed"]
    )
    assert (first, early, missed, stable, safe) == (
        score["first_truth_control"],
        score["premature_stop"],
        score["missed_completed_event"],
        score["post_stop_stable"],
        score["safely_completed"],
    )
    if claim is not None and first is not None and not early:
        same((claim - first) / 20, score["confirmation_delay_s"])
        stop = next(e for e in events if e["event"] == "confirmed_stop")
        same(
            stop["confirmed_monotonic"] - bindings[first]["captured_monotonic"],
            score["confirmation_wall_delay_s"],
        )
    work = [e["work_wall_s"] for e in dispatch]
    periods = [
        b["started_monotonic"] - a["started_monotonic"]
        for a, b in zip(dispatch, dispatch[1:])
    ]
    timing = score["timing"]
    assert timing["controls"] == len(work)
    assert timing["deadline_misses"] == sum(v > config["period_s"] for v in work)
    same(quantile(work, 0.95), timing["work_p95_s"])
    assert (
        timing["stale_observations"]
        == sum(e["event"] == "stale_observation" for e in events)
        == 0
    )
    same(quantile(periods, 0.95), timing["period_p95_s"])
    same(max(periods), timing["period_max_s"])
    retry = [e for e in events if e["event"] == "recovery_started"]
    assert outcome["attempts"] == 1 + len(retry) <= config["max_attempts"]
    assert (
        outcome["control_steps"]
        <= config["max_attempts"]
        * (config["policy_controls"] + config["settle_controls"])
        + (config["max_attempts"] - 1) * config["recovery_open_controls"]
    )
    end = config["policy_controls"] + config["settle_controls"]
    eligible = bool(
        fault and modified and len(truth) > end and not any(truth[: end + 1])
    )
    recovered = bool(eligible and safe and retry and claim > retry[0]["control"])
    assert score["recovery"]["physical_failure_eligible"] == eligible
    assert score["recovery"]["recovered_from_failure"] == recovered
    return score


def replay(bundle=None):
    manifest = json.loads((HERE / "evidence_manifest.json").read_text())
    backup = json.loads((HERE / "backup-verification.json").read_text())
    assert backup["status"] == "PASS" and not backup["errors"]
    assert backup["verified_files"] == manifest["files"] > 0
    assert backup["archives"] == manifest
    archive = HERE / "raw_records.tar.gz"
    assert sha(archive_bytes(archive)) == manifest["compact"]["sha256"]
    with open_archive(archive) as tar:
        raw = {
            m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()
        }

    def read(name):
        return json.loads(raw[name])

    def lines(name):
        return [json.loads(line) for line in raw[name].splitlines()]

    files = read("budget-files.json")["files"]
    assert len(files) == manifest["files"]
    for name, entry in files.items():
        if name in raw:
            assert (
                len(raw[name]) == entry["bytes"] and sha(raw[name]) == entry["sha256"]
            ), name
        else:
            assert Path(name).suffix in (".npz", ".pt"), name
        if bundle is not None:
            path = bundle / name
            assert path.stat().st_size == entry["bytes"], name
            with path.open("rb") as stream:
                assert (
                    hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
                ), name
    frozen, protocol = read(RUN + "freeze.json"), read(RUN + "protocol.json")
    delivery = read("current-delivery.json")
    assert delivery["status"] == "PASS"
    assert delivery["source_commit"] == frozen["source_commit"]
    assert delivery["completed_unix"] < frozen["frozen_unix"]
    assert delivery["supported_score"]["safely_completed"]
    assert delivery["blocked"]["backend_created"] is False
    assert all(value == 0 for value in delivery["caches"].values())
    assert delivery["dependency_versions_equal_original"] is True
    setup = read("acceptance-setup-failure.json")
    assert setup["new_data_seen"] is False
    assert frozen["new_data_seen"] is False
    assert sha(raw[RUN + "protocol.json"]) == frozen["protocol_sha256"]
    assert sha(raw[RUN + "excluded_layouts.json"]) == frozen["exclusions_sha256"]
    for name, expected in frozen["source_sha256"].items():
        assert sha(raw[RUN + "source/" + name]) == expected, name
    layouts = set(read(RUN + "excluded_layouts.json"))
    assert len(layouts) == 142
    results = {}
    gates = protocol["gates"]
    for phase in ("normal", "recovery"):
        value = read(RUN + phase + "/result.json")
        assert value["started_unix"] > frozen["frozen_unix"]
        assert value["source_commit"] == frozen["source_commit"]
        assert value["freeze_sha256"] == sha(raw[RUN + "freeze.json"])
        assert [e["seed"] for e in value["episodes"]] == protocol[phase + "_seeds"]
        scores = []
        for index, episode in enumerate(value["episodes"]):
            prefix = RUN + phase + f"/episode-{index:02d}/"
            layout = read(prefix + "initial_layout.json")["sha256"]
            assert layout not in layouts
            layouts.add(layout)
            score = replay_episode(
                read,
                lines,
                prefix,
                protocol,
                protocol["fault"]["controls_exclusive"] if phase == "recovery" else 0,
            )
            assert score == episode["score"]
            scores.append(score)
        common = len(scores) == 10 and all(
            s["integrity_passed"]
            and not s["premature_stop"]
            and not s["missed_completed_event"]
            and (
                s["confirmation_delay_s"] is None
                or s["confirmation_delay_s"] <= gates["max_confirmation_delay_s"]
            )
            and (
                s["confirmation_wall_delay_s"] is None
                or 0
                <= s["confirmation_wall_delay_s"]
                <= gates["max_confirmation_delay_s"]
            )
            for s in scores
        )
        safe = sum(s["safely_completed"] for s in scores)
        timing_pass = sum(
            s["timing"]["period_p95_s"] <= gates["normal_period_p95_s_max"]
            and s["timing"]["period_max_s"] <= gates["normal_period_max_s_max"]
            and s["timing"]["deadline_misses"] / s["timing"]["controls"]
            <= gates["normal_deadline_miss_fraction_max"]
            for s in scores
        )
        eligible = sum(s["recovery"]["physical_failure_eligible"] for s in scores)
        recovered = sum(s["recovery"]["recovered_from_failure"] for s in scores)
        passed = common and (
            safe >= gates["normal_safe_completions_min"] and timing_pass == 10
            if phase == "normal"
            else eligible >= gates["recovery_eligible_min"]
            and recovered >= gates["recovered_min"]
        )
        assert value["status"] == ("GO" if passed else "NO-GO")
        assert value["safely_completed"] == safe
        results[phase] = dict(
            status=value["status"],
            safe=safe,
            timing_pass=timing_pass,
            eligible=eligible,
            recovered=recovered,
        )
    audit = read(RUN + "state-rgb-audit/summary.json")
    assert audit["episodes"] == 20 and len(audit["records"]) == 20
    assert audit["freeze_sha256"] == sha(raw[RUN + "freeze.json"])
    restored_count = clip_count = 0
    for row in audit["records"]:
        phase, episode = row["id"].split("/")
        restored = read(RUN + f"state-rgb-audit/{phase}-{episode}.json")
        recorded = read(RUN + row["id"] + "/private_truth.json")
        assert len(restored) == len(recorded) == row["states"]
        assert (
            row["trajectory_sha256"]
            == files[RUN + row["id"] + "/trajectory.npz"]["sha256"]
        )
        assert row["scoring_matches"] is True
        assert all(
            a["strict_complete"] == b["strict_complete"]
            for a, b in zip(restored, recorded)
        )
        restored_count += len(restored)
        clip_count += row["clips_checked"]
    assert restored_count == audit["states"] and restored_count > 0
    assert clip_count == audit["clips_checked"] and clip_count > 0
    assert audit["status"] == "PASS" and audit["changed_strict"] == 0
    runtime = read("runtime-source-verification.json")
    assert runtime["status"] == "PASS"
    assert runtime["source_sha256"] == frozen["source_sha256"]
    return dict(
        status="PASS",
        acceptance=results,
        state_audit=audit["status"],
        full_bundle="PASS" if bundle is not None else "not_requested",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    print(json.dumps(replay(parser.parse_args().bundle), indent=2))
