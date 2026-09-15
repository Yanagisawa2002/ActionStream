"""Independent, standard-library audit of frozen heldout acceptance evidence."""

from collections import Counter
import importlib.util
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Reuse only the prior independent auditor's pure math; no production code imports.
spec = importlib.util.spec_from_file_location(
    "development_audit",
    HERE.parent / "completion_development_20260915/verify_evidence.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def raw(path):
    if path.exists():
        return path.read_bytes()
    with tarfile.open(path.parent / "raw_records.tar.gz") as t:
        return t.extractfile(path.name).read()


def read(path):
    return json.loads(raw(path))


def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()


def score(rows, p):
    delays = [
        r["confirmation_delay_s"] for r in rows if r["confirmation_delay_s"] is not None
    ]
    stopped = [r for r in rows if r["first_claim_control"] is not None]
    result = dict(
        episodes=len(rows),
        physically_completed_episodes=sum(
            r["first_truth_control"] is not None for r in rows
        ),
        premature_stops=sum(r["premature_stop"] for r in rows),
        missed_completed_episodes=sum(r["missed_completed_event"] for r in rows),
        no_stop=sum(r["no_stop"] for r in rows),
        post_stop_failures=sum(r["post_stop_stable"] is False for r in stopped),
        missing_post_stop_probes=sum(r["post_stop_stable"] is None for r in stopped),
        maximum_confirmation_delay_s=max(delays) if delays else None,
        median_confirmation_delay_s=statistics.median(delays) if delays else None,
        late_confirmations=sum(d > 2 for d in delays),
        exact_seed_coverage=sorted(r["seed"] for r in rows) == sorted(p["seeds"]),
    )
    result["passed"] = (
        result["exact_seed_coverage"]
        and len(rows) == 20
        and result["physically_completed_episodes"] >= 16
        and all(
            result[k] == 0
            for k in (
                "premature_stops",
                "missed_completed_episodes",
                "post_stop_failures",
                "missing_post_stop_probes",
                "late_confirmations",
            )
        )
    )
    return result


def main():
    p = read(HERE / "protocol.json")
    freeze = read(HERE / "freeze.json")
    collection = read(HERE / "collection/manifest.json")
    challenges = read(HERE / "challenges/manifest.json")
    paired = read(HERE / "paired-evaluation/summary.json")
    live = read(HERE / "live/summary.json")
    verdict = read(HERE / "verdict.json")
    state_audit = read(HERE / "state-replay-audit/summary.json")
    assert p["truth"] == dict(
        window_steps=10, linear_speed_max=0.03, angular_speed_max=0.3
    )
    assert p["rgb_offsets"] == [10, 5, 0] and p["confirmation_controls"] == 10
    assert p["thresholds"] == dict(complete=0.95, incomplete=0.9)
    assert p["closed_loop_gate"] == dict(
        expected_episodes=20,
        minimum_physically_completed_episodes=16,
        premature_stops_max=0,
        missed_completed_episodes_max=0,
        max_confirmation_delay_s=2.0,
        post_stop_failures_max=0,
        missing_post_stop_probes_max=0,
    )
    assert p["challenges"]["gates"] == dict(
        expected_sequences=32,
        confirmed_false_complete_max=0,
        blackout_confident_decisions_max=0,
    )
    assert (
        not freeze["independent_test_seen"] and not collection["model_predictions_used"]
    )
    assert not verdict["test_used_for_fitting"] and live["independent_acceptance"]
    assert (
        freeze["frozen_unix"]
        < collection["completed_unix"]
        < live["completed_unix"]
        <= verdict["completed_unix"]
    )
    assert (
        freeze["protocol_sha256"]
        == sha(HERE / "protocol.json")
        == sha(ROOT / "configs/completion_acceptance_v3.json")
        == verdict["protocol_sha256"]
    )
    assert (
        sha(HERE / "source_snapshot.py")
        == freeze["source_sha256"]["scripts/engineering/completion_acceptance.py"]
    )
    for name, expected in freeze["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    for folder in ("collection", "challenges", "paired-evaluation", "live"):
        assert sha(HERE / folder / "source_snapshot.py") == sha(
            HERE / "source_snapshot.py"
        )
        assert sha(HERE / folder / "protocol.json") == sha(HERE / "protocol.json")
    for obj in (collection, paired, live, verdict, state_audit):
        assert obj["freeze_sha256"] == sha(HERE / "freeze.json")
    assert (
        paired["collection_sha256"]
        == challenges["collection_sha256"]
        == sha(HERE / "collection/manifest.json")
    )
    assert paired["challenge_manifest_sha256"] == sha(HERE / "challenges/manifest.json")
    assert (
        p["checkpoint_sha256"]
        == freeze["checkpoint_sha256"]
        == verdict["checkpoint_sha256"]
        == paired["results"]["adapted"]["checkpoint_sha256"]
    )
    assert paired["results"]["baseline"]["checkpoint_sha256"] == p["baseline_sha256"]
    assert read(HERE / "smoke_receipt.json")["status"] == "PASS"
    if (HERE / "archived_records.json").exists():
        for name, expected in read(HERE / "archived_records.json").items():
            assert sha(HERE / name) == expected, name
    entries = collection["entries"]
    assert [e["seed"] for e in entries] == p["seeds"]
    assert len(entries) == len({e["initial_layout_sha256"] for e in entries}) == 20
    dev = read(HERE.parent / "completion_development_20260915/collection/manifest.json")
    old = read(HERE.parent / "completion_v2_20260915/fresh_holdout/manifest.json")
    assert freeze["development_freeze_sha256"] == sha(
        HERE.parent / "completion_development_20260915/adaptation-training/frozen.json"
    )
    assert freeze["development_collection_sha256"] == sha(
        HERE.parent / "completion_development_20260915/collection/manifest.json"
    )
    assert freeze["consumed_v2_collection_sha256"] == sha(
        HERE.parent / "completion_v2_20260915/fresh_holdout/manifest.json"
    )
    old_layouts = {e["initial_layout_sha256"] for e in dev["entries"]} | {
        e["metadata"]["initial_layout_sha256"] for e in old["entries"]
    }
    assert not old_layouts & {e["initial_layout_sha256"] for e in entries}
    assert not set(p["seeds"]) & (
        {e["seed"] for e in dev["entries"]}
        | {e["metadata"]["seed"] for e in old["entries"]}
    )
    facts, restored = {}, {}
    for folder, items in (("collection", entries), ("live", live["episodes"])):
        for e in items:
            name = e["episode"]
            for filename, expected in e["files_sha256"].items():
                if not filename.endswith(".npz"):
                    assert sha(HERE / folder / filename) == expected
            trace = read(HERE / folder / (name + "_truth.json"))
            post = read(HERE / folder / (name + "_post.json"))
            audit.truth_check(trace + post)
            assert (
                len(trace) == e["control_records"]
                and len(post) == e["post_control_records"]
            )
            assert len(read(HERE / folder / (name + "_actions.json"))) == len(trace) - 1
            if folder == "collection":
                assert len(trace) == 361 and not post
            facts[folder, name] = trace
    assert state_audit["source_sha256"] == sha(
        HERE / "state-replay-audit/source_snapshot.py"
    )
    restored_changes, audited_controls = [], 0
    for r in state_audit["records"]:
        name, folder = r["episode"], r["stage"]
        fs = read(HERE / "state-replay-audit" / (folder + "_" + name + ".json"))
        original = facts[folder, name] + read(HERE / folder / (name + "_post.json"))
        audit.truth_check(fs)
        assert len(fs) == len(original) == r["controls"] + r["post_controls"]
        changes = [
            i
            for i, (a, b) in enumerate(zip(original, fs, strict=True))
            if a["strict_complete"] != b["strict_complete"]
        ]
        restored_changes.extend((folder, name, i) for i in changes)
        restored[folder, name] = fs
        item = next(
            e
            for e in (entries if folder == "collection" else live["episodes"])
            if e["episode"] == name
        )
        assert r["image_state_sha256"] == item["files_sha256"][name + ".npz"]
        if folder == "live":
            assert (
                item["reference_prefix_rgb_equal"] == r["reference_prefix"]["rgb_equal"]
            )
            assert (
                item["reference_prefix_max_state_delta"]
                == r["reference_prefix"]["max_state_delta"]
            )
        audited_controls += len(fs)
    assert len(state_audit["records"]) == 40
    diagnostics = {}
    for label, result in paired["results"].items():
        cells = Counter()
        for e in result["episodes"]:
            name = e["episode"]
            rows = read(HERE / "paired-evaluation" / f"{name}_{label}_predictions.json")
            assert [r["control"] for r in rows] == list(range(10, 361))
            indices = audit.claims(rows, "dense_20hz_confirm_0.5s")
            post = read(HERE / "paired-evaluation" / f"{name}_{label}_post.json")
            if post:
                audit.truth_check(
                    post,
                    prior=restored["collection", name][
                        indices[0] - 10 : indices[0] + 1
                    ],
                )
            actual = audit.events(facts["collection", name], indices, post)
            assert all(e[k] == v for k, v in actual.items()), (label, name)
            for r in rows:
                cells[
                    int(facts["collection", name][r["control"]]["strict_complete"]),
                    r["decision"],
                ] += 1
        assert score(result["episodes"], p) == result["closed_loop"]
        diagnostics[label] = {f"truth_{a}_{b}": n for (a, b), n in cells.items()}
        for e in result["challenges"]:
            case = e["case"]
            private = read(HERE / "challenges" / (case + ".json"))
            rows = read(HERE / "paired-evaluation" / f"{case}_{label}_predictions.json")
            assert [r["control"] for r in rows] == list(range(10, 21))
            indices = audit.claims(rows, "dense_20hz_confirm_0.5s")
            false_claims = [
                i for i in indices if not private["private_truth"][i]["strict_complete"]
            ]
            assert e["confirmed_false_complete_controls"] == false_claims
            assert e["blackout_confident_decisions"] == (
                sum(r["decision"] != "unknown" for r in rows)
                if e["category"].endswith("hidden")
                else 0
            )
        gate = dict(
            sequences=len(result["challenges"]),
            confirmed_false_complete=sum(
                len(r["confirmed_false_complete_controls"])
                for r in result["challenges"]
            ),
            blackout_confident_decisions=sum(
                r["blackout_confident_decisions"] for r in result["challenges"]
            ),
        )
        gate["passed"] = (
            gate["sequences"] == 32
            and gate["confirmed_false_complete"]
            == gate["blackout_confident_decisions"]
            == 0
        )
        assert result["challenge_gate"] == gate
    counts = Counter()
    assert not challenges["selection_uses_model_predictions"]
    for e in challenges["entries"]:
        path = HERE / "challenges" / (e["case"] + ".json")
        assert e["record_sha256"] == sha(path)
        r = read(path)
        counts[r["category"]] += 1
        assert r["source_episode"] in challenges["source_episodes"]
        prior = facts["collection", r["source_episode"]][
            r["source_controls"][0] - 10 : r["source_controls"][0]
        ]
        audit.truth_check(r["private_truth"], prior=prior)
        for i, f in enumerate(r["private_truth"]):
            assert r["labels"][i] == (
                2 if r["category"].endswith("hidden") else int(f["strict_complete"])
            )
            if r["category"] in ("above", "outside", "wrong_object") or (
                r["category"] == "recent_drop" and i >= 11
            ):
                assert not f["inside"]
    assert dict(counts) == {c: 4 for c in p["challenges"]["categories"]}
    for e in live["episodes"]:
        name = e["episode"]
        rows = read(HERE / "live" / (name + "_observations.json"))
        trace = facts["live", name]
        assert [r["control"] for r in rows] == list(range(10, len(trace)))
        indices = audit.claims(rows, "dense_20hz_confirm_0.5s")
        assert len(indices) <= 1
        if indices:
            assert indices[0] == len(trace) - 1
        else:
            assert len(trace) == 361
        post = read(HERE / "live" / (name + "_post.json"))
        actual = audit.events(trace, indices, post or None)
        assert all(e[k] == v for k, v in actual.items()), name
        ref = next(r for r in entries if r["episode"] == name)
        assert ref["initial_layout_sha256"] == e["initial_layout_sha256"]
        ref_actions = read(HERE / "collection" / (name + "_actions.json"))[
            : len(trace) - 1
        ]
        assert e["reference_prefix_actions_equal"] == (
            ref_actions == read(HERE / "live" / (name + "_actions.json"))
        )
    live_score = score(live["episodes"], p)
    assert live_score == live["closed_loop"] == verdict["live"]
    assert verdict["paired_adapted"] == paired["results"]["adapted"]["closed_loop"]
    assert verdict["paired_baseline"] == paired["results"]["baseline"]["closed_loop"]
    assert verdict["challenges"] == paired["results"]["adapted"]["challenge_gate"]
    expected = (
        "GO"
        if all(
            x["passed"]
            for x in (verdict["live"], verdict["paired_adapted"], verdict["challenges"])
        )
        else "NO-GO"
    )
    assert verdict["status"] == expected
    result = dict(
        status="PASS" if not restored_changes else "STRICT_LABEL_REPLAY_MISMATCH",
        verdict=expected,
        audited_simulator_states=audited_controls,
        restored_strict_label_changes=restored_changes,
        baseline_and_adapted_predictions=20 * 351 * 2,
        challenge_decisions=32 * 11 * 2,
        clip_diagnostics=diagnostics,
        live=live_score,
        failed_live_episodes=[
            {
                k: e[k]
                for k in (
                    "seed",
                    "first_truth_control",
                    "first_claim_control",
                    "premature_stop",
                    "missed_completed_event",
                    "confirmation_delay_s",
                    "post_stop_stable",
                )
            }
            for e in live["episodes"]
            if e["premature_stop"]
            or e["missed_completed_event"]
            or e["post_stop_stable"] is False
            or (e["confirmation_delay_s"] is not None and e["confirmation_delay_s"] > 2)
        ],
        paired_baseline=verdict["paired_baseline"],
        challenges=verdict["challenges"],
        reference_prefix_divergences=sum(
            not e["reference_prefix_rgb_equal"]
            or not e["reference_prefix_actions_equal"]
            or e["reference_prefix_max_state_delta"] != 0
            for e in live["episodes"]
        ),
        limitation="Frozen heldout protocol, one task and checkpoint; source replay and simulator state audit, not physical-robot acceptance",
    )
    (HERE / "independent_replay.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
