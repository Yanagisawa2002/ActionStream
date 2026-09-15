"""Independent stdlib replay of original requests, RGB decisions and physical logs.

No production controller, model, authorization or scorer modules are imported.
RGB arrays and simulator restoration are covered by the separately retained audit.
"""

import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def proof(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def unique(pairs):
    result = {}
    for k, v in pairs:
        assert k not in result, k
        result[k] = v
    return result


def whole_request(text, contract, grammar):
    # Independently tokenize and consume the entire utterance against frozen rules.
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*|[0-9]+|[^\w\s]"
    matches = list(re.finditer(pattern, text, re.UNICODE))
    tokens = [m.group().casefold() for m in matches]
    residue = list(text)
    for m in matches:
        residue[m.start() : m.end()] = " " * (m.end() - m.start())
    aliases = []
    for slot, marker in (("target", "T"), ("destination", "B")):
        for alias in contract["aliases"][slot]:
            aliases.append(
                ([m.group().casefold() for m in re.finditer(pattern, alias)], marker)
            )
    aliases.sort(key=lambda row: -len(row[0]))
    symbols, i = [], 0
    while i < len(tokens):
        found = next((a for a in aliases if tokens[i : i + len(a[0])] == a[0]), None)
        symbols.append(found[1] if found else tokens[i])
        i += len(found[0]) if found else 1
    normalized = " ".join(x for x in symbols if x != ",").replace(" ; ", " and ")
    normalized = re.sub(r" [.?]$", "", normalized).replace('item labeled " T "', "T")
    negated = bool(
        set(tokens)
        & {"no", "not", "never", "don't", "cannot", "can't", "won't", "without"}
    )
    good = (
        not "".join(residue).strip()
        and not negated
        and any(
            re.fullmatch(grammar["prefix"] + rule + grammar["suffix"], normalized)
            for rule in grammar["productions"]
        )
    )
    return bool(good), "reject" if negated else "unknown"


def language_decision(parsed, config, grammar):
    call, original = parsed["call"], parsed["original"]
    if call["status"] != "COMPLETED":
        return "error", "EXPLICIT_FAILURE"
    try:
        raw = json.loads(call["raw_output"], object_pairs_hook=unique)
        assert type(raw) is dict
        decision = raw["decision"]
        if decision in ("reject", "unknown"):
            assert set(raw) == {"decision", "reason"}
            assert raw["reason"] in config["contract"]["reasons"][decision]
            return decision, "VALID"
        assert decision == "accept" and set(raw) == {
            "decision",
            "target_quote",
            "destination_quote",
        }
        assert all(
            isinstance(raw[k], str) and raw[k].strip()
            for k in ("target_quote", "destination_quote")
        )
    except (ValueError, AssertionError, TypeError, KeyError):
        return "explicit_failure", "EXPLICIT_FAILURE"
    for slot in ("target", "destination"):
        quote = raw[slot + "_quote"]
        normalized = re.sub(r"^(?:the|a|an) ", "", " ".join(quote.casefold().split()))
        if (
            len(re.findall(r"(?<!\w)" + re.escape(quote) + r"(?!\w)", original["text"]))
            != 1
            or normalized not in config["contract"]["aliases"][slot]
        ):
            return "explicit_failure", "VALID"
    for guard in config["contract"]["text_guards"]:
        if re.search(guard["pattern"], original["text"], re.IGNORECASE):
            return "reject", "VALID"
    good, failure = whole_request(original["text"], config["contract"], grammar)
    return ("accept" if good else failure), "VALID"


def replay():
    manifest = json.loads((HERE / "evidence_manifest.json").read_text())
    archive = HERE / "raw_records.tar.gz"
    assert sha(archive.read_bytes()) == manifest["archive_sha256"]
    with tarfile.open(archive, "r:gz") as tar:
        raw = {
            m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()
        }
    assert set(raw) == set(manifest["files"])
    for name, digest in manifest["files"].items():
        assert sha(raw[name]) == digest, name

    def read(name):
        return json.loads(raw[name])

    p, freeze, config = (
        read("protocol.json"),
        read("freeze.json"),
        read("language_config.json"),
    )
    for name in (
        "freeze.json",
        "protocol.json",
        "language_config.json",
        "smoke.json",
        "verdict.json",
        "external_blobs.json",
        "audit_restart.json",
        "audit_cache_restart.json",
        "run_receipt.json",
    ):
        assert (HERE / name).read_bytes() == raw[name], name
    grammar = json.loads((HERE / "coverage_grammar.json").read_text())
    assert sha(raw["protocol.json"]) == freeze["protocol_sha256"]
    for name, digest in freeze["source_sha256"].items():
        assert sha(raw["source/" + name]) == digest, name
    assert read("run_receipt.json")["status"] == "COMPLETED"
    assert (
        freeze["frozen_unix"]
        < read("smoke.json")["completed_unix"]
        <= read("run_receipt.json")["started_unix"]
    )
    assert freeze["new_data_seen"] is False and read("smoke.json")["status"] == "PASS"
    assert (
        p["checkpoint_sha256"]
        == freeze["checkpoint_sha256"]
        == "28482b40e470d932dfe44312d2dbcb7fa732146a2b2f9c7ebf7683b47cdfa7f9"
    )
    assert config == read("source/configs/finite_agent_language.json")
    state_audit = read("state-rgb-audit/summary.json")
    assert state_audit["source_sha256"] == sha(
        raw["state-rgb-audit/source_snapshot.py"]
    )
    assert state_audit["freeze_sha256"] == sha(raw["freeze.json"])
    state_records = {r["id"]: r for r in state_audit["records"]}
    language, scores, layouts, diagnostic = (
        [],
        [],
        set(read("excluded_layouts.json")),
        {},
    )
    case_ids = [c["id"] for c in p["requests"]]
    raw_negative_accepts = 0
    assert len(case_ids) == len(set(case_ids)) == 40
    assert set(case_ids) == set(p["labels"])
    for case in p["requests"]:
        base = "run/" + case["id"] + "/"
        parsed = read(base + "parser_call.json")
        assert parsed["original"] == dict(request_id=case["id"], text=case["text"])
        decision, schema = language_decision(parsed, config, grammar)
        if p["labels"][case["id"]] == "unsupported" and schema == "VALID":
            raw_negative_accepts += (
                json.loads(parsed["call"]["raw_output"])["decision"] == "accept"
            )
        assert (
            decision == parsed["verdict"]["decision"]
            and schema == parsed["verdict"]["schema_status"]
        )
        language.append(
            dict(
                id=case["id"],
                decision=decision,
                model_status=parsed["call"]["status"],
                schema_status=schema,
            )
        )
        if "seed" not in case:
            assert base + "runtime.jsonl" not in raw
            continue
        outcome = read(base + "outcome.json")
        if decision != "accept":
            assert outcome["status"] in {"LANGUAGE_BLOCKED", "LANGUAGE_ERROR"}
            assert (
                outcome["backend_created"] is False
                and base + "runtime.jsonl" not in raw
            )
            continue
        initial = read(base + "initial_layout.json")
        assert initial["seed"] == case["seed"] and initial["sha256"] not in layouts
        layouts.add(initial["sha256"])
        runtime = [
            json.loads(line) for line in raw[base + "runtime.jsonl"].splitlines()
        ]
        authorization = runtime[0]
        assert authorization["event"] == "authorized"
        permit = authorization["permit"]
        assert (
            permit["original"] == parsed["original"]
            and permit["raw"] == parsed["call"]["raw_output"]
        )
        assert permit["proof_sha256"] == proof(parsed["verdict"])
        assert permit["contract_sha256"] == proof(config["contract"])
        assert permit["grammar_sha256"] == proof(grammar)
        facts, bindings, actions = (
            read(base + name)
            for name in (
                "private_truth.json",
                "observation_bindings.json",
                "actions.json",
            )
        )
        strict, consecutive = [], 0
        for row in facts:
            instant = (
                row["inside"]
                and not row["finger_contact"]
                and row["basket_contact"]
                and math.isfinite(row["linear_speed"])
                and 0 <= row["linear_speed"] <= 0.03
                and math.isfinite(row["angular_speed"])
                and 0 <= row["angular_speed"] <= 0.3
            )
            consecutive = consecutive + 1 if instant else 0
            strict.append(consecutive >= 11)
            assert strict[-1] == row["strict_complete"]
        restored = read("state-rgb-audit/" + case["id"] + ".json")
        audit_row = state_records[case["id"]]
        assert len(restored) == len(facts) == audit_row["states"]
        assert (
            audit_row["image_state_sha256"]
            == read("external_blobs.json")[base + "trajectory.npz"]["sha256"]
        )
        assert [r["strict_complete"] for r in restored] == strict
        differences = []
        for control, (before, after) in enumerate(zip(facts, restored)):
            fields = [
                k
                for k in (
                    "inside",
                    "finger_contact",
                    "basket_contact",
                    "strict_complete",
                )
                if before[k] != after[k]
            ]
            if fields:
                differences.append(
                    dict(
                        control=control, fields=fields, recorded=before, restored=after
                    )
                )
        assert differences == audit_row["changed"]
        checks = [r for r in runtime if r["event"] == "verification"]
        controls = outcome["control_steps"]
        assert audit_row["rgb_clips_checked"] == max(0, controls - 9)
        assert [r["control"] for r in checks] == list(range(controls + 1))
        positives, claim = 0, None
        for row in checks:
            c, probabilities = row["control"], row["probabilities"]
            if c < 10:
                assert probabilities is None and row["clip_sha256"] is None
                value = "unknown"
            else:
                assert len(probabilities) == 3 and all(
                    math.isfinite(v) and 0 <= v <= 1 for v in probabilities
                )
                assert abs(sum(probabilities) - 1) <= 2e-5
                value = (
                    "complete"
                    if probabilities[1] >= 0.95
                    else "incomplete"
                    if probabilities[0] >= 0.90
                    else "unknown"
                )
                key = f"truth_{int(strict[c])}_{value}"
                diagnostic[key] = diagnostic.get(key, 0) + 1
            positives = positives + 1 if value == "complete" else 0
            assert row["decision"] == value and row["confirmed"] == (positives >= 11)
            assert (
                row["history_ready"] == (c >= 10) and row["observation"] == bindings[c]
            )
            assert (
                bindings[c]["request_id"] == case["id"] and bindings[c]["revision"] == 0
            )
            if row["confirmed"] and claim is None:
                claim = c
        assert len({b["observation_id"] for b in bindings}) == len(bindings)
        assert all(
            a["captured_monotonic"] < b["captured_monotonic"]
            for a, b in zip(bindings, bindings[1:])
        )
        assert claim == outcome["first_claim_control"] and outcome["status"] != "ERROR"
        assert (
            (claim == controls and outcome["status"] == "complete")
            if claim is not None
            else (controls == 360 and outcome["status"] == "unconfirmed_horizon")
        )
        post = strict[claim + 1 :] if claim is not None else None
        assert outcome["post_stop_controls"] == (40 if claim is not None else 0)
        assert (
            len(actions)
            == controls + outcome["post_stop_controls"]
            == len(strict) - 1
            == len(bindings) - 1
        )
        chunks = {
            r["source_control"]: r["actions"] for r in runtime if r["event"] == "chunk"
        }
        assert sorted(chunks) == list(range(0, min(controls, 300), 30))
        for index, action in enumerate(actions):
            expected = (
                [0.0] * 6 + [-1.0]
                if index >= min(controls, 300)
                else chunks[index // 30 * 30][index % 30]
            )
            assert action == expected
        first_truth = next(
            (i for i, yes in enumerate(strict[: controls + 1]) if yes), None
        )
        premature = claim is not None and not strict[claim]
        metric = dict(
            first_truth_control=first_truth,
            first_claim_control=claim,
            first_correct_control=claim
            if claim is not None and strict[claim]
            else None,
            premature_stop=premature,
            missed_completed_event=first_truth is not None
            and (claim is None or not strict[claim]),
            no_stop=claim is None,
            confirmation_delay_s=(claim - first_truth) / 20
            if claim is not None and first_truth is not None and not premature
            else None,
            post_stop_stable=all(post) if post else None,
            integrity_passed=True,
            integrity_errors=[],
        )
        assert metric == read(base + "independent_score.json"), case["id"]
        scores.append(
            dict(id=case["id"], seed=case["seed"], outcome=outcome, score=metric)
        )
    assert language == read("language_progress.json")
    assert scores == read("execution_progress.json")
    assert set(state_records) == {e["id"] for e in scores}
    accepted = {r["id"] for r in language if r["decision"] == "accept"}
    supported = {k for k, v in p["labels"].items() if v == "supported"}
    executable = {r["id"] for r in p["requests"] if "seed" in r}
    lm = dict(
        cases=len(language),
        supported=len(supported),
        supported_accepted=len(accepted & supported),
        unsupported=len(p["labels"]) - len(supported),
        unsupported_accepted=len(accepted - supported),
        model_or_schema_errors=sum(
            r["model_status"] != "COMPLETED" or r["schema_status"] != "VALID"
            for r in language
        ),
        exact_request_coverage=True,
    )
    delays = [
        e["score"]["confirmation_delay_s"]
        for e in scores
        if e["score"]["confirmation_delay_s"] is not None
    ]
    rm = dict(
        requested_tasks=len(executable),
        executed_tasks=len(scores),
        language_blocked_tasks=len(executable - accepted),
        physically_completed=sum(
            e["score"]["first_truth_control"] is not None for e in scores
        ),
        safely_completed_tasks=sum(
            e["outcome"]["status"] == "complete"
            and not e["score"]["premature_stop"]
            and e["score"]["post_stop_stable"] is True
            for e in scores
        ),
        premature_stops=sum(e["score"]["premature_stop"] for e in scores),
        missed_completed_events=sum(
            e["score"]["missed_completed_event"] for e in scores
        ),
        post_stop_failures=sum(e["score"]["post_stop_stable"] is False for e in scores),
        integrity_failures=0,
        maximum_confirmation_delay_s=max(delays) if delays else None,
        exact_execution_coverage={e["id"] for e in scores} == executable & accepted,
    )
    passed = (
        lm["supported_accepted"] >= p["gates"]["supported_accepted_min"]
        and lm["unsupported_accepted"] == lm["model_or_schema_errors"] == 0
        and rm["safely_completed_tasks"] >= p["gates"]["safely_completed_tasks_min"]
        and rm["premature_stops"]
        == rm["missed_completed_events"]
        == rm["post_stop_failures"]
        == 0
        and delays
        and max(delays) <= p["gates"]["max_confirmation_delay_s"]
        and rm["exact_execution_coverage"]
    )
    verdict = dict(status="GO" if passed else "NO-GO", language=lm, runtime=rm)
    assert verdict == read("verdict.json")
    return dict(
        audit="PASS",
        **verdict,
        diagnostic=diagnostic,
        raw_model_negative_accepts=raw_negative_accepts,
        delay_median_s=statistics.median(delays) if delays else None,
        delay_mean_s=statistics.mean(delays) if delays else None,
        delay_sample_std_s=statistics.stdev(delays) if len(delays) > 1 else None,
        audited_states=sum(r["states"] for r in state_records.values()),
        audited_rgb_clips=sum(r["rgb_clips_checked"] for r in state_records.values()),
        restored_contact_or_containment_changes=sum(
            len(r["changed"]) for r in state_records.values()
        ),
        restored_strict_label_changes=0,
        physical_controls=sum(
            e["outcome"]["control_steps"] + e["outcome"]["post_stop_controls"]
            for e in scores
        ),
        limitation="Independent record replay; single-task synchronous integration, not real-time or physical-robot acceptance",
    )


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
