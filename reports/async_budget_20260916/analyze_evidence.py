"""Generate report metrics exclusively from retained evidence."""

import json
from pathlib import Path
from record_io import open_archive

HERE = Path(__file__).resolve().parent


def analyze():
    value = {}
    with open_archive(HERE / "raw_records.tar.gz") as tar:

        def read(name):
            return json.load(tar.extractfile(name))

        for phase in ("normal", "recovery"):
            result = read(f"budget-acceptance-v2/{phase}/result.json")
            scores = [e["score"] for e in result["episodes"]]
            controls = sum(s["timing"]["controls"] for s in scores)
            misses = sum(s["timing"]["deadline_misses"] for s in scores)
            warmup = []
            for i in range(10):
                events = [
                    json.loads(line)
                    for line in tar.extractfile(
                        f"budget-acceptance-v2/{phase}/episode-{i:02d}/runtime.jsonl"
                    )
                ]
                warmup.extend(
                    e["wall_s"]
                    for e in events
                    if e["event"] == "worker_warmup_complete"
                )

            def bounds(key):
                rows = [s["timing"][key] * 1000 for s in scores]
                return [min(rows), max(rows)]

            value[phase] = dict(
                status=result["status"],
                safe=sum(s["safely_completed"] for s in scores),
                controls=controls,
                misses=misses,
                miss_percent=100 * misses / controls,
                worst_episode_miss_percent=max(
                    100 * s["timing"]["deadline_misses"] / s["timing"]["controls"]
                    for s in scores
                ),
                work_p95_ms_range=bounds("work_p95_s"),
                period_p95_ms_range=bounds("period_p95_s"),
                period_max_ms=max(s["timing"]["period_max_s"] for s in scores) * 1000,
                premature_stops=sum(s["premature_stop"] for s in scores),
                missed_completion_events=sum(
                    s["missed_completed_event"] for s in scores
                ),
                stale_observations=sum(
                    s["timing"]["stale_observations"] for s in scores
                ),
                integrity_errors=sum(not s["integrity_passed"] for s in scores),
                action_source_max_age_s=max(
                    s["timing"]["action_source_max_age_s"] for s in scores
                ),
                confirmation_wall_max_s=max(
                    s["confirmation_wall_delay_s"] or 0 for s in scores
                ),
                physical_failure_eligible=sum(
                    s["recovery"]["physical_failure_eligible"] for s in scores
                ),
                recovered=sum(s["recovery"]["recovered_from_failure"] for s in scores),
                worker_start_and_warmup_s_range=[min(warmup), max(warmup)],
                group_elapsed_s=result["completed_unix"] - result["started_unix"],
            )
        audit = read("budget-acceptance-v2/state-rgb-audit/summary.json")
        value["state_audit"] = {k: v for k, v in audit.items() if k != "records"}
        samples = list(tar.extractfile("gpu-residency-samples.csv"))
        value["gpu_memory"] = dict(
            sample_count=len(samples),
            interval_s=1,
            observed_max_mib=max(
                float(line.decode().split(",")[1]) for line in samples
            ),
            scope="90-second sampling window; not a certified episode peak",
        )
    with open_archive(HERE / "development_records.tar.gz") as tar:
        value["development"] = []
        names = sorted(
            m.name for m in tar.getmembers() if m.name.endswith("/profile-summary.json")
        )
        for name in names:
            summary = json.load(tar.extractfile(name))
            scores = summary["episodes"]
            journal_ms, inclusive_work, dispatch_log_ms = [], [], []
            for row in scores:
                prefix = name.split("/")[0] + "/" + str(row["seed"]) + "/"
                profile = json.load(tar.extractfile(prefix + "profile.json"))
                events = [
                    json.loads(line)
                    for line in tar.extractfile(prefix + "runtime.jsonl")
                ]
                owners = {
                    s["thread"]
                    for s in profile["cpu"]
                    if s["label"] == "simulator_step"
                }
                assert len(owners) == 1
                writes = sorted(
                    (
                        s
                        for s in profile["cpu"]
                        if s["label"] == "journal_write" and s["thread"] in owners
                    ),
                    key=lambda s: s["start"],
                )
                journal_ms.extend((s["end"] - s["start"]) * 1000 for s in writes)
                for event in events:
                    if event["event"] != "dispatch":
                        continue
                    end = event["started_monotonic"] + event["work_wall_s"]
                    write = next(s for s in writes if s["start"] >= end)
                    assert 0 <= write["start"] - end < 0.05
                    inclusive_work.append(write["end"] - event["started_monotonic"])
                    dispatch_log_ms.append((write["end"] - end) * 1000)
            controls = sum(s["timing"]["controls"] for s in scores)
            misses = sum(s["timing"]["deadline_misses"] for s in scores)
            value["development"].append(
                dict(
                    run=name.split("/")[0],
                    cases=len(scores),
                    controls=controls,
                    misses=misses,
                    miss_percent=100 * misses / controls,
                    safe=sum(s["safely_completed"] for s in scores),
                    every_case_below_one_percent=all(
                        s["timing"]["deadline_misses"] / s["timing"]["controls"] <= 0.01
                        for s in scores
                    ),
                    spans=summary["spans"],
                    missed_control_spans=summary["missed_control_spans"],
                    all_main_thread_journal_writes=dict(
                        n=len(journal_ms),
                        mean_ms=sum(journal_ms) / len(journal_ms),
                        max_ms=max(journal_ms),
                    ),
                    post_dispatch_diagnostic=dict(
                        controls=len(inclusive_work),
                        misses_including_dispatch_journal=sum(
                            v > 0.05 for v in inclusive_work
                        ),
                        mean_added_ms=sum(dispatch_log_ms) / len(dispatch_log_ms),
                        max_added_ms=max(dispatch_log_ms),
                        scope="Development-only work through dispatch-journal completion, including serialization and lock wait; frozen acceptance timer unchanged.",
                    ),
                )
            )
    return value


if __name__ == "__main__":
    result = analyze()
    (HERE / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "development"}, indent=2))
