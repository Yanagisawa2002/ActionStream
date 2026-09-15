"""Rebuild the readable per-request analysis from independently verified records."""

import json
from pathlib import Path
import statistics
import tarfile

from verify_evidence import replay, sha

HERE = Path(__file__).resolve().parent


def main():
    result = replay()
    with tarfile.open(HERE / "raw_records.tar.gz", "r:gz") as tar:
        raw = {
            m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()
        }

    def read(name):
        return json.loads(raw[name])

    language, episodes = [], []
    for case in read("protocol.json")["requests"]:
        base = "run/" + case["id"] + "/"
        parsed = read(base + "parser_call.json")
        generated = json.loads(parsed["call"]["raw_output"])
        language.append(
            dict(
                id=case["id"],
                text=case["text"],
                supported="seed" in case,
                raw_model_decision=generated["decision"],
                final_decision=parsed["verdict"]["decision"],
                reason=parsed["verdict"].get("reason"),
                model_wall_s=parsed["call"]["wall_s"],
            )
        )
        if "seed" in case:
            out, score = (
                read(base + "outcome.json"),
                read(base + "independent_score.json"),
            )
            episodes.append(
                dict(
                    id=case["id"],
                    seed=case["seed"],
                    instruction=case["text"],
                    control_steps=out["control_steps"],
                    post_stop_controls=out["post_stop_controls"],
                    wall_s=out["wall_s"],
                    **score,
                )
            )
    analysis = dict(
        raw_model_negative_accepts=result["raw_model_negative_accepts"],
        host_blocked_model_negative_accepts=[
            r
            for r in language
            if not r["supported"] and r["raw_model_decision"] == "accept"
        ],
        language_latency_median_s=statistics.median(
            r["model_wall_s"] for r in language
        ),
        language_latency_max_s=max(r["model_wall_s"] for r in language),
        agent_episode_wall_s_mean=statistics.mean(r["wall_s"] for r in episodes),
        agent_episode_wall_s_max=max(r["wall_s"] for r in episodes),
        excluded_layouts=len(read("excluded_layouts.json")),
        recorded_controls=sum(r["control_steps"] for r in episodes),
        post_stop_controls=sum(r["post_stop_controls"] for r in episodes),
        freeze_sha256=sha(raw["freeze.json"]),
        archive_sha256=sha((HERE / "raw_records.tar.gz").read_bytes()),
        language=language,
        episodes=episodes,
    )
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (HERE / "independent_replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
