"""One immutable held-out evaluation of the already selected completion model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from actionstream.llm_vla.completion import (
    CHECKPOINT_SHA256,
    MANIFEST_SHA256,
    FrozenCompletion,
    decision,
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(rows):
    positive = [r for r in rows if r["truth"]]
    negative = [r for r in rows if not r["truth"]]
    return dict(
        frames=len(rows),
        positive=len(positive),
        negative=len(negative),
        false_complete=sum(r["decision"] == "complete" for r in negative),
        true_complete=sum(r["decision"] == "complete" for r in positive),
        missed_complete=sum(r["decision"] != "complete" for r in positive),
        false_incomplete=sum(r["decision"] == "incomplete" for r in positive),
        unknown=sum(r["decision"] == "unknown" for r in rows),
        positive_unknown=sum(r["decision"] == "unknown" for r in positive),
        negative_unknown=sum(r["decision"] == "unknown" for r in negative),
    )


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data", "output", "protocol"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    assert protocol["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert protocol["thresholds"] == {"incomplete": 0.1, "complete": 0.9}
    if digest(args.data / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Frozen data manifest mismatch")
    manifest = json.loads((args.data / "manifest.json").read_text())
    splits = [set(manifest["splits"][key]) for key in ("train", "validation", "test")]
    if any(splits[i] & splits[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Episode leakage")
    if len(splits[2]) != 10:
        raise ValueError("Expected ten test episodes")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_bytes(args.protocol.read_bytes())
    torch.set_num_threads(8)
    model = FrozenCompletion(args.checkpoint)
    started, rows, episodes = time.time(), [], {}
    with (args.output / "predictions.jsonl").open("x") as journal:
        for name in manifest["splits"]["test"]:
            path = args.data / f"{name}.npz"
            if digest(path) != manifest["episodes"][name]["sha256"]:
                raise ValueError(f"Shard changed: {name}")
            episode_rows = []
            with np.load(path, allow_pickle=False) as shard:
                for start in range(0, len(shard["labels"]), 64):
                    images = shard["rgb"][start : start + 64]
                    probabilities = model.predict(images)
                    for offset, probability in enumerate(probabilities):
                        index = start + offset
                        row = dict(
                            episode=name,
                            frame_index=int(shard["frame_indices"][index]),
                            truth=bool(shard["labels"][index]),
                            probability=float(probability),
                            decision=decision(float(probability)),
                            rgb_sha256=hashlib.sha256(
                                images[offset].tobytes()
                            ).hexdigest(),
                        )
                        journal.write(json.dumps(row) + "\n")
                        journal.flush()
                        episode_rows.append(row)
            episodes[name] = metrics(episode_rows)
            rows.extend(episode_rows)
            print(json.dumps(dict(episode=name, **episodes[name])), flush=True)
    if len(rows) != 485:
        raise ValueError("Expected 485 frozen test frames")
    summary = dict(
        status="COMPLETED",
        checkpoint_sha256=CHECKPOINT_SHA256,
        manifest_sha256=MANIFEST_SHA256,
        protocol_sha256=digest(args.protocol),
        evaluator_sha256=digest(__file__),
        metrics=metrics(rows),
        episodes=episodes,
        episodes_with_false_complete=sum(
            v["false_complete"] > 0 for v in episodes.values()
        ),
        elapsed_seconds=time.time() - started,
        thresholds=protocol["thresholds"],
        weights_updated=False,
        thresholds_tuned=False,
        prediction_file_sha256=digest(args.output / "predictions.jsonl"),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
