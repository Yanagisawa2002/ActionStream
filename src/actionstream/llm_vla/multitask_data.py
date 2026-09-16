"""Offline split guard and hash-bound episode loader for completion training."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .finite_native import digest
from .task_completion import TaskCondition
from .task_registry import development_task, registry_sha256

FILES = (
    "trajectory.npz",
    "private_truth.json",
    "task.json",
    "condition.json",
    "initial_layout.json",
    "visibility.json",
    "purpose.json",
)


def episode_entry(directory, split):
    directory = Path(directory)
    identity = json.loads((directory / "task.json").read_text())
    task = development_task(identity["task"]["key"])
    if identity["task"] != asdict(task) or identity["sha256"] != task.sha256:
        raise ValueError("Episode task identity mismatch")
    if split not in ("train", "validation"):
        raise ValueError("Only development episodes may enter a training manifest")
    purpose = json.loads((directory / "purpose.json").read_text())
    if purpose != dict(mode="collect", split=split):
        raise ValueError("Collection split differs from the manifest split")
    return dict(
        directory=str(directory.resolve()),
        split=split,
        task_key=task.key,
        task_sha256=task.sha256,
        layout_sha256=json.loads((directory / "initial_layout.json").read_text())[
            "sha256"
        ],
        files={name: digest(directory / name) for name in FILES},
    )


def validate_manifest(manifest):
    """Validate ALL task/split metadata before opening ANY trajectory."""
    if manifest["registry_sha256"] != registry_sha256() or not manifest["episodes"]:
        raise ValueError("Empty or foreign development manifest")
    seen = set()
    for entry in manifest["episodes"]:
        task = development_task(entry["task_key"])
        if entry["task_sha256"] != task.sha256 or entry["split"] not in (
            "train",
            "validation",
        ):
            raise ValueError(
                "Heldout/evaluation data cannot enter training or selection"
            )
        key = (task.key, entry["layout_sha256"])
        if key in seen:
            raise ValueError(
                "Duplicated physical layout, including across development splits"
            )
        seen.add(key)
        if set(entry["files"]) != set(FILES):
            raise ValueError("Incomplete episode manifest")
    return manifest


def load_clips(manifest, split, stride=5, min_target_pixels=24):
    validate_manifest(manifest)
    if split not in ("train", "validation") or type(stride) is not int or stride < 1:
        raise ValueError("Invalid development split or sampling stride")
    clips, labels, vectors, rows = [], [], [], []
    for entry in manifest["episodes"]:
        if entry["split"] != split:
            continue
        directory = Path(entry["directory"])
        # Reconstruct metadata, checking actual purpose and task, not caller labels.
        if episode_entry(directory, split) != entry:
            raise ValueError("Collection identity or retained file hashes changed")
        task = development_task(entry["task_key"])
        condition = TaskCondition(
            **json.loads((directory / "condition.json").read_text())
        )
        vector = condition.validate(task)
        facts = json.loads((directory / "private_truth.json").read_text())
        visibility = json.loads((directory / "visibility.json").read_text())
        from .task_truth import StableTaskTruth

        truth = StableTaskTruth()
        for control, fact in enumerate(facts):
            if (
                fact["task_sha256"] != task.sha256
                or fact["task_key"] != task.key
                or truth.update(control, fact) != fact["strict_complete"]
            ):
                raise ValueError("Invalid offline completion labels")
        with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
            rgb = data["rgb"]
            if (
                rgb.shape != (len(facts), 2, 192, 192, 3)
                or rgb.dtype != np.uint8
                or len(visibility) != len(facts)
            ):
                raise ValueError("RGB/truth/visibility coverage mismatch")
            for control in range(10, len(rgb), stride):
                label = int(facts[control]["strict_complete"])
                # Segmentation is a labeling proxy, never an online feature.
                if max(visibility[control]) < min_target_pixels:
                    label = 2
                clips.append(rgb[[control - 10, control - 5, control]])
                labels.append(label)
                vectors.append(vector)
                rows.append(
                    dict(
                        task_key=task.key,
                        layout=entry["layout_sha256"],
                        control=control,
                    )
                )
    if not clips:
        raise ValueError("No eligible clips in this development split")
    return np.stack(clips), np.asarray(labels), np.stack(vectors), rows
