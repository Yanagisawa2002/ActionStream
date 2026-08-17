"""Run the frozen X-VLA image/state counterfactual on one reset observation.

This is an offline, first-chunk attribution diagnostic.  It does not execute an
episode and therefore cannot establish task success.  Each condition in a
paired seed resets the policy RNG and processor state before inference so the
only intended difference is the selected image/state source.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

REFERENCE_CONDITION = "official_images__official_state"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _require_sha256(path: Path, expected: str, *, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _finite_array(value: Any, shape: tuple[int, ...], *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}, got {array.shape}")
    return array


def validate_unbatched_robot_state(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a JSON-safe unbatched X-VLA robot state."""

    def vector(path: tuple[str, ...], shape: tuple[int, ...]) -> list[Any]:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                raise ValueError(f"robot_state missing {'.'.join(path)}")
            current = current[key]
        return _finite_array(current, shape, label=".".join(path)).tolist()

    return {
        "eef": {
            "mat": vector(("eef", "mat"), (3, 3)),
            "pos": vector(("eef", "pos"), (3,)),
            "quat": vector(("eef", "quat"), (4,)),
        },
        "gripper": {
            "qpos": vector(("gripper", "qpos"), (2,)),
            "qvel": vector(("gripper", "qvel"), (2,)),
        },
        "joints": {
            "pos": vector(("joints", "pos"), (7,)),
            "vel": vector(("joints", "vel"), (7,)),
        },
    }


def unbatch_robot_state(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the single vector-env batch dimension from a robot state."""

    def unbatch(path: tuple[str, ...], shape: tuple[int, ...]) -> list[Any]:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                raise ValueError(f"robot_state missing {'.'.join(path)}")
            current = current[key]
        array = _finite_array(current, (1, *shape), label=".".join(path))
        return array[0].tolist()

    return {
        "eef": {
            "mat": unbatch(("eef", "mat"), (3, 3)),
            "pos": unbatch(("eef", "pos"), (3,)),
            "quat": unbatch(("eef", "quat"), (4,)),
        },
        "gripper": {
            "qpos": unbatch(("gripper", "qpos"), (2,)),
            "qvel": unbatch(("gripper", "qvel"), (2,)),
        },
        "joints": {
            "pos": unbatch(("joints", "pos"), (7,)),
            "vel": unbatch(("joints", "vel"), (7,)),
        },
    }


def batch_robot_state(value: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a validated unbatched robot state to vector-env arrays."""

    state = validate_unbatched_robot_state(value)
    return {
        group: {
            key: np.asarray(child, dtype=np.float64)[None, ...]
            for key, child in children.items()
        }
        for group, children in state.items()
    }


def load_rgb_batch(path: Path) -> np.ndarray:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing non-empty RGB image: {path}")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {rgb.shape}")
    return rgb[None, ...]


def save_rgb_batch(value: Any, path: Path) -> str:
    array = np.asarray(value)
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[-1] != 3:
        raise ValueError(f"Expected [1,H,W,3] RGB batch, got {array.shape}")
    if array.dtype != np.uint8:
        if not np.isfinite(array).all():
            raise ValueError("RGB batch contains non-finite values")
        if float(array.min()) >= 0.0 and float(array.max()) <= 1.0:
            array = np.rint(array * 255.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array[0]).save(path)
    return sha256_file(path)


def build_observation(
    *,
    image: np.ndarray,
    image2: np.ndarray,
    robot_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "pixels": {
            "image": np.asarray(image).copy(),
            "image2": np.asarray(image2).copy(),
        },
        "robot_state": copy.deepcopy(dict(robot_state)),
    }


def paired_action_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    candidate = _finite_array(candidate, (30, 7), label="candidate actions")
    reference = _finite_array(reference, (30, 7), label="reference actions")
    delta = candidate - reference
    return {
        "full_chunk_rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "first_action_xyz_l2": float(np.linalg.norm(delta[0, :3])),
        "first_action_7d_l2": float(np.linalg.norm(delta[0])),
        "chunk_xyz_mean_l2": float(np.mean(np.linalg.norm(delta[:, :3], axis=1))),
        "chunk_7d_mean_l2": float(np.mean(np.linalg.norm(delta, axis=1))),
    }


def action_descriptors(actions: np.ndarray) -> dict[str, float]:
    actions = _finite_array(actions, (30, 7), label="actions")
    gripper = actions[:, 6]
    return {
        "gripper_negative_fraction": float(np.mean(gripper < 0.0)),
        "gripper_positive_fraction": float(np.mean(gripper > 0.0)),
        "xyz_path_length": float(np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1).sum()),
        "xyz_endpoint_displacement": float(np.linalg.norm(actions[-1, :3] - actions[0, :3])),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[str, np.ndarray]] = {}
    for record in records:
        seed = int(record["inference_seed"])
        key = str(record["condition"])
        grouped.setdefault(seed, {})[key] = _finite_array(
            record["actions"], (30, 7), label=f"{seed}/{key} actions"
        )

    effect_map = {
        "image_effect_holding_official_state": "native_images__official_state",
        "state_effect_holding_official_images": "official_images__native_state",
        "joint_shift": "native_images__native_state",
    }
    per_seed: list[dict[str, Any]] = []
    for seed, conditions in sorted(grouped.items()):
        if set(conditions) != {
            REFERENCE_CONDITION,
            "native_images__official_state",
            "official_images__native_state",
            "native_images__native_state",
        }:
            raise ValueError(f"Inference seed {seed} does not contain the frozen four conditions")
        reference = conditions[REFERENCE_CONDITION]
        effects = {
            name: paired_action_metrics(conditions[key], reference)
            for name, key in effect_map.items()
        }
        per_seed.append({"inference_seed": seed, "effects": effects})

    aggregate: dict[str, Any] = {}
    for effect_name in effect_map:
        aggregate[effect_name] = {}
        metric_names = next(
            item["effects"][effect_name].keys() for item in per_seed
        )
        for metric_name in metric_names:
            values = np.asarray(
                [item["effects"][effect_name][metric_name] for item in per_seed],
                dtype=np.float64,
            )
            aggregate[effect_name][metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
    return {"per_seed": per_seed, "aggregate": aggregate}


def _reset_inference_state(backend: Any, seed: int) -> None:
    backend._set_seed(int(seed))
    backend.policy.reset()
    for pipeline in (
        backend.env_preprocessor,
        backend.preprocessor,
        backend.postprocessor,
        backend.env_postprocessor,
    ):
        pipeline.reset()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    if config.get("freeze_status") != "frozen_before_formal_inference":
        raise ValueError("Counterfactual config is not marked frozen before inference")
    keys = [item["key"] for item in config["conditions"]]
    if set(keys) != {
        REFERENCE_CONDITION,
        "native_images__official_state",
        "official_images__native_state",
        "native_images__native_state",
    }:
        raise ValueError("Config must contain exactly the frozen four conditions")
    for item in config["paired_schedule"]:
        if set(item["condition_order"]) != set(keys):
            raise ValueError("Every paired schedule row must contain all four conditions")
    return config


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _report(summary: Mapping[str, Any]) -> str:
    effects = summary["analysis"]["aggregate"]
    lines = [
        "# X-VLA native-Isaac input counterfactual",
        "",
        "Status: completed offline first-chunk diagnostic; this is not an episode success result.",
        "",
        "| Frozen comparison | Full chunk RMSE | First action XYZ L2 | First action 7D L2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "image_effect_holding_official_state": "Native vs official images, official state held",
        "state_effect_holding_official_images": "Native vs official state, official images held",
        "joint_shift": "Fully native vs fully official",
    }
    for key, label in labels.items():
        item = effects[key]
        lines.append(
            "| " + label + " | "
            f"{item['full_chunk_rmse']['mean']:.6f} | "
            f"{item['first_action_xyz_l2']['mean']:.6f} | "
            f"{item['first_action_7d_l2']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "All values are paired across the three frozen inference seeds. Interpret larger "
            "action-space displacement as stronger input sensitivity, not as proof of task-level "
            "causality. The exact actions and per-seed values are in `summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--native-summary", type=Path, required=True)
    parser.add_argument("--native-image", type=Path, required=True)
    parser.add_argument("--native-image2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    # Keep the pure validation/analysis helpers importable on machines without
    # the pinned LeRobot runtime.  Formal inference still imports the exact
    # backend here and fails early if that environment is unavailable.
    from actionstream.current_baselines import CurrentLeRobotBackend, load_protocol

    args = _parser().parse_args()
    config = _load_config(args.config.resolve())
    config_sha256 = sha256_file(args.config.resolve())
    _require_sha256(
        args.protocol.resolve(), config["policy"]["protocol_sha256"], label="protocol"
    )
    native = config["native_observation"]
    native_summary_sha256 = _require_sha256(
        args.native_summary.resolve(), native["summary_sha256"], label="native summary"
    )
    native_image_sha256 = _require_sha256(
        args.native_image.resolve(), native["image_sha256"], label="native image"
    )
    native_image2_sha256 = _require_sha256(
        args.native_image2.resolve(), native["image2_sha256"], label="native image2"
    )
    native_summary = json.loads(args.native_summary.read_text(encoding="utf-8"))
    if native_summary.get("instruction") != config["official_observation"]["instruction"]:
        raise ValueError("Native and frozen official instructions differ")
    request_records = native_summary.get("request_records", [])
    if not request_records:
        raise ValueError("Native summary has no request_records")
    native_state = validate_unbatched_robot_state(request_records[0]["robot_state"])
    native_images = {
        "image": load_rgb_batch(args.native_image.resolve()),
        "image2": load_rgb_batch(args.native_image2.resolve()),
    }

    protocol = load_protocol(args.protocol.resolve())
    policy_key = config["policy"]["key"]
    spec = protocol.models[policy_key]
    official = config["official_observation"]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    backend = CurrentLeRobotBackend(
        spec=spec,
        task_ids=[int(official["task_id"])],
        suite=str(official["suite"]),
        episode_length=800,
        seed=int(official["environment_seed"]),
        device=str(config["policy"]["device"]),
    )
    try:
        official_observation, _info, instruction = backend.reset_episode(
            task_id=int(official["task_id"]),
            seed=int(official["environment_seed"]),
            initial_state_index=int(official["initial_state_index"]),
        )
        if instruction != official["instruction"]:
            raise ValueError(
                f"Frozen instruction mismatch: expected {official['instruction']!r}, got {instruction!r}"
            )
        official_images = {
            "image": np.asarray(official_observation["pixels"]["image"]).copy(),
            "image2": np.asarray(official_observation["pixels"]["image2"]).copy(),
        }
        official_state = unbatch_robot_state(official_observation["robot_state"])

        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        official_image_sha256 = save_rgb_batch(
            official_images["image"], output / "official_image.png"
        )
        official_image2_sha256 = save_rgb_batch(
            official_images["image2"], output / "official_image2.png"
        )

        conditions = {item["key"]: item for item in config["conditions"]}
        records: list[dict[str, Any]] = []
        for schedule in config["paired_schedule"]:
            inference_seed = int(schedule["inference_seed"])
            for condition_key in schedule["condition_order"]:
                condition = conditions[condition_key]
                image_source = str(condition["images"])
                state_source = str(condition["robot_state"])
                chosen_images = official_images if image_source == "official" else native_images
                chosen_state = official_state if state_source == "official" else native_state
                observation = build_observation(
                    image=chosen_images["image"],
                    image2=chosen_images["image2"],
                    robot_state=batch_robot_state(chosen_state),
                )
                _reset_inference_state(backend, inference_seed)
                inference = backend.infer_action_chunk(observation, instruction)
                actions = np.asarray(inference.actions, dtype=np.float32)
                record = {
                    "inference_seed": inference_seed,
                    "condition": condition_key,
                    "image_source": image_source,
                    "robot_state_source": state_source,
                    "actions": actions.tolist(),
                    "actions_sha256": _sha256_bytes(actions.tobytes(order="C")),
                    "raw_actions_sha256": _sha256_bytes(
                        np.asarray(inference.raw_actions).tobytes(order="C")
                    ),
                    "model_latency_seconds": inference.model_latency_seconds,
                    "raw_shape": list(inference.raw_shape),
                    "raw_dtype": inference.raw_dtype,
                    "descriptors": action_descriptors(actions),
                }
                records.append(record)

        analysis = summarize_records(records)
        summary = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "evidence_class": "offline_first_chunk_input_counterfactual",
            "task_success_evidence": False,
            "config_sha256": config_sha256,
            "policy": config["policy"],
            "instruction": instruction,
            "official_observation": {
                "image_sha256": official_image_sha256,
                "image2_sha256": official_image2_sha256,
                "robot_state": official_state,
                "robot_state_sha256": _canonical_json_sha256(official_state),
            },
            "native_observation": {
                "summary_sha256": native_summary_sha256,
                "image_sha256": native_image_sha256,
                "image2_sha256": native_image2_sha256,
                "robot_state": native_state,
                "robot_state_sha256": _canonical_json_sha256(native_state),
            },
            "records": records,
            "analysis": analysis,
            "peak_cuda_memory_mib": backend.peak_cuda_memory_mib,
            "limitations": [
                "Only the first action chunk is inferred; no environment action is executed.",
                "Action displacement measures input sensitivity, not task-level causal success.",
                "The native image and state come from one development smoke episode.",
            ],
        }
        _write_json(output / "summary.json", summary)
        (output / "report.md").write_text(_report(summary), encoding="utf-8")
        print(
            json.dumps(
                {
                    "event": "completed",
                    "output": str(output),
                    "summary_sha256": sha256_file(output / "summary.json"),
                    "peak_cuda_memory_mib": backend.peak_cuda_memory_mib,
                },
                sort_keys=True,
            )
        )
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
