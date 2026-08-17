"""Command-line orchestration for reference and ROS/C++ M7 runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from .analysis import analyze_manifest
from .faults import (
    FaultProfile,
    FaultTrace,
    generate_fault_trace,
    load_fault_trace,
    resolve_profile,
    write_fault_trace,
)
from .figures import generate_figures
from .reference_benchmark import Strategy, run_reference_episode
from .replay import (
    validate_episode_log,
    validate_manifest,
    write_summary_from_event_log,
)
from .schema import (
    MILESTONE,
    SCHEMA_VERSION,
    canonical_sha256,
    read_json,
    write_json_atomic,
)


STRATEGIES = tuple(strategy.value for strategy in Strategy)


def _csv(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated value")
    return result


def _load_seeds(path: Path | str) -> list[int]:
    payload = read_json(path)
    seeds = [int(seed) for seed in payload.get("seeds", [])]
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError(f"{path}: seeds must be unique non-negative integers")
    return seeds


def _profiles(values: Iterable[str]) -> list[FaultProfile]:
    profiles = [resolve_profile(value) for value in values]
    identifiers = [profile.profile_id for profile in profiles]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("profile identifiers must be unique")
    return profiles


def _strategies(values: Iterable[str]) -> list[str]:
    result = list(values)
    unknown = sorted(set(result) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"unknown strategies: {unknown}")
    if len(result) != len(set(result)):
        raise ValueError("strategies must be unique")
    return result


def _validate_split(split: str, seeds: list[int], comparison: list[int] | None) -> None:
    if split == "development" and not 1 <= len(seeds) <= 10:
        raise ValueError("development runs require 1 to 10 seeds")
    if split == "holdout" and len(seeds) < 30:
        raise ValueError("holdout runs require at least 30 seeds")
    if comparison is not None and set(seeds) & set(comparison):
        raise ValueError("seed file overlaps comparison split")


def _enforce_holdout_lock(
    root: Path,
    *,
    seeds: list[int],
    profiles: list[FaultProfile],
    request_interval_steps: int,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "frozen_before_first_holdout_result": True,
        "seeds": seeds,
        "profiles": [asdict(profile) for profile in profiles],
        "protocol": {
            "horizon": 30,
            "action_dimension": 7,
            "control_frequency_hz": 20.0,
            "max_steps": 180,
            "request_interval_steps": request_interval_steps,
            "strategies": list(STRATEGIES),
        },
    }
    lock = root / "HOLDOUT_PROTOCOL_LOCK.json"
    if lock.exists():
        existing = read_json(lock)
        if canonical_sha256(existing) != canonical_sha256(payload):
            raise RuntimeError(f"holdout protocol is already locked with different values: {lock}")
    else:
        write_json_atomic(lock, payload)


def _trace_path(root: Path, profile: FaultProfile, seed: int) -> Path:
    return root / "traces" / profile.profile_id / f"seed_{seed}.trace.json"


def _ensure_trace(
    root: Path,
    profile: FaultProfile,
    seed: int,
    *,
    request_count: int,
) -> tuple[Path, FaultTrace]:
    path = _trace_path(root, profile, seed)
    expected = generate_fault_trace(profile, seed=seed, request_count=request_count)
    if path.exists():
        loaded = load_fault_trace(path)
        if loaded.sha256 != expected.sha256:
            raise RuntimeError(f"existing trace differs from frozen generation: {path}")
        return path, loaded
    write_fault_trace(path, expected)
    return path, expected


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _episode_paths(root: Path, profile_id: str, seed: int, strategy: str) -> tuple[Path, Path]:
    directory = root / "episodes" / profile_id / f"seed_{seed}"
    return (
        directory / f"{strategy}.events.jsonl",
        directory / f"{strategy}.summary.json",
    )


def _write_manifest(
    root: Path,
    *,
    split: str,
    evidence_class: str,
    episodes: list[dict[str, str]],
    profiles: list[FaultProfile],
    seeds: list[int],
) -> Path:
    path = root / "manifest.json"
    write_json_atomic(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "milestone": MILESTONE,
            "split": split,
            "evidence_class": evidence_class,
            "headline_eligible": evidence_class in {"ros_cpp_test_plant", "ros_cpp_isaac_sim"},
            "profiles": [asdict(profile) for profile in profiles],
            "seeds": seeds,
            "episodes": episodes,
        },
    )
    return path


def _prepare_matrix(args) -> tuple[Path, list[int], list[FaultProfile], list[str]]:
    root = Path(args.output_root).resolve()
    seeds = _load_seeds(args.seed_file)
    comparison = _load_seeds(args.comparison_seed_file) if args.comparison_seed_file else None
    _validate_split(args.split, seeds, comparison)
    profiles = _profiles(args.profiles)
    strategies = _strategies(args.strategies)
    if args.split == "holdout":
        if set(strategies) != set(STRATEGIES):
            raise ValueError("formal holdout runs require all three strategies")
        _enforce_holdout_lock(
            root,
            seeds=seeds,
            profiles=profiles,
            request_interval_steps=args.request_interval_steps,
        )
    return root, seeds, profiles, strategies


def command_generate_traces(args) -> int:
    root = Path(args.output_root).resolve()
    seeds = _load_seeds(args.seed_file)
    profiles = _profiles(args.profiles)
    paths = []
    for profile in profiles:
        for seed in seeds:
            path, trace = _ensure_trace(root, profile, seed, request_count=args.request_count)
            paths.append({"path": str(path), "sha256": trace.sha256})
    print(json.dumps({"trace_count": len(paths), "traces": paths}, indent=2))
    return 0


def command_reference_run(args) -> int:
    root, seeds, profiles, strategies = _prepare_matrix(args)
    episodes: list[dict[str, str]] = []
    for profile in profiles:
        for seed in seeds:
            trace_path, trace = _ensure_trace(
                root, profile, seed, request_count=args.request_count
            )
            for strategy in strategies:
                event_path, summary_path = _episode_paths(root, profile.profile_id, seed, strategy)
                if (event_path.exists() or summary_path.exists()) and not args.overwrite:
                    raise FileExistsError(
                        f"refusing to overwrite episode for {strategy}: {event_path}"
                    )
                run_reference_episode(
                    strategy=strategy,
                    trace=trace,
                    event_log_path=event_path,
                    summary_path=summary_path,
                    max_steps=args.max_steps,
                    request_interval_steps=args.request_interval_steps,
                )
                audit = validate_episode_log(event_path, summary_path)
                if not audit["passed"]:
                    raise RuntimeError(f"reference replay validation failed: {event_path}")
                episodes.append(
                    {
                        "event_log": _relative(event_path, root),
                        "summary": _relative(summary_path, root),
                        "trace": _relative(trace_path, root),
                    }
                )
    manifest = _write_manifest(
        root,
        split=args.split,
        evidence_class="deterministic_ros_test_plant_reference",
        episodes=episodes,
        profiles=profiles,
        seeds=seeds,
    )
    print(str(manifest))
    return 0


def _run_ros_episode(
    *,
    root: Path,
    profile: FaultProfile,
    seed: int,
    strategy: str,
    request_count: int,
    max_steps: int,
    request_interval_steps: int,
    timeout_seconds: float,
    ros2_executable: str,
) -> dict[str, str]:
    trace_path, trace = _ensure_trace(root, profile, seed, request_count=request_count)
    event_path, summary_path = _episode_paths(root, profile.profile_id, seed, strategy)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    episode_id = f"m7-ros-{profile.profile_id}-{seed}-{strategy}"
    command = [
        ros2_executable,
        "launch",
        "action_stream_benchmark",
        "test_plant_episode.launch.py",
        f"strategy:={strategy}",
        f"seed:={seed}",
        f"episode_id:={episode_id}",
        f"profile_id:={profile.profile_id}",
        f"trace_file:={trace_path}",
        f"trace_sha256:={trace.sha256}",
        f"output_path:={event_path}",
        f"max_steps:={max_steps}",
        f"request_interval_steps:={request_interval_steps}",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    ros_log = event_path.with_suffix(".ros.log")
    ros_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"ROS episode failed with exit code {completed.returncode}; see {ros_log}"
        )
    if not event_path.exists():
        raise RuntimeError(f"ROS episode produced no atomic event log: {event_path}")
    write_summary_from_event_log(
        event_path,
        summary_path,
        profile_id=profile.profile_id,
        profile=asdict(profile),
        seed=seed,
        strategy=strategy,
        trace_sha256=trace.sha256,
        evidence_class="ros_cpp_test_plant",
        request_interval_steps=request_interval_steps,
    )
    audit = validate_episode_log(event_path, summary_path)
    audit_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".audit.json"))
    write_json_atomic(audit_path, audit)
    if not audit["passed"]:
        raise RuntimeError(f"ROS replay validation failed: {audit_path}")
    return {
        "event_log": _relative(event_path, root),
        "summary": _relative(summary_path, root),
        "trace": _relative(trace_path, root),
        "audit": _relative(audit_path, root),
    }


def command_ros_episode(args) -> int:
    root = Path(args.output_root).resolve()
    profile = resolve_profile(args.profile)
    entry = _run_ros_episode(
        root=root,
        profile=profile,
        seed=args.seed,
        strategy=args.strategy,
        request_count=args.request_count,
        max_steps=args.max_steps,
        request_interval_steps=args.request_interval_steps,
        timeout_seconds=args.timeout_seconds,
        ros2_executable=args.ros2_executable,
    )
    print(json.dumps(entry, indent=2))
    return 0


def command_ros_run(args) -> int:
    root, seeds, profiles, strategies = _prepare_matrix(args)
    episodes: list[dict[str, str]] = []
    for profile in profiles:
        for seed in seeds:
            for strategy in strategies:
                event_path, summary_path = _episode_paths(root, profile.profile_id, seed, strategy)
                if (event_path.exists() or summary_path.exists()) and not args.overwrite:
                    raise FileExistsError(f"refusing to overwrite ROS episode: {event_path}")
                episodes.append(
                    _run_ros_episode(
                        root=root,
                        profile=profile,
                        seed=seed,
                        strategy=strategy,
                        request_count=args.request_count,
                        max_steps=args.max_steps,
                        request_interval_steps=args.request_interval_steps,
                        timeout_seconds=args.timeout_seconds,
                        ros2_executable=args.ros2_executable,
                    )
                )
    manifest = _write_manifest(
        root,
        split=args.split,
        evidence_class="ros_cpp_test_plant",
        episodes=episodes,
        profiles=profiles,
        seeds=seeds,
    )
    print(str(manifest))
    return 0


def command_validate(args) -> int:
    result = validate_manifest(args.manifest, output_path=args.output)
    print(json.dumps({"passed": result["passed"], "episode_count": result["episode_count"]}))
    return 0 if result["passed"] else 1


def command_analyze(args) -> int:
    result = analyze_manifest(
        args.manifest,
        output_path=args.output,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(json.dumps(result["primary_numerical_gate"], indent=2))
    return 0


def command_figures(args) -> int:
    paths = generate_figures(
        analysis_path=args.analysis,
        example_event_log=args.example_event_log,
        output_dir=args.output_dir,
    )
    print(json.dumps(paths, indent=2))
    return 0


def _matrix_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--comparison-seed-file", type=Path)
    parser.add_argument("--profiles", type=_csv, default=["profile_a", "profile_b"])
    parser.add_argument("--strategies", type=_csv, default=list(STRATEGIES))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request-count", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--request-interval-steps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="action-stream-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    traces = subparsers.add_parser("generate-traces", help="materialize immutable paired traces")
    traces.add_argument("--seed-file", type=Path, required=True)
    traces.add_argument("--profiles", type=_csv, default=["profile_a", "profile_b"])
    traces.add_argument("--output-root", type=Path, required=True)
    traces.add_argument("--request-count", type=int, default=512)
    traces.set_defaults(handler=command_generate_traces)

    reference = subparsers.add_parser(
        "reference-run",
        help="run the ROS-free deterministic calibration/reference model",
    )
    _matrix_arguments(reference)
    reference.set_defaults(handler=command_reference_run)

    ros_episode = subparsers.add_parser(
        "ros-episode",
        help="smoke one episode through ROS topics and the compiled C++ executor",
    )
    ros_episode.add_argument("--profile", default="profile_a")
    ros_episode.add_argument("--seed", type=int, default=2026080300)
    ros_episode.add_argument("--strategy", choices=STRATEGIES, default="aligned_async")
    ros_episode.add_argument("--output-root", type=Path, required=True)
    ros_episode.add_argument("--request-count", type=int, default=512)
    ros_episode.add_argument("--max-steps", type=int, default=180)
    ros_episode.add_argument("--request-interval-steps", type=int, default=10)
    ros_episode.add_argument("--timeout-seconds", type=float, default=120.0)
    ros_episode.add_argument("--ros2-executable", default="ros2")
    ros_episode.set_defaults(handler=command_ros_episode)

    ros_run = subparsers.add_parser(
        "ros-run",
        help="run a paired matrix through ROS topics and the compiled C++ executor",
    )
    _matrix_arguments(ros_run)
    ros_run.add_argument("--timeout-seconds", type=float, default=120.0)
    ros_run.add_argument("--ros2-executable", default="ros2")
    ros_run.set_defaults(handler=command_ros_run)

    validate = subparsers.add_parser("validate", help="independently replay every episode")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(handler=command_validate)

    analyze = subparsers.add_parser("analyze", help="compute paired seed-level statistics")
    analyze.add_argument("--manifest", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--bootstrap-resamples", type=int, default=20_000)
    analyze.set_defaults(handler=command_analyze)

    figures = subparsers.add_parser("figures", help="render the three artifact-backed figures")
    figures.add_argument("--analysis", type=Path, required=True)
    figures.add_argument("--example-event-log", type=Path, required=True)
    figures.add_argument("--output-dir", type=Path, required=True)
    figures.set_defaults(handler=command_figures)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
