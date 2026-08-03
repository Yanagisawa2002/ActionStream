"""Build one persistent-Isaac batch matrix without creating result summaries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from .m8_faults import generate_fault_trace, load_profile, write_fault_trace
from .m8_protocol import (
    M8_MILESTONE,
    M8_SCHEMA_VERSION,
    NATIVE_ISAAC_EVIDENCE_CLASS,
    PROFILE_STRATEGIES,
    load_protocol,
    load_seed_file,
    sha256_file,
    validate_candidate_contract,
    validate_freeze_manifest,
)
from .m8_scenario import ScenarioSpace, generate_scenario, write_scenario
from .schema import canonical_sha256, read_json, write_json_atomic


def _reference(path: Path, *, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def build_matrix_manifest(
    *,
    repository_root: Path | str,
    freeze_manifest_path: Path | str | None,
    candidate_protocol_path: Path | str | None,
    seed_path: Path | str,
    profile_paths: Iterable[Path | str],
    output_root: Path | str,
    manifest_path: Path | str,
    split: str,
    request_count: int = 512,
    strategy_filter: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Materialize scenarios/traces and a batch manifest, but never fake results."""

    if split not in {"baseline_gate", "development", "frozen_holdout"}:
        raise ValueError(f"unsupported M8 split: {split}")
    if request_count < 32:
        raise ValueError("fault traces must cover at least 32 requests")
    selected_strategies = tuple(dict.fromkeys(strategy_filter or ()))
    known_strategies = {strategy for values in PROFILE_STRATEGIES.values() for strategy in values}
    if selected_strategies and split != "development":
        raise ValueError("a strategy filter is permitted only for non-headline development")
    if any(strategy not in known_strategies for strategy in selected_strategies):
        raise ValueError(f"unknown development strategy filter: {selected_strategies}")
    root = Path(repository_root).resolve()
    freeze_path: Path | None = None
    freeze: dict[str, Any] | None = None
    candidate_path: Path | None = None
    candidate: dict[str, Any] | None = None
    matrix_protocol: dict[str, Any]
    if split == "frozen_holdout":
        if freeze_manifest_path is None:
            raise ValueError("frozen holdout matrix requires --freeze-manifest")
        freeze_path = Path(freeze_manifest_path).resolve()
        freeze_audit = validate_freeze_manifest(freeze_path, repository_root=root)
        if not freeze_audit["passed"]:
            raise ValueError(f"cannot build matrix from invalid freeze: {freeze_audit['errors']}")
        freeze = freeze_audit["manifest"]
        protocol_record = next(
            (
                record
                for record in freeze.get("inputs", ())
                if record.get("role") == "protocol"
            ),
            None,
        )
        if not isinstance(protocol_record, dict):
            raise ValueError("freeze manifest does not bind its finalized protocol input")
        frozen_protocol_path = root / str(protocol_record.get("path", ""))
        matrix_protocol = load_protocol(frozen_protocol_path)
        if canonical_sha256(matrix_protocol) != freeze.get("protocol_sha256"):
            raise ValueError("freeze protocol input does not match its canonical protocol hash")
    else:
        if candidate_protocol_path is None:
            raise ValueError("baseline/development matrix requires --candidate-protocol")
        candidate_path = Path(candidate_protocol_path).resolve()
        try:
            candidate_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("candidate protocol must be inside the repository") from exc
        candidate = load_protocol(candidate_path)
        if (
            candidate.get("protocol_status")
            != "preregistered_candidate_pending_native_baseline_and_development"
            or candidate.get("holdout_freeze_status") != "not_yet_frozen"
        ):
            raise ValueError("candidate protocol must remain explicitly pending before holdout")
        validate_candidate_contract(candidate, repository_root=root)
        matrix_protocol = candidate
    seeds = load_seed_file(seed_path)
    if split == "frozen_holdout":
        assert freeze is not None
        try:
            relative_seed = Path(seed_path).resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("holdout seed file must be inside the frozen repository") from exc
        frozen_seed = next(
            (
                record
                for record in freeze.get("inputs", [])
                if record.get("role") == "holdout_seeds"
            ),
            None,
        )
        if (
            frozen_seed is None
            or frozen_seed.get("path") != relative_seed
            or frozen_seed.get("sha256") != sha256_file(seed_path)
        ):
            raise ValueError("matrix holdout seed path/hash does not match frozen holdout_seeds role")
    if split == "baseline_gate" and len(seeds) < 20:
        raise ValueError("native simulator baseline gate requires at least 20 seeds")
    if split == "development" and not 1 <= len(seeds) <= 12:
        raise ValueError("development matrices require 1 to 12 seeds")
    if split == "frozen_holdout" and len(seeds) < 40:
        raise ValueError("frozen holdout requires at least 40 seeds")
    profiles = {}
    profile_records = {}
    for value in profile_paths:
        profile_path = Path(value).resolve()
        payload = read_json(profile_path)
        profile = load_profile(profile_path)
        expected_strategies = PROFILE_STRATEGIES.get(profile.profile_id)
        if expected_strategies is None:
            raise ValueError(f"unknown M8 profile: {profile.profile_id}")
        if tuple(payload.get("profile", {}).get("strategies", ())) != expected_strategies:
            raise ValueError(f"strategy set drift in {profile_path}")
        if candidate is not None:
            candidate_profile = candidate["profiles"][profile.profile_id]
            candidate_profile_path = (
                root / str(candidate_profile["profile_file"])
            ).resolve()
            if (
                candidate_profile_path != profile_path
                or candidate_profile["profile_sha256"] != sha256_file(profile_path)
            ):
                raise ValueError(
                    f"candidate protocol does not bind profile bytes: {profile.profile_id}"
                )
        if profile.profile_id in profiles:
            raise ValueError(f"duplicate profile: {profile.profile_id}")
        if split == "frozen_holdout":
            assert freeze is not None
            try:
                relative_profile = profile_path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError("holdout profile must be inside the frozen repository") from exc
            frozen_profile = next(
                (
                    record
                    for record in freeze.get("inputs", [])
                    if record.get("role") == f"profile:{profile.profile_id}"
                ),
                None,
            )
            if (
                frozen_profile is None
                or frozen_profile.get("path") != relative_profile
                or frozen_profile.get("sha256") != sha256_file(profile_path)
            ):
                raise ValueError(
                    f"matrix profile path/hash does not match freeze role: {profile.profile_id}"
                )
        profiles[profile.profile_id] = profile
        profile_records[profile.profile_id] = {
            "path": _reference(profile_path, base=Path(manifest_path).resolve().parent),
            "sha256": sha256_file(profile_path),
        }
    if split == "baseline_gate" and set(profiles) != {"profile_0_sanity"}:
        raise ValueError("baseline_gate matrix requires exactly Profile 0 sync")
    if split == "development" and not set(profiles).issubset(PROFILE_STRATEGIES):
        raise ValueError("development matrix includes an unknown candidate profile")
    if split == "frozen_holdout" and set(profiles) != set(PROFILE_STRATEGIES):
        raise ValueError(f"frozen holdout requires exactly profiles {sorted(PROFILE_STRATEGIES)}")
    strategies_by_profile = {
        profile_id: tuple(
            strategy
            for strategy in PROFILE_STRATEGIES[profile_id]
            if not selected_strategies or strategy in selected_strategies
        )
        for profile_id in profiles
    }
    if any(not strategies for strategies in strategies_by_profile.values()):
        raise ValueError("development strategy filter leaves a selected profile with no method")

    destination = Path(output_root).resolve()
    matrix_path = Path(manifest_path).resolve()
    if matrix_path.exists():
        raise FileExistsError(f"refusing to overwrite matrix manifest: {matrix_path}")
    planned_batch_paths = {
        (profile_id, strategy): matrix_path.parent / f"batch_{profile_id}_{strategy}.json"
        for profile_id in profiles
        for strategy in strategies_by_profile[profile_id]
    }
    existing_batches = [str(path) for path in planned_batch_paths.values() if path.exists()]
    if existing_batches:
        raise FileExistsError(f"refusing to overwrite batch manifests: {existing_batches}")
    destination.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    for directory in ("scenarios", "fault_traces", "events", "summaries"):
        (destination / directory).mkdir(parents=True, exist_ok=True)

    task_contract = matrix_protocol["task"]
    object_z_m = float(task_contract["cube_side_m"]) / 2.0
    scenario_space = ScenarioSpace(
        object_x_range=tuple(float(value) for value in task_contract["object_x_range_m"]),
        object_y_range=tuple(float(value) for value in task_contract["object_y_range_m"]),
        object_z_m=object_z_m,
        zone_negative_y_xyz=(
            float(task_contract["zone_a_xy_m"][0]),
            float(task_contract["zone_a_xy_m"][1]),
            object_z_m,
        ),
        zone_positive_y_xyz=(
            float(task_contract["zone_b_xy_m"][0]),
            float(task_contract["zone_b_xy_m"][1]),
            object_z_m,
        ),
        switch_steps=tuple(int(value) for value in task_contract["switch_steps"]),
    )
    scenario_space.validate()
    scenario_records: dict[int, tuple[Path, str]] = {}
    for seed in seeds:
        scenario = generate_scenario(seed, space=scenario_space)
        scenario_path = destination / "scenarios" / f"seed_{seed}.json"
        if scenario_path.exists():
            raise FileExistsError(f"refusing to overwrite scenario: {scenario_path}")
        write_scenario(scenario_path, scenario)
        scenario_records[seed] = (scenario_path, scenario.sha256)

    episodes = []
    trace_records = []
    for profile_id in PROFILE_STRATEGIES:
        if profile_id not in profiles:
            continue
        profile = profiles[profile_id]
        for seed in seeds:
            trace = generate_fault_trace(profile, seed=seed, request_count=request_count)
            trace_path = destination / "fault_traces" / f"{profile_id}_seed_{seed}.json"
            if trace_path.exists():
                raise FileExistsError(f"refusing to overwrite fault trace: {trace_path}")
            write_fault_trace(trace_path, trace)
            trace_records.append(
                {
                    "profile_id": profile_id,
                    "seed": seed,
                    "path": _reference(trace_path, base=matrix_path.parent),
                    "trace_sha256": trace.sha256,
                }
            )
            scenario_path, scenario_sha256 = scenario_records[seed]
            for strategy in strategies_by_profile[profile_id]:
                episode_id = f"m8-{split}-{profile_id}-{seed}-{strategy}"
                event_path = destination / "events" / f"{episode_id}.jsonl"
                summary_path = destination / "summaries" / f"{episode_id}.json"
                if event_path.exists() or summary_path.exists():
                    raise FileExistsError(f"refusing to overwrite episode output: {episode_id}")
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "seed": seed,
                        "strategy": strategy,
                        "profile_id": profile_id,
                        "fault_trace_file": _reference(trace_path, base=matrix_path.parent),
                        "fault_trace_sha256": trace.sha256,
                        "scenario_file": _reference(scenario_path, base=matrix_path.parent),
                        "scenario_sha256": scenario_sha256,
                        "event_log_path": _reference(event_path, base=matrix_path.parent),
                        "summary_path": _reference(summary_path, base=matrix_path.parent),
                    }
                )

    common: dict[str, Any] = {
        "schema_version": M8_SCHEMA_VERSION,
        "milestone": M8_MILESTONE,
        "evidence_class": NATIVE_ISAAC_EVIDENCE_CLASS,
        "split": split,
        "headline_eligible": split == "frozen_holdout",
        "persistent_isaac_process_required": True,
        "seed_file": _reference(Path(seed_path), base=matrix_path.parent),
        "seed_file_sha256": sha256_file(seed_path),
        "seed_count": len(seeds),
        "profiles": profile_records,
        "fault_traces": trace_records,
    }
    if selected_strategies:
        common["development_strategy_filter"] = list(selected_strategies)
    expected_episode_count = len(seeds) * sum(
        len(strategies_by_profile[profile_id]) for profile_id in profiles
    )
    if len(episodes) != expected_episode_count:
        raise RuntimeError("internal split/method episode-count mismatch")
    if split == "frozen_holdout":
        assert freeze_path is not None and freeze is not None
        common.update(
            {
                "freeze_manifest": _reference(freeze_path, base=matrix_path.parent),
                "freeze_sha256": freeze["freeze_sha256"],
                "protocol_sha256": freeze["protocol_sha256"],
                "holdout_freeze_status": "frozen",
                "frozen_before_first_holdout_result": True,
            }
        )
    else:
        assert candidate_path is not None and candidate is not None
        common.update(
            {
                "candidate_manifest": _reference(candidate_path, base=matrix_path.parent),
                "protocol_sha256": canonical_sha256(candidate),
                "holdout_freeze_status": "pending",
                "frozen_before_first_holdout_result": False,
            }
        )
    batch_records = []
    for (profile_id, strategy), batch_path in planned_batch_paths.items():
        batch_episodes = [
            episode
            for episode in episodes
            if episode["profile_id"] == profile_id and episode["strategy"] == strategy
        ]
        if len(batch_episodes) != len(seeds):
            raise RuntimeError("persistent batch must contain exactly one episode per seed")
        batch = {
            **common,
            "native_results_status_at_creation": "not_run",
            "batch_strategy": strategy,
            "batch_profile_id": profile_id,
            "expected_episode_count": len(seeds),
            "episodes": batch_episodes,
        }
        write_json_atomic(batch_path, batch)
        batch_records.append(
            {
                "profile_id": profile_id,
                "strategy": strategy,
                "path": _reference(batch_path, base=matrix_path.parent),
                "episode_count": len(batch_episodes),
            }
        )
    manifest = {
        **common,
        "native_results_status_at_creation": "not_run",
        "persistent_batch_count": len(batch_records),
        "batch_manifests": batch_records,
        "expected_episode_count": expected_episode_count,
        "episodes": episodes,
    }
    write_json_atomic(matrix_path, manifest)
    return manifest
