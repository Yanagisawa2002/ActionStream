"""Frozen, seed-controlled paired latency and fault traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Final, Iterable, Mapping

from .schema import (
    MILESTONE,
    SCHEMA_VERSION,
    canonical_sha256,
    read_json,
    write_json_atomic,
)


@dataclass(frozen=True, slots=True)
class FaultProfile:
    profile_id: str
    base_latency_ms: int
    jitter_ms: int
    drop_probability: float
    extra_delay_probability: float
    extra_delay_ms: int
    duplicate_probability: float
    communication_pause_probability: float
    communication_pause_ms: int
    frozen: bool = True

    def validate(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        for name, value in (
            ("base_latency_ms", self.base_latency_ms),
            ("jitter_ms", self.jitter_ms),
            ("extra_delay_ms", self.extra_delay_ms),
            ("communication_pause_ms", self.communication_pause_ms),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.jitter_ms > self.base_latency_ms:
            raise ValueError("jitter cannot make base latency negative")
        for name, value in (
            ("drop_probability", self.drop_probability),
            ("extra_delay_probability", self.extra_delay_probability),
            ("duplicate_probability", self.duplicate_probability),
            ("communication_pause_probability", self.communication_pause_probability),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not self.frozen:
            raise ValueError("benchmark profiles must be frozen before trace generation")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FaultProfile":
        allowed = {field for field in cls.__dataclass_fields__}
        payload = {key: child for key, child in value.items() if key in allowed}
        profile = cls(**payload)
        profile.validate()
        return profile


PROFILE_A: Final = FaultProfile(
    profile_id="profile_a",
    base_latency_ms=950,
    jitter_ms=0,
    drop_probability=0.0,
    extra_delay_probability=0.0,
    extra_delay_ms=0,
    duplicate_probability=0.0,
    communication_pause_probability=0.0,
    communication_pause_ms=0,
)

PROFILE_B: Final = FaultProfile(
    profile_id="profile_b",
    base_latency_ms=900,
    jitter_ms=250,
    drop_probability=0.05,
    extra_delay_probability=0.12,
    extra_delay_ms=1000,
    duplicate_probability=0.02,
    communication_pause_probability=0.05,
    communication_pause_ms=300,
)

BUILTIN_PROFILES: Final = {profile.profile_id: profile for profile in (PROFILE_A, PROFILE_B)}
DEVELOPMENT_SEEDS: Final[tuple[int, ...]] = tuple(range(2026080300, 2026080308))
HOLDOUT_SEEDS: Final[tuple[int, ...]] = tuple(range(2026080400, 2026080436))


@dataclass(frozen=True, slots=True)
class FaultTraceEntry:
    request_ordinal: int
    latency_ms: int
    jitter_ms: int
    dropped: bool
    extra_delay_ms: int
    duplicate_count: int
    communication_pause_ms: int

    @property
    def total_delivery_delay_ms(self) -> int:
        return self.latency_ms + self.extra_delay_ms + self.communication_pause_ms


@dataclass(frozen=True, slots=True)
class FaultTrace:
    profile: FaultProfile
    seed: int
    entries: tuple[FaultTraceEntry, ...]

    def core_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "milestone": MILESTONE,
            "profile": asdict(self.profile),
            "seed": self.seed,
            "entries": [asdict(entry) for entry in self.entries],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.core_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.core_payload(), "trace_sha256": self.sha256}

    def entry(self, request_ordinal: int) -> FaultTraceEntry:
        if request_ordinal < 0 or request_ordinal >= len(self.entries):
            raise IndexError(
                f"request ordinal {request_ordinal} exceeds frozen trace length "
                f"{len(self.entries)}"
            )
        return self.entries[request_ordinal]


def _entry_rng(profile_id: str, seed: int, ordinal: int) -> random.Random:
    digest = hashlib.sha256(f"{profile_id}:{seed}:{ordinal}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def generate_fault_trace(
    profile: FaultProfile,
    *,
    seed: int,
    request_count: int = 512,
) -> FaultTrace:
    profile.validate()
    if seed < 0 or request_count <= 0:
        raise ValueError("seed must be non-negative and request_count positive")
    entries: list[FaultTraceEntry] = []
    for ordinal in range(request_count):
        rng = _entry_rng(profile.profile_id, seed, ordinal)
        jitter = rng.randint(-profile.jitter_ms, profile.jitter_ms)
        dropped = rng.random() < profile.drop_probability
        extra_delay = (
            profile.extra_delay_ms if rng.random() < profile.extra_delay_probability else 0
        )
        duplicate_count = 1 if rng.random() < profile.duplicate_probability else 0
        pause = (
            profile.communication_pause_ms
            if rng.random() < profile.communication_pause_probability
            else 0
        )
        entries.append(
            FaultTraceEntry(
                request_ordinal=ordinal,
                latency_ms=profile.base_latency_ms + jitter,
                jitter_ms=jitter,
                dropped=dropped,
                extra_delay_ms=extra_delay,
                duplicate_count=duplicate_count,
                communication_pause_ms=pause,
            )
        )
    return FaultTrace(profile=profile, seed=seed, entries=tuple(entries))


def write_fault_trace(path: Path | str, trace: FaultTrace) -> None:
    write_json_atomic(path, trace.to_dict())


def load_fault_trace(path: Path | str) -> FaultTrace:
    payload = read_json(path)
    profile = FaultProfile.from_mapping(payload["profile"])
    trace = FaultTrace(
        profile=profile,
        seed=int(payload["seed"]),
        entries=tuple(FaultTraceEntry(**entry) for entry in payload["entries"]),
    )
    if payload.get("trace_sha256") != trace.sha256:
        raise ValueError(f"fault trace hash mismatch: {path}")
    return trace


def resolve_profile(value: str | Path | FaultProfile) -> FaultProfile:
    if isinstance(value, FaultProfile):
        value.validate()
        return value
    key = str(value)
    if key in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[key]
    payload = read_json(Path(value))
    return FaultProfile.from_mapping(payload.get("profile", payload))


def validate_seed_split(dev_seeds: Iterable[int], holdout_seeds: Iterable[int]) -> None:
    development = tuple(int(seed) for seed in dev_seeds)
    holdout = tuple(int(seed) for seed in holdout_seeds)
    if not development or len(development) > 10:
        raise ValueError("development split must contain 1 to 10 seeds")
    if len(holdout) < 30:
        raise ValueError("holdout split must contain at least 30 seeds")
    if len(set(development)) != len(development) or len(set(holdout)) != len(holdout):
        raise ValueError("seed splits must not contain duplicates")
    overlap = set(development) & set(holdout)
    if overlap:
        raise ValueError(f"development and holdout seeds overlap: {sorted(overlap)}")
