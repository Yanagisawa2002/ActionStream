"""M8-specific deterministic latency and response-fault traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Final, Mapping, Sequence

from .m8_protocol import M8_MILESTONE, M8_SCHEMA_VERSION
from .schema import canonical_sha256, read_json, write_json_atomic


FAULT_PROFILE_INTEGER_FIELDS: Final = (
    "base_latency_ms",
    "jitter_ms",
    "extra_delay_ms",
    "duplicate_delivery_offset_ms",
    "communication_pause_ms",
)
FAULT_PROFILE_PROBABILITY_FIELDS: Final = (
    "drop_probability",
    "extra_delay_probability",
    "duplicate_probability",
    "communication_pause_probability",
)
FAULT_PROFILE_NUMERIC_FIELDS: Final = (
    *FAULT_PROFILE_INTEGER_FIELDS,
    *FAULT_PROFILE_PROBABILITY_FIELDS,
)


@dataclass(frozen=True, slots=True)
class M8FaultProfile:
    profile_id: str
    base_latency_ms: int
    jitter_ms: int
    drop_probability: float
    extra_delay_probability: float
    extra_delay_ms: int
    duplicate_probability: float
    duplicate_delivery_offset_ms: int
    communication_pause_probability: float
    communication_pause_ms: int
    frozen: bool = True

    def validate(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        for name in FAULT_PROFILE_INTEGER_FIELDS:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.jitter_ms > self.base_latency_ms:
            raise ValueError("jitter cannot make latency negative")
        for name in FAULT_PROFILE_PROBABILITY_FIELDS:
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be a JSON number")
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not self.frozen:
            raise ValueError("profile must be frozen before trace generation")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "M8FaultProfile":
        allowed = set(cls.__dataclass_fields__)
        profile = cls(**{key: child for key, child in value.items() if key in allowed})
        profile.validate()
        return profile


TRACE_PROFILE_FIELDS: Final = frozenset(M8FaultProfile.__dataclass_fields__)
DECLARED_PROFILE_FIELDS: Final = frozenset(
    {
        *(TRACE_PROFILE_FIELDS - {"frozen"}),
        "strategies",
        "candidate_status",
    }
)


def declared_fault_profile(
    payload: Mapping[str, Any],
    *,
    expected_profile_id: str,
    expected_strategies: Sequence[str],
) -> M8FaultProfile:
    """Parse the exact profile bytes that are allowed to generate a trace."""

    if payload.get("schema_version") != M8_SCHEMA_VERSION:
        raise ValueError("declared fault profile schema mismatch")
    if payload.get("milestone") != M8_MILESTONE:
        raise ValueError("declared fault profile milestone mismatch")
    raw = payload.get("profile")
    if not isinstance(raw, Mapping):
        raise ValueError("declared fault profile payload is missing")
    actual_fields = set(raw)
    if actual_fields != DECLARED_PROFILE_FIELDS:
        raise ValueError(
            "declared fault profile field drift: "
            f"missing={sorted(DECLARED_PROFILE_FIELDS - actual_fields)}, "
            f"unknown={sorted(actual_fields - DECLARED_PROFILE_FIELDS)}"
        )
    if raw.get("profile_id") != expected_profile_id:
        raise ValueError("declared fault profile identity mismatch")
    if tuple(raw.get("strategies", ())) != tuple(expected_strategies):
        raise ValueError("declared fault profile strategy set mismatch")
    if not isinstance(raw.get("candidate_status"), str) or not raw["candidate_status"]:
        raise ValueError("declared fault profile candidate_status is invalid")
    profile = M8FaultProfile.from_mapping({**raw, "frozen": True})
    if expected_profile_id == "profile_0_sanity" and any(
        getattr(profile, name) != 0 for name in FAULT_PROFILE_NUMERIC_FIELDS
    ):
        raise ValueError("Profile 0 cannot carry latency or response faults")
    return profile


def fault_profile_binding_mismatches(
    actual: M8FaultProfile,
    declared: M8FaultProfile,
) -> dict[str, dict[str, Any]]:
    """Return every profile field that differs from the declared profile bytes."""

    actual_payload = asdict(actual)
    declared_payload = asdict(declared)
    return {
        name: {"declared": declared_payload[name], "actual": actual_payload[name]}
        for name in sorted(TRACE_PROFILE_FIELDS)
        if actual_payload[name] != declared_payload[name]
    }


@dataclass(frozen=True, slots=True)
class M8FaultEntry:
    request_ordinal: int
    latency_ms: int
    jitter_ms: int
    dropped: bool
    extra_delay_ms: int
    duplicate_count: int
    duplicate_delivery_offset_ms: int
    communication_pause_ms: int

    @property
    def total_delivery_delay_ms(self) -> int:
        return self.latency_ms + self.extra_delay_ms + self.communication_pause_ms


@dataclass(frozen=True, slots=True)
class M8FaultTrace:
    profile: M8FaultProfile
    seed: int
    entries: tuple[M8FaultEntry, ...]

    def core_payload(self) -> dict[str, Any]:
        return {
            "schema_version": M8_SCHEMA_VERSION,
            "milestone": M8_MILESTONE,
            "profile": asdict(self.profile),
            "seed": self.seed,
            "entries": [asdict(entry) for entry in self.entries],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.core_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.core_payload(), "trace_sha256": self.sha256}

    def entry(self, request_ordinal: int) -> M8FaultEntry:
        if not 0 <= request_ordinal < len(self.entries):
            raise IndexError(f"request ordinal outside frozen trace: {request_ordinal}")
        return self.entries[request_ordinal]


def _entry_rng(profile_id: str, seed: int, ordinal: int) -> random.Random:
    digest = hashlib.sha256(
        f"{M8_MILESTONE}:{profile_id}:{seed}:{ordinal}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def generate_fault_trace(
    profile: M8FaultProfile,
    *,
    seed: int,
    request_count: int = 512,
) -> M8FaultTrace:
    profile.validate()
    if seed < 0 or request_count <= 0:
        raise ValueError("seed must be non-negative and request_count positive")
    entries = []
    for ordinal in range(request_count):
        rng = _entry_rng(profile.profile_id, seed, ordinal)
        jitter = rng.randint(-profile.jitter_ms, profile.jitter_ms)
        dropped = rng.random() < profile.drop_probability
        extra = (
            profile.extra_delay_ms
            if rng.random() < profile.extra_delay_probability
            else 0
        )
        duplicate = 1 if rng.random() < profile.duplicate_probability else 0
        pause = (
            profile.communication_pause_ms
            if rng.random() < profile.communication_pause_probability
            else 0
        )
        entries.append(
            M8FaultEntry(
                request_ordinal=ordinal,
                latency_ms=profile.base_latency_ms + jitter,
                jitter_ms=jitter,
                dropped=dropped,
                extra_delay_ms=extra,
                duplicate_count=duplicate,
                duplicate_delivery_offset_ms=profile.duplicate_delivery_offset_ms,
                communication_pause_ms=pause,
            )
        )
    return M8FaultTrace(profile=profile, seed=seed, entries=tuple(entries))


def load_profile(path: Path | str) -> M8FaultProfile:
    payload = read_json(path)
    if payload.get("milestone") != M8_MILESTONE:
        raise ValueError(f"profile milestone must be {M8_MILESTONE}: {path}")
    return M8FaultProfile.from_mapping(payload.get("profile", payload))


def write_fault_trace(path: Path | str, trace: M8FaultTrace) -> None:
    write_json_atomic(path, trace.to_dict())


def fault_trace_from_mapping(
    payload: Mapping[str, Any],
    *,
    source: str = "<mapping>",
) -> M8FaultTrace:
    """Parse and self-hash-check a trace from loose or archived JSON."""

    if payload.get("milestone") != M8_MILESTONE:
        raise ValueError(f"fault trace milestone mismatch: {source}")
    raw_profile = payload.get("profile")
    if not isinstance(raw_profile, Mapping) or set(raw_profile) != TRACE_PROFILE_FIELDS:
        raise ValueError(f"fault trace profile field drift: {source}")
    trace = M8FaultTrace(
        profile=M8FaultProfile.from_mapping(raw_profile),
        seed=int(payload["seed"]),
        entries=tuple(M8FaultEntry(**entry) for entry in payload["entries"]),
    )
    if payload.get("trace_sha256") != trace.sha256:
        raise ValueError(f"fault trace hash mismatch: {source}")
    return trace


def load_fault_trace(path: Path | str) -> M8FaultTrace:
    return fault_trace_from_mapping(read_json(path), source=str(path))
