"""Canonical serialization and crash-safe artifact helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MILESTONE = "M7-G0"


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(child) for child in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON artifacts cannot contain NaN or infinity")
        return value
    if hasattr(value, "item"):
        return json_safe(value.item())
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_replace(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path | str, payload: Any) -> None:
    serialized = json.dumps(
        json_safe(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    _atomic_replace(Path(path), serialized + "\n")


def write_jsonl_atomic(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [canonical_json(row) for row in rows]
    _atomic_replace(Path(path), "".join(f"{line}\n" for line in lines))


def read_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


class AtomicJsonlLog:
    """Stream rows to a temporary file and publish only a complete JSONL file."""

    def __init__(self, path: Path | str, *, overwrite: bool = False) -> None:
        self.path = Path(path)
        self.overwrite = overwrite
        self._temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        self._stream = None

    def __enter__(self) -> "AtomicJsonlLog":
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(f"refusing to overwrite {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self._temporary.open("w", encoding="utf-8", newline="\n")
        return self

    def append(self, row: Mapping[str, Any]) -> None:
        if self._stream is None:
            raise RuntimeError("AtomicJsonlLog is not open")
        self._stream.write(canonical_json(row) + "\n")
        self._stream.flush()

    def commit(self) -> None:
        if self._stream is None:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._stream = None
        os.replace(self._temporary, self.path)

    def abort(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._temporary.exists():
            self._temporary.unlink()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self.commit()
        else:
            self.abort()
        return False
