"""Durable, payload-minimal JSONL telemetry for inference backends."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class JsonlTelemetrySink:
    """Append one flushed JSON object per runtime event.

    Observations and actions are intentionally excluded.  Callers provide only
    scalar provenance/queue fields so telemetry can be retained without leaking
    camera frames, credentials, or model payloads.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine_id = uuid.uuid4().hex
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        payload = {
            "schema_version": 1,
            "engine_id": self.engine_id,
            "event": event,
            "utc_unix_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            **fields,
        }
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
            stream.flush()
