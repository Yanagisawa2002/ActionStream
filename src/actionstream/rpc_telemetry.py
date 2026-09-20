"""RPC telemetry snapshots shared by the client and compatibility facade."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RpcTransportTelemetry:
    requests_started: int
    requests_completed: int
    reconnects: int
    cancellations: int
    deadlines: int
    transport_errors: int
    server_errors: int
    bytes_sent: int
    bytes_received: int
    latest_connect_latency_ms: float | None
    latest_roundtrip_latency_ms: float | None
    latest_server_inference_ms: float | None
    startup_deadlines: int
    steady_deadlines: int
    cancelled_invalidated_requests: int
    connections_established: int
    reconnections: int
    latest_response_telemetry: dict[str, Any]
    reset_requests: int = 0
    reset_successes: int = 0
    reset_failures: int = 0
    reset_timeouts: int = 0
    reset_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

