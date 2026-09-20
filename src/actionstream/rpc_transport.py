"""Compatibility facade for ActionStream TCP RPC.

Implementation ownership is split across rpc_client, rpc_server_runtime,
rpc_executor, rpc_telemetry, and rpc_protocol. Existing imports from this module
remain supported.
"""

from actionstream.rpc_client import RemoteResetError, TcpInferenceTransport
from actionstream.rpc_protocol import (
    _PROTOCOL_VERSION,
    _recv_exact,
    _recv_frame,
    _send_frame,
)
from actionstream.rpc_server_runtime import (
    RpcFaultProfile,
    RpcInferenceServer,
    load_worker,
)
from actionstream.rpc_telemetry import RpcTransportTelemetry

__all__ = [
    "RemoteResetError",
    "RpcFaultProfile",
    "RpcInferenceServer",
    "RpcTransportTelemetry",
    "TcpInferenceTransport",
    "load_worker",
    "_PROTOCOL_VERSION",
    "_recv_exact",
    "_recv_frame",
    "_send_frame",
]
