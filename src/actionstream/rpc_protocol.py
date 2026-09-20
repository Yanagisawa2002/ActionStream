"""Wire encoding and framing primitives for ActionStream TCP RPC.

This module is transport-agnostic: it owns the versioned JSON framing contract
and recursive tensor/array serialization. Connection, reset, executor, and
lifecycle semantics remain in rpc_transport.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from actionstream.inference_transport import InferenceDeadlineExceeded


_PROTOCOL_VERSION = 1
_HEADER = struct.Struct("!I")
_MAX_FRAME_BYTES = 256 * 1024 * 1024
_TENSOR_TAG = "__actionstream_tensor__"
_NDARRAY_TAG = "__actionstream_ndarray__"
_BYTES_TAG = "__actionstream_bytes__"

_TORCH_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise InferenceDeadlineExceeded("TCP inference deadline exceeded")
    return remaining


def _encode_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            _TENSOR_TAG: True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            _NDARRAY_TAG: True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, bytes | bytearray | memoryview):
        return {
            _BYTES_TAG: True,
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("RPC mappings require string keys")
            result[key] = _encode_value(child)
        return result
    if isinstance(value, tuple):
        return {"__actionstream_tuple__": [_encode_value(child) for child in value]}
    if isinstance(value, list):
        return [_encode_value(child) for child in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported RPC value type: {type(value)!r}")


def _decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_value(child) for child in value]
    if not isinstance(value, dict):
        return value
    if value.get(_TENSOR_TAG) is True:
        dtype_name = value["dtype"]
        try:
            dtype = _TORCH_DTYPES[dtype_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported torch dtype on wire: {dtype_name}") from exc
        raw = base64.b64decode(value["data"], validate=True)
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
        return tensor.reshape(tuple(int(item) for item in value["shape"]))
    if value.get(_NDARRAY_TAG) is True:
        dtype = np.dtype(value["dtype"])
        raw = base64.b64decode(value["data"], validate=True)
        array = np.frombuffer(raw, dtype=dtype).copy()
        return array.reshape(tuple(int(item) for item in value["shape"]))
    if value.get(_BYTES_TAG) is True:
        return base64.b64decode(value["data"], validate=True)
    if "__actionstream_tuple__" in value:
        return tuple(_decode_value(child) for child in value["__actionstream_tuple__"])
    return {key: _decode_value(child) for key, child in value.items()}


def _encode_frame(payload: Mapping[str, Any]) -> bytes:
    body = json.dumps(
        _encode_value(dict(payload)),
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(body) > _MAX_FRAME_BYTES:
        raise ValueError(f"RPC frame exceeds {_MAX_FRAME_BYTES} bytes")
    return _HEADER.pack(len(body)) + body


def _recv_exact(
    connection: socket.socket, size: int, deadline: float | None = None
) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        if deadline is not None:
            connection.settimeout(_remaining(deadline))
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError("RPC peer closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_frame(
    connection: socket.socket, deadline: float | None = None
) -> tuple[dict[str, Any], int]:
    header = _recv_exact(connection, _HEADER.size, deadline)
    (length,) = _HEADER.unpack(header)
    if length <= 0 or length > _MAX_FRAME_BYTES:
        raise ValueError(f"Invalid RPC frame length: {length}")
    body = _recv_exact(connection, length, deadline)
    decoded = json.loads(body.decode("utf-8"))
    value = _decode_value(decoded)
    if not isinstance(value, dict):
        raise ValueError("RPC top-level frame must be a mapping")
    return value, _HEADER.size + length


def _send_frame(
    connection: socket.socket, payload: Mapping[str, Any], deadline: float | None = None
) -> int:
    frame = _encode_frame(payload)
    if deadline is not None:
        connection.settimeout(_remaining(deadline))
    connection.sendall(frame)
    return len(frame)
