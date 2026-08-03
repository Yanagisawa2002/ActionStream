"""Small ROS-independent timestamp helpers shared by adapter nodes."""

from __future__ import annotations


NANOSECONDS_PER_SECOND = 1_000_000_000


def split_nanoseconds(value: int) -> tuple[int, int]:
    if value < 0:
        raise ValueError("ROS simulation timestamps must be non-negative")
    seconds, nanoseconds = divmod(int(value), NANOSECONDS_PER_SECOND)
    if seconds > 2_147_483_647:
        raise OverflowError("builtin_interfaces/Time seconds overflow int32")
    return seconds, nanoseconds


def combine_nanoseconds(seconds: int, nanoseconds: int) -> int:
    if seconds < 0 or not 0 <= nanoseconds < NANOSECONDS_PER_SECOND:
        raise ValueError("invalid ROS timestamp components")
    return int(seconds) * NANOSECONDS_PER_SECOND + int(nanoseconds)


def assign_time(message: object, value: int) -> None:
    seconds, nanoseconds = split_nanoseconds(value)
    message.sec = seconds
    message.nanosec = nanoseconds
