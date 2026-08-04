"""Application configuration for the snabelrobot controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GrblConfig:
    """Serial and motion limits used when communicating with GRBL."""

    port: str = "/dev/ttyACM0"
    baudrate: int = 115_200
    startup_delay: float = 2.0
    command_timeout: float = 2.0
    max_jog_distance_mm: float = 10.0
    max_jog_speed_mm_min: float = 500.0

