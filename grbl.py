"""Safe, hardware-independent GRBL communication layer."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from config import GrblConfig

LOGGER = logging.getLogger(__name__)


class GrblError(RuntimeError):
    """Base exception for GRBL communication failures."""


class GrblConnectionError(GrblError):
    """Raised when an operation requires an active serial connection."""


class GrblCommandError(GrblError):
    """Raised when GRBL rejects a command or reports an alarm."""


@runtime_checkable
class SerialPort(Protocol):
    """Small subset of ``pyserial.Serial`` needed by the controller."""

    is_open: bool

    def close(self) -> None: ...

    def readline(self) -> bytes: ...

    def reset_input_buffer(self) -> None: ...

    def write(self, data: bytes) -> int: ...


SerialFactory = Callable[..., SerialPort]


@dataclass(frozen=True, slots=True)
class GrblStatus:
    """Parsed subset of a GRBL real-time status response."""

    state: str
    machine_position: tuple[float, float, float] | None = None
    feed_rate: float | None = None
    spindle_speed: float | None = None
    raw: str = ""


_STATUS_PATTERN = re.compile(r"^<(?P<body>.*)>$")


def parse_status(response: str) -> GrblStatus:
    """Parse a GRBL ``?`` response into a structured status object."""

    match = _STATUS_PATTERN.match(response.strip())
    if not match:
        raise GrblCommandError(f"Invalid GRBL status response: {response!r}")

    fields = match.group("body").split("|")
    state = fields[0]
    values = dict(field.split(":", 1) for field in fields[1:] if ":" in field)

    position = None
    if "MPos" in values:
        coordinates = tuple(float(value) for value in values["MPos"].split(","))
        if len(coordinates) == 3:
            position = coordinates

    feed_rate = None
    spindle_speed = None
    if "FS" in values:
        feed_and_speed = values["FS"].split(",")
        if len(feed_and_speed) == 2:
            feed_rate = float(feed_and_speed[0])
            spindle_speed = float(feed_and_speed[1])

    return GrblStatus(
        state=state,
        machine_position=position,
        feed_rate=feed_rate,
        spindle_speed=spindle_speed,
        raw=response.strip(),
    )


def _default_serial_factory(**kwargs: object) -> SerialPort:
    try:
        import serial
    except ImportError as error:
        raise GrblConnectionError(
            "pyserial is missing; install dependencies with "
            "'python -m pip install -r requirements.txt'"
        ) from error
    return serial.Serial(**kwargs)


class GrblController:
    """Own the serial connection and expose safe GRBL operations."""

    def __init__(
        self,
        config: GrblConfig | None = None,
        serial_factory: SerialFactory = _default_serial_factory,
    ) -> None:
        self.config = config or GrblConfig()
        self._serial_factory = serial_factory
        self._serial: SerialPort | None = None
        self._lock = threading.RLock()
        self.version: str | None = None

    @property
    def is_connected(self) -> bool:
        """Return whether the serial connection is currently open."""

        return self._serial is not None and self._serial.is_open

    def connect(self) -> str:
        """Open the serial port and return GRBL's startup/version line."""

        with self._lock:
            if self.is_connected:
                return self.version or "connected"

            LOGGER.info("Connecting to GRBL on %s", self.config.port)
            try:
                self._serial = self._serial_factory(
                    port=self.config.port,
                    baudrate=self.config.baudrate,
                    timeout=0.1,
                    write_timeout=self.config.command_timeout,
                )
                time.sleep(self.config.startup_delay)
                startup_lines = self._drain_lines()
            except Exception as error:
                self._serial = None
                raise GrblConnectionError(
                    f"Could not open GRBL on {self.config.port}: {error}"
                ) from error

            version = next((line for line in startup_lines if line.startswith("Grbl ")), None)
            if version is None:
                self.disconnect()
                raise GrblConnectionError("Serial port opened, but no GRBL greeting was received")
            self.version = version
            LOGGER.info("Connected: %s", version)
            return version

    def disconnect(self) -> None:
        """Close the serial connection if it is open."""

        with self._lock:
            if self._serial is not None:
                self._serial.close()
            self._serial = None
            LOGGER.info("Disconnected from GRBL")

    def command(self, command: str) -> list[str]:
        """Send one line-oriented command and return response lines before ``ok``."""

        normalized = command.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("Exactly one non-empty GRBL command is required")

        with self._lock:
            serial_port = self._require_connection()
            LOGGER.debug("GRBL <- %s", normalized)
            serial_port.write(f"{normalized}\n".encode("ascii"))
            deadline = time.monotonic() + self.config.command_timeout
            responses: list[str] = []

            while time.monotonic() < deadline:
                line = serial_port.readline().decode("ascii", errors="replace").strip()
                if not line:
                    continue
                LOGGER.debug("GRBL -> %s", line)
                if line == "ok":
                    return responses
                if line.startswith("error:") or line.startswith("ALARM:"):
                    raise GrblCommandError(f"GRBL rejected {normalized!r}: {line}")
                responses.append(line)

            raise GrblCommandError(f"Timed out waiting for response to {normalized!r}")

    def status(self) -> GrblStatus:
        """Request and parse GRBL's real-time status."""

        with self._lock:
            serial_port = self._require_connection()
            serial_port.write(b"?")
            deadline = time.monotonic() + self.config.command_timeout
            while time.monotonic() < deadline:
                line = serial_port.readline().decode("ascii", errors="replace").strip()
                if line.startswith("<") and line.endswith(">"):
                    return parse_status(line)
            raise GrblCommandError("Timed out waiting for GRBL status")

    def jog(self, x_mm: float, y_mm: float, speed_mm_min: float) -> None:
        """Perform a bounded relative XY jog through GRBL's jog command."""

        if x_mm == 0 and y_mm == 0:
            raise ValueError("A jog must move at least one axis")
        if max(abs(x_mm), abs(y_mm)) > self.config.max_jog_distance_mm:
            raise ValueError("Jog distance exceeds the configured safety limit")
        if not 0 < speed_mm_min <= self.config.max_jog_speed_mm_min:
            raise ValueError("Jog speed is outside the configured safety limit")

        axes = []
        if x_mm:
            axes.append(f"X{x_mm:g}")
        if y_mm:
            axes.append(f"Y{y_mm:g}")
        self.command(f"$J=G91 {' '.join(axes)} F{speed_mm_min:g}")

    def stop(self) -> None:
        """Request an immediate GRBL jog cancel."""

        with self._lock:
            self._require_connection().write(b"\x85")
            LOGGER.warning("Jog cancel sent")

    def feed_hold(self) -> None:
        """Pause the active motion using GRBL's real-time feed hold."""

        with self._lock:
            self._require_connection().write(b"!")
            LOGGER.warning("Feed hold sent")

    def soft_reset(self) -> None:
        """Reset GRBL using Ctrl-X; this stops motion and clears serial state."""

        with self._lock:
            self._require_connection().write(b"\x18")
            LOGGER.critical("GRBL soft reset sent")

    def _require_connection(self) -> SerialPort:
        if not self.is_connected or self._serial is None:
            raise GrblConnectionError("Not connected to GRBL")
        return self._serial

    def _drain_lines(self) -> list[str]:
        serial_port = self._require_connection()
        lines: list[str] = []
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            line = serial_port.readline().decode("ascii", errors="replace").strip()
            if line:
                lines.append(line)
        return lines

    def __enter__(self) -> "GrblController":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

