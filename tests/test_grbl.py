"""Unit tests for the hardware-independent GRBL layer."""

from __future__ import annotations

from collections import deque
import unittest

from config import GrblConfig
from grbl import GrblCommandError, GrblController, parse_status


class FakeSerial:
    """Deterministic in-memory serial port used by unit tests."""

    def __init__(self, responses: list[bytes]) -> None:
        self.responses = deque(responses)
        self.writes: list[bytes] = []
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def readline(self) -> bytes:
        return self.responses.popleft() if self.responses else b""

    def reset_input_buffer(self) -> None:
        return None

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)


def make_controller(fake: FakeSerial) -> GrblController:
    config = GrblConfig(startup_delay=0, command_timeout=0.01)
    controller = GrblController(config, serial_factory=lambda **_: fake)
    controller._serial = fake  # Test seam: avoid consuming the scripted greeting.
    return controller


class GrblTests(unittest.TestCase):
    def test_parse_status(self) -> None:
        status = parse_status(
            "<Idle|MPos:0.200,0.000,0.000|FS:0,0|WCO:0.000,0.000,0.000>"
        )
        self.assertEqual(status.state, "Idle")
        self.assertEqual(status.machine_position, (0.2, 0.0, 0.0))
        self.assertEqual(status.feed_rate, 0.0)

    def test_command_collects_lines_until_ok(self) -> None:
        fake = FakeSerial([b"$0=10\n", b"$1=25\n", b"ok\n"])
        controller = make_controller(fake)
        self.assertEqual(controller.command("$$"), ["$0=10", "$1=25"])
        self.assertEqual(fake.writes, [b"$$\n"])

    def test_command_raises_on_grbl_error(self) -> None:
        fake = FakeSerial([b"error:1\n"])
        controller = make_controller(fake)
        with self.assertRaises(GrblCommandError):
            controller.command("bad")

    def test_jog_is_bounded_and_uses_grbl_jog_mode(self) -> None:
        fake = FakeSerial([b"ok\n"])
        controller = make_controller(fake)
        controller.jog(x_mm=2, y_mm=-1, speed_mm_min=60)
        self.assertEqual(fake.writes, [b"$J=G91 X2 Y-1 F60\n"])

    def test_jog_rejects_unsafe_distance(self) -> None:
        fake = FakeSerial([])
        controller = make_controller(fake)
        with self.assertRaisesRegex(ValueError, "distance"):
            controller.jog(x_mm=11, y_mm=0, speed_mm_min=60)

    def test_stop_uses_grbl_jog_cancel_character(self) -> None:
        fake = FakeSerial([])
        controller = make_controller(fake)
        controller.stop()
        self.assertEqual(fake.writes, [b"\x85"])


if __name__ == "__main__":
    unittest.main()
