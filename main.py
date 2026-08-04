"""Command-line diagnostics and safe jogging for Snabelrobot V1."""

from __future__ import annotations

import argparse
import logging

from config import GrblConfig
from grbl import GrblController, GrblError


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Snabelrobot GRBL controller")
    parser.add_argument("--port", default="/dev/ttyACM0", help="GRBL serial port")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("info", help="show GRBL version")
    subparsers.add_parser("status", help="show controller status")
    subparsers.add_parser("settings", help="read GRBL settings")

    jog_parser = subparsers.add_parser("jog", help="perform a bounded relative jog")
    jog_parser.add_argument("--x", type=float, default=0.0, help="relative X distance in mm")
    jog_parser.add_argument("--y", type=float, default=0.0, help="relative Y distance in mm")
    jog_parser.add_argument("--speed", type=float, required=True, help="speed in mm/min")

    return parser


def run() -> int:
    """Run the requested command and return a process exit code."""

    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    controller = GrblController(GrblConfig(port=args.port))
    try:
        with controller:
            if args.action == "info":
                print(controller.version)
            elif args.action == "status":
                print(controller.status())
            elif args.action == "settings":
                print("\n".join(controller.command("$$")))
            elif args.action == "jog":
                controller.jog(args.x, args.y, args.speed)
                print(controller.status())
    except (GrblError, ValueError) as error:
        logging.getLogger(__name__).error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

