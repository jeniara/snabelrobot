"""Standalone web control panel for Snabelrobot V1."""

from __future__ import annotations

import atexit
import logging
import threading
import time
from dataclasses import asdict

from flask import Flask, jsonify, render_template, request

from config import GrblConfig
from grbl import GrblController, GrblError

LOGGER = logging.getLogger(__name__)
WATCHDOG_SECONDS = 0.75


class RobotService:
    """Coordinate GRBL access and stop motion when heartbeats disappear."""

    def __init__(self, controller: GrblController) -> None:
        self.controller = controller
        self._last_heartbeat = 0.0
        self._jog_active = False
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog.start()

    def connect(self) -> str:
        with self._lock:
            return self.controller.connect()

    def disconnect(self) -> None:
        with self._lock:
            if self._jog_active and self.controller.is_connected:
                self.controller.stop()
            self._jog_active = False
            self.controller.disconnect()

    def status(self) -> dict[str, object]:
        with self._lock:
            status = self.controller.status()
            result = asdict(status)
            result["connected"] = True
            result["version"] = self.controller.version
            return result

    def jog(self, x: float, y: float, speed: float) -> None:
        with self._lock:
            self.controller.jog(x, y, speed)
            self._last_heartbeat = time.monotonic()
            self._jog_active = True

    def heartbeat(self) -> None:
        with self._lock:
            if self._jog_active:
                self._last_heartbeat = time.monotonic()

    def stop(self) -> None:
        with self._lock:
            if self.controller.is_connected:
                self.controller.stop()
            self._jog_active = False

    def hold(self) -> None:
        with self._lock:
            self.controller.feed_hold()
            self._jog_active = False

    def emergency_stop(self) -> None:
        with self._lock:
            self.controller.soft_reset()
            self._jog_active = False

    def close(self) -> None:
        self._shutdown.set()
        self.disconnect()

    def _watchdog_loop(self) -> None:
        while not self._shutdown.wait(0.1):
            with self._lock:
                expired = (
                    self._jog_active
                    and time.monotonic() - self._last_heartbeat > WATCHDOG_SECONDS
                )
                if expired:
                    LOGGER.warning("Jog watchdog expired; cancelling motion")
                    try:
                        self.controller.stop()
                    except GrblError:
                        LOGGER.exception("Watchdog could not cancel jog")
                    self._jog_active = False


def create_app(service: RobotService | None = None) -> Flask:
    """Create the Flask application, optionally with an injected test service."""

    app = Flask(__name__)
    robot = service or RobotService(GrblController(GrblConfig()))
    app.config["ROBOT_SERVICE"] = robot

    def success(**values: object):
        return jsonify(ok=True, **values)

    @app.errorhandler(GrblError)
    @app.errorhandler(ValueError)
    def handle_robot_error(error: Exception):
        LOGGER.warning("Robot request rejected: %s", error)
        return jsonify(ok=False, error=str(error)), 400

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/connect")
    def connect():
        return success(version=robot.connect())

    @app.post("/api/disconnect")
    def disconnect():
        robot.disconnect()
        return success()

    @app.get("/api/status")
    def status():
        if not robot.controller.is_connected:
            return jsonify(ok=True, connected=False)
        return success(**robot.status())

    @app.post("/api/jog")
    def jog():
        payload = request.get_json(force=True)
        robot.jog(float(payload.get("x", 0)), float(payload.get("y", 0)), float(payload["speed"]))
        return success()

    @app.post("/api/heartbeat")
    def heartbeat():
        robot.heartbeat()
        return success()

    @app.post("/api/stop")
    def stop():
        robot.stop()
        return success()

    @app.post("/api/hold")
    def hold():
        robot.hold()
        return success()

    @app.post("/api/emergency-stop")
    def emergency_stop():
        robot.emergency_stop()
        return success()

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    application = create_app()
    service = application.config["ROBOT_SERVICE"]
    atexit.register(service.close)
    try:
        service.connect()
    except GrblError:
        LOGGER.exception("Startup connection failed; the web UI can retry")
    application.run(host="0.0.0.0", port=8080, threaded=True)

