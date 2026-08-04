"""Dual Camera Module 3 viewer and calibrated stereo distance estimator."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from picamera2 import Picamera2

LOGGER = logging.getLogger(__name__)
FRAME_SIZE = (640, 360)
CALIBRATION_FILE = Path("camera_data/stereo_calibration.npz")
CAPTURE_DIRECTORY = Path("camera_data/calibration_pairs")


class StereoCamera:
    """Capture two cameras and calculate depth when calibration is available."""

    def __init__(self) -> None:
        self.left = Picamera2(0)
        self.right = Picamera2(1)
        config = {"main": {"size": FRAME_SIZE, "format": "RGB888"}}
        self.left.configure(self.left.create_video_configuration(**config))
        self.right.configure(self.right.create_video_configuration(**config))
        controls = {"AfMode": 2, "FrameRate": 20.0}
        self.left.set_controls(controls)
        self.right.set_controls(controls)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._frames: dict[str, bytes | None] = {"left": None, "right": None, "depth": None}
        self._raw: tuple[np.ndarray, np.ndarray] | None = None
        self.distance_m: float | None = None
        self.valid_fraction = 0.0
        self.calibration = self._load_calibration()
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=7,
            P1=8 * 3 * 7**2,
            P2=32 * 3 * 7**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
        )
        self.thread = threading.Thread(target=self._run, daemon=True)

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    def start(self) -> None:
        self.left.start()
        self.right.start()
        time.sleep(1.0)
        self.thread.start()

    def close(self) -> None:
        self._stop.set()
        self.thread.join(timeout=2)
        self.left.stop()
        self.right.stop()

    def frame(self, name: str) -> bytes | None:
        with self._lock:
            return self._frames[name]

    def capture_calibration_pair(self) -> int:
        with self._lock:
            if self._raw is None:
                raise RuntimeError("No camera frames are available")
            left, right = (frame.copy() for frame in self._raw)
        CAPTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        existing = list(CAPTURE_DIRECTORY.glob("left_*.png"))
        number = len(existing) + 1
        cv2.imwrite(str(CAPTURE_DIRECTORY / f"left_{number:03d}.png"), left)
        cv2.imwrite(str(CAPTURE_DIRECTORY / f"right_{number:03d}.png"), right)
        return number

    def _load_calibration(self) -> dict[str, np.ndarray] | None:
        if not CALIBRATION_FILE.exists():
            return None
        data = np.load(CALIBRATION_FILE)
        return {key: data[key] for key in data.files}

    def _run(self) -> None:
        while not self._stop.is_set():
            left = self.left.capture_array("main")
            right = self.right.capture_array("main")
            with self._lock:
                self._raw = (left, right)
            shown_left, shown_right = left, right
            depth_jpeg = None
            if self.calibration:
                shown_left, shown_right, depth_jpeg = self._calculate_depth(left, right)
            left_jpeg = self._jpeg(shown_left)
            right_jpeg = self._jpeg(shown_right)
            with self._lock:
                self._frames.update(left=left_jpeg, right=right_jpeg, depth=depth_jpeg)

    def _calculate_depth(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bytes]:
        cal = self.calibration
        assert cal is not None
        left_rect = cv2.remap(left, cal["map1x"], cal["map1y"], cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, cal["map2x"], cal["map2y"], cv2.INTER_LINEAR)
        gray_left = cv2.cvtColor(left_rect, cv2.COLOR_RGB2GRAY)
        gray_right = cv2.cvtColor(right_rect, cv2.COLOR_RGB2GRAY)
        disparity = self.matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
        points = cv2.reprojectImageTo3D(disparity, cal["q"])
        depth = np.abs(points[:, :, 2])
        valid = (disparity > 1.0) & np.isfinite(depth) & (depth > 0.10) & (depth < 5.0)
        margin_x, margin_y = FRAME_SIZE[0] // 10, FRAME_SIZE[1] // 10
        valid[:margin_y] = False
        valid[-margin_y:] = False
        valid[:, :margin_x] = False
        valid[:, -margin_x:] = False
        values = depth[valid]
        self.valid_fraction = float(valid.mean())
        self.distance_m = float(np.percentile(values, 10)) if values.size > 500 else None
        normalized = np.zeros_like(gray_left)
        if values.size:
            clipped = np.clip(depth, 0.15, 3.0)
            normalized = np.uint8(255 * (3.0 - clipped) / 2.85)
            normalized[~valid] = 0
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        label = "No reliable depth" if self.distance_m is None else f"Nearest stable depth: {self.distance_m:.2f} m"
        cv2.putText(heatmap, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return left_rect, right_rect, self._jpeg(heatmap)

    @staticmethod
    def _jpeg(frame: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return encoded.tobytes()


def create_app(stereo: StereoCamera) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("stereo.html")

    @app.get("/stream/<name>")
    def stream(name: str):
        if name not in {"left", "right", "depth"}:
            return "Unknown stream", 404

        def generate():
            while True:
                frame = stereo.frame(name)
                if frame:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.05)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/status")
    def status():
        return jsonify(
            calibrated=stereo.calibrated,
            distance_m=stereo.distance_m,
            valid_fraction=stereo.valid_fraction,
            baseline_cm=3.45,
        )

    @app.post("/api/capture-calibration")
    def capture_calibration():
        return jsonify(ok=True, pair=stereo.capture_calibration_pair())

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    camera = StereoCamera()
    camera.start()
    try:
        create_app(camera).run(host="0.0.0.0", port=8081, threaded=True)
    finally:
        camera.close()

