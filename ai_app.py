"""Web viewer for Raspberry Pi camera inference pipelines."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

LOGGER = logging.getLogger(__name__)
CAMERAS = {0, 1}
PIPELINES = {
    "objects": Path("/usr/share/rpi-camera-assets/hailo_yolov8_inference.json"),
    "faces": Path("/usr/share/rpi-camera-assets/face_detect_cv.json"),
}
app = Flask(__name__)


def _validated_options() -> tuple[int, str]:
    camera = request.args.get("camera", default=0, type=int)
    mode = request.args.get("mode", default="objects", type=str)
    if camera not in CAMERAS:
        raise ValueError("Kameran måste vara 0 eller 1")
    if mode not in PIPELINES:
        raise ValueError("Okänt AI-läge")
    return camera, mode


def _rpicam_command(camera: int, mode: str) -> list[str]:
    return [
        "rpicam-vid", "--camera", str(camera), "--timeout", "0",
        "--width", "1280", "--height", "720", "--framerate", "15",
        "--codec", "mjpeg", "--nopreview", "--post-process-file",
        str(PIPELINES[mode]), "--output", "-",
    ]


def _jpeg_frames(process: subprocess.Popen[bytes]) -> Iterator[bytes]:
    """Split an MJPEG stream into browser multipart frames."""
    assert process.stdout is not None
    buffer = bytearray()
    try:
        while chunk := process.stdout.read(64 * 1024):
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2)
                if start < 0 or end < 0:
                    if len(buffer) > 4 * 1024 * 1024:
                        del buffer[:-2]
                    break
                frame = bytes(buffer[start:end + 2])
                del buffer[:end + 2]
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


@app.get("/")
def index() -> str:
    return render_template("ai.html")


@app.get("/video_feed")
def video_feed() -> Response:
    try:
        camera, mode = _validated_options()
    except ValueError as error:
        return Response(str(error), status=400)
    config = PIPELINES[mode]
    if not config.exists():
        return Response(f"Konfiguration saknas: {config}", status=503)
    LOGGER.info("Starting %s detection on camera %d", mode, camera)
    process = subprocess.Popen(
        _rpicam_command(camera, mode),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return Response(
        _jpeg_frames(process),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
def status() -> Response:
    return jsonify({
        "hailo_ready": Path("/dev/hailo0").exists(),
        "object_accelerator": "Hailo-8 (26 TOPS)",
        "face_accelerator": "CPU (Hailo-8 face model not bundled)",
    })


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=8082, threaded=True)


if __name__ == "__main__":
    main()
