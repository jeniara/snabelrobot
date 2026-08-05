"""Web viewer for Raspberry Pi camera inference pipelines."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

LOGGER = logging.getLogger(__name__)
CAMERAS = {0, 1}
PROJECT_DIR = Path(__file__).resolve().parent
PIPELINES = {
    "objects": Path("/usr/share/rpi-camera-assets/hailo_yolov8_inference.json"),
    "faces": PROJECT_DIR / "config" / "face_detect_cv.json",
}
MODEL_LABELS = {
    "objects": "Modell: YOLOv8s | Hailo-8 26 TOPS",
    "faces": "Modell: OpenCV Face + Eyes | CPU",
}
FACE_CASCADE = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
)
app = Flask(__name__)


def _validated_options() -> tuple[int, str]:
    camera = request.args.get("camera", default=0, type=int)
    if camera not in CAMERAS:
        raise ValueError("Kameran måste vara 0 eller 1")
    mode = "objects" if camera == 0 else "faces"
    return camera, mode


def _rpicam_command(camera: int, mode: str) -> list[str]:
    command = [
        "rpicam-vid", "--camera", str(camera), "--timeout", "0",
        "--width", "1280", "--height", "720", "--framerate", "15",
        "--codec", "mjpeg", "--nopreview",
    ]
    if mode == "objects":
        command.extend(["--post-process-file", str(PIPELINES[mode])])
    command.extend(["--output", "-"])
    return command


def _draw_face_analysis(image: np.ndarray) -> None:
    """Draw face count and an approximate left/front/right head direction."""
    scale = 0.5
    small = cv2.resize(image, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32)
    )
    directions: list[str] = []
    for x, y, width, height in faces:
        upper_face = gray[y:y + int(height * 0.65), x:x + width]
        eyes = EYE_CASCADE.detectMultiScale(
            upper_face, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10)
        )
        direction = "framåt"
        if len(eyes) >= 2:
            eye_centres = sorted(
                (eye_x + eye_w / 2 for eye_x, _, eye_w, _ in eyes)
            )[:2]
            eye_midpoint = sum(eye_centres) / 2
            offset = (eye_midpoint - width / 2) / width
            if offset < -0.07:
                direction = "vänster"
            elif offset > 0.07:
                direction = "höger"
        directions.append(direction)
        x1, y1 = int(x / scale), int(y / scale)
        x2, y2 = int((x + width) / scale), int((y + height) / scale)
        cv2.rectangle(image, (x1, y1), (x2, y2), (80, 235, 180), 3)
        cv2.putText(
            image, f"Huvud: {direction}", (x1, max(80, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 235, 180), 2, cv2.LINE_AA,
        )
    summary = f"Ansikten: {len(faces)}"
    if directions:
        summary += " | " + ", ".join(directions)
    cv2.rectangle(image, (12, 66), (650, 108), (10, 20, 28), -1)
    cv2.putText(
        image, summary, (25, 95), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )


def _jpeg_frames(process: subprocess.Popen[bytes], mode: str) -> Iterator[bytes]:
    """Split an MJPEG stream and draw the active model on every frame."""
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
                image = cv2.imdecode(
                    np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is not None:
                    if mode == "faces" and not FACE_CASCADE.empty():
                        _draw_face_analysis(image)
                    cv2.rectangle(image, (12, 12), (530, 58), (10, 20, 28), -1)
                    cv2.putText(
                        image,
                        MODEL_LABELS[mode],
                        (25, 44),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (80, 235, 180),
                        2,
                        cv2.LINE_AA,
                    )
                    encoded, jpeg = cv2.imencode(
                        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88]
                    )
                    if encoded:
                        frame = jpeg.tobytes()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
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
        _jpeg_frames(process, mode),
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
