"""Calibrate the stereo cameras from captured checkerboard image pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def calibrate(
    columns: int,
    rows: int,
    square_x_mm: float,
    square_y_mm: float,
) -> None:
    directory = Path("camera_data/calibration_pairs")
    left_files = sorted(directory.glob("left_*.png"))
    right_files = sorted(directory.glob("right_*.png"))
    if len(left_files) != len(right_files) or len(left_files) < 10:
        raise SystemExit("At least 10 matching image pairs are required")
    pattern = (columns, rows)
    object_template = np.zeros((columns * rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    object_template[:, 0] *= square_x_mm / 1000.0
    object_template[:, 1] *= square_y_mm / 1000.0
    objects, left_points, right_points = [], [], []
    size = None
    for left_path, right_path in zip(left_files, right_files):
        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        size = (left.shape[1], left.shape[0])
        found_l, corners_l = cv2.findChessboardCorners(left, pattern)
        found_r, corners_r = cv2.findChessboardCorners(right, pattern)
        if found_l and found_r:
            objects.append(object_template)
            left_points.append(corners_l)
            right_points.append(corners_r)
    if len(objects) < 10 or size is None:
        raise SystemExit(f"Only {len(objects)} usable pairs; capture more varied views")
    _, k1, d1, _, _ = cv2.calibrateCamera(objects, left_points, size, None, None)
    _, k2, d2, _, _ = cv2.calibrateCamera(objects, right_points, size, None, None)
    rms, k1, d1, k2, d2, r, t, _, _ = cv2.stereoCalibrate(
        objects, left_points, right_points, k1, d1, k2, d2, size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    measured_baseline = float(np.linalg.norm(t))
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(k1, d1, k2, d2, size, r, t)
    map1x, map1y = cv2.initUndistortRectifyMap(k1, d1, r1, p1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(k2, d2, r2, p2, size, cv2.CV_32FC1)
    output = Path("camera_data/stereo_calibration.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y, q=q)
    print(f"Saved {output}; RMS={rms:.3f}, measured baseline={measured_baseline*100:.2f} cm")
    print("Compare measured baseline with the physical 3.45 cm before trusting distances.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", type=int, default=7, help="inner checkerboard columns")
    parser.add_argument("--rows", type=int, default=4, help="inner checkerboard rows")
    parser.add_argument(
        "--square-x-mm", type=float, default=32.675,
        help="measured horizontal square pitch in millimetres",
    )
    parser.add_argument(
        "--square-y-mm", type=float, default=33.075,
        help="measured vertical square pitch in millimetres",
    )
    args = parser.parse_args()
    calibrate(
        args.columns,
        args.rows,
        args.square_x_mm,
        args.square_y_mm,
    )
