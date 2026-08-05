"""Tests for robust stereo distance selection."""

import numpy as np

from stereo_app import StereoCamera


def test_nearest_stable_region_ignores_sparse_false_near_pixels() -> None:
    depth = np.full((360, 640), np.inf, dtype=np.float32)
    valid = np.zeros_like(depth, dtype=bool)

    # A cup-sized coherent surface at 50 cm.
    depth[100:260, 220:400] = 0.50
    valid[100:260, 220:400] = True

    # Numerous incorrect near pixels that are not a coherent object.
    valid[20:320:8, 20:620:8] = True
    depth[20:320:8, 20:620:8] = 0.18

    distance, region = StereoCamera._nearest_stable_region(depth, valid)

    assert distance is not None
    assert abs(distance - 0.50) < 0.01
    assert region is not None
    assert int(region.sum()) > 20_000


def test_nearest_stable_region_requires_a_large_surface() -> None:
    depth = np.full((360, 640), 0.18, dtype=np.float32)
    valid = np.zeros_like(depth, dtype=bool)
    valid[10:25, 10:25] = True

    distance, region = StereoCamera._nearest_stable_region(depth, valid)

    assert distance is None
    assert region is None


def test_left_right_consistency_rejects_one_way_match() -> None:
    left = np.full((4, 12), 3.0, dtype=np.float32)
    right = np.full((4, 12), -3.0, dtype=np.float32)
    right[:, 5] = -8.0

    consistent = StereoCamera._left_right_consistency(left, right)

    assert consistent[:, 8].sum() == 0
    assert consistent[:, 7].all()
