"""Tests for clustering, gravity OBB and MOVED pairing (no video needed)."""

import numpy as np
import pytest

from src.cluster import cluster_points, gravity_obb, hue_sat_histogram, pair_moved

CCFG = {
    "dbscan_eps": 0.06,
    "dbscan_min_points": 5,
    "min_frontal_area": 0.005,
    "min_cluster_points": 20,
    "min_cluster_thickness": 0.10,
    "sheet_aspect": 3.0,
    "min_surface_gap": 0.06,
}
MCFG = {
    "max_volume_ratio": 2.0,
    "max_extent_ratio": 2.0,
    "min_appearance_sim": 0.3,
    "max_centroid_distance": 3.0,
}
VOX = 0.03


def box_surface(center, size, rng, step=VOX):
    """Evidence-like samples on the visible surface (top + 4 sides) of a box."""
    cx, cy, cz = center
    sx, sy, sz = size
    pts = []
    for x in np.arange(-sx / 2, sx / 2 + 1e-9, step):
        for y in np.arange(-sy / 2, sy / 2 + 1e-9, step):
            pts.append([cx + x, cy + y, cz + sz / 2])
    for z in np.arange(-sz / 2, sz / 2 + 1e-9, step):
        for x in np.arange(-sx / 2, sx / 2 + 1e-9, step):
            pts += [[cx + x, cy - sy / 2, cz + z], [cx + x, cy + sy / 2, cz + z]]
        for y in np.arange(-sy / 2, sy / 2 + 1e-9, step):
            pts += [[cx - sx / 2, cy + y, cz + z], [cx + sx / 2, cy + y, cz + z]]
    return np.array(pts) + rng.normal(0, 0.002, (len(pts), 3))


def test_gravity_obb_recovers_rotated_box():
    rng = np.random.default_rng(0)
    pts = box_surface((0, 0, 0.2), (0.6, 0.3, 0.4), rng)
    yaw = np.deg2rad(30)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    pts = pts @ R.T + np.array([1.0, 2.0, 0.0])
    box = gravity_obb(pts)
    np.testing.assert_allclose(box["center"], [1.0, 2.0, 0.2], atol=0.02)
    np.testing.assert_allclose(sorted(box["extents"][:2]), [0.3, 0.6], atol=0.02)
    assert box["extents"][2] == pytest.approx(0.4, abs=0.02)
    assert np.rad2deg(box["yaw"]) % 180 == pytest.approx(30, abs=3)


def test_cluster_separates_objects_and_drops_noise():
    rng = np.random.default_rng(1)
    a = box_surface((0, 0, 0.15), (0.3, 0.3, 0.3), rng)
    b = box_surface((1.5, 0, 0.1), (0.4, 0.2, 0.2), rng)
    noise = rng.uniform(-3, 3, (15, 3))  # sparse isolated voxels
    pts = np.vstack([a, b, noise])
    views = np.full(len(pts), 4)
    clusters = cluster_points(pts, views, CCFG, VOX)
    assert len(clusters) == 2
    centers = sorted(c["center"][0] for c in clusters)
    assert centers[0] == pytest.approx(0.0, abs=0.03)
    assert centers[1] == pytest.approx(1.5, abs=0.03)


def test_flat_sheet_rejected_box_kept():
    """A sheet of evidence on a wall (depth bias) is dropped; a box with thickness is kept."""
    rng = np.random.default_rng(4)
    yy, zz = np.meshgrid(np.arange(0, 1.2, VOX), np.arange(0, 0.6, VOX))
    sheet = np.c_[np.full(yy.size, 2.0) + rng.normal(0, 0.01, yy.size), yy.ravel(), zz.ravel()]
    box = box_surface((0, 0, 0.15), (0.3, 0.3, 0.3), rng)
    clusters = cluster_points(np.vstack([sheet, box]), np.full(len(sheet) + len(box), 5), CCFG, VOX)
    assert len(clusters) == 1
    assert clusters[0]["center"][0] == pytest.approx(0.0, abs=0.05)
    assert clusters[0]["principal_extents"][0] > 0.2


def _cluster(center, size, hist):
    return {
        "center": np.array(center, float),
        "extents": np.array(size, float),
        "box_volume": float(np.prod(size)),
        "hist": hist,
    }


def test_pair_moved_matches_similar_object_only():
    red = hue_sat_histogram(np.tile([0.9, 0.1, 0.1], (50, 1)), 16)
    blue = hue_sat_histogram(np.tile([0.1, 0.1, 0.9], (50, 1)), 16)
    removed = [_cluster((0, 0, 0.2), (0.3, 0.3, 0.4), red)]
    added = [
        _cluster((1, 1, 0.2), (0.3, 0.3, 0.4), blue),  # same size, wrong colour
        _cluster((1.5, 0, 0.2), (0.32, 0.29, 0.4), red),  # the moved object
        _cluster((0.5, 0, 0.5), (1.2, 1.0, 1.0), red),  # right colour, far too big
    ]
    pairs = pair_moved(removed, added, MCFG)
    assert len(pairs) == 1 and pairs[0]["added_idx"] == 1
    assert pairs[0]["appearance_similarity"] == pytest.approx(1.0)


def test_pair_moved_respects_distance_limit():
    h = hue_sat_histogram(np.tile([0.2, 0.7, 0.2], (10, 1)), 16)
    removed = [_cluster((0, 0, 0), (0.3, 0.3, 0.3), h)]
    added = [_cluster((10, 0, 0), (0.3, 0.3, 0.3), h)]
    assert pair_moved(removed, added, MCFG) == []
