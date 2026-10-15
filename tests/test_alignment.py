"""Floor RANSAC + similarity tests on a synthetic room (no video needed)."""

import numpy as np
import pytest

from src.alignment import compute_similarity, estimate_camera_up, fit_floor
from src.utils.geometry import camera_center, look_at_T_cw

CFG = {
    "floor_ransac_threshold_rel": 0.01,
    "floor_min_inliers_ratio": 0.05,
    "max_tilt_deg": 30.0,
    "max_below_ratio": 0.05,
    "ransac_iterations": 2000,
    "seed": 0,
    "fallback_floor_percentile": 2.0,
    "scale_mode": "camera_height",
    "assumed_camera_height_m": 1.3,
    "known_scale": 1.0,
    "normalized_scene_scale": 2.0,
}


def synthetic_room(rng):
    """Metric room (z up, floor z=0): floor, a large table top at z=0.75, two walls.

    Table top has MORE points than the floor, so a naive 'largest plane' would pick it.
    Cameras at 1.3 m looking at the table.
    """
    floor = np.c_[rng.uniform(-2, 2, (600, 2)), np.zeros(600)]
    table = np.c_[rng.uniform(-0.8, 0.8, (900, 2)), np.full(900, 0.75)]
    wall1 = np.c_[np.full(500, 2.0), rng.uniform(-2, 2, 500), rng.uniform(0, 2.5, 500)]
    wall2 = np.c_[rng.uniform(-2, 2, 500), np.full(500, 2.0), rng.uniform(0, 2.5, 500)]
    pts = np.vstack([floor, table, wall1, wall2]) + rng.normal(0, 0.003, (2500, 3))
    T_cws = []
    for ang in np.linspace(0, np.pi, 12):
        eye = np.array([1.5 * np.cos(ang) - 0.5, 1.5 * np.sin(ang) - 1.5, 1.3])
        T_cws.append(look_at_T_cw(eye, np.array([0, 0, 0.6]), np.array([0, 0, 1.0])))
    return pts, np.array(T_cws)


def random_similarity(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return 0.37, R, rng.normal(size=3)


def to_sfm_frame(pts, T_cws, s, R, t):
    """Map metric room into an arbitrary 'SfM' frame: X' = s R X + t (cameras rescaled too)."""
    pts2 = s * pts @ R.T + t
    S = np.eye(4)
    S[:3, :3] = s * R
    S[:3, 3] = t
    S_inv = np.linalg.inv(S)
    out = []
    for T in T_cws:
        T2 = T @ S_inv  # world' -> metric camera frame (rotation part scaled by 1/s)
        T2[:3, :] *= s  # camera frame in SfM units: orthonormal R, scaled t
        out.append(T2)
    return pts2, np.array(out)


def test_floor_beats_larger_table_and_recovers_metric_frame():
    rng = np.random.default_rng(0)
    pts, T_cws = synthetic_room(rng)
    s0, R0, t0 = random_similarity(rng)
    pts_s, T_s = to_sfm_frame(pts, T_cws, s0, R0, t0)
    cams_s = np.array([camera_center(T) for T in T_s])
    cams_metric = np.array([camera_center(T) for T in T_cws])
    np.testing.assert_allclose(cams_s, s0 * cams_metric @ R0.T + t0, atol=1e-9)
    up = estimate_camera_up(T_s)

    floor = fit_floor(pts_s, cams_s, up, CFG)
    assert floor.found
    s, R, t, mode, is_metric = compute_similarity(floor, pts_s, cams_s, up, CFG)
    assert mode == "camera_height" and is_metric
    pts_new = s * pts_s @ R.T + t
    cams_new = s * cams_s @ R.T + t

    floor_pts = pts_new[:600]
    table_pts = pts_new[600:1500]
    assert np.abs(floor_pts[:, 2]).mean() < 0.01
    assert table_pts[:, 2].mean() == pytest.approx(0.75, abs=0.02)
    assert np.median(cams_new[:, 2]) == pytest.approx(1.3, abs=1e-6)
    assert abs(cams_new[:, :2].mean(axis=0)).max() < 1e-6


def test_no_floor_falls_back_to_normalized():
    rng = np.random.default_rng(1)
    pts, T_cws = synthetic_room(rng)
    pts = pts[600:]  # drop the floor; the table has floor-less space below it
    pts = pts[pts[:, 2] > 0.7]  # keep only the table top plus wall tops
    cams = np.array([camera_center(T) for T in T_cws])
    up = estimate_camera_up(T_cws)
    cfg = dict(CFG, max_below_ratio=-1.0)  # forbid every candidate
    floor = fit_floor(pts, cams, up, cfg)
    assert not floor.found
    s, R, t, mode, is_metric = compute_similarity(floor, pts, cams, up, cfg)
    assert mode == "normalized" and not is_metric


def test_camera_up_estimate():
    _, T_cws = synthetic_room(np.random.default_rng(2))
    np.testing.assert_allclose(estimate_camera_up(T_cws), [0, 0, 1], atol=0.35)


def test_camera_up_with_phone_pitched_down():
    """Walking with the phone tilted 35 deg down: image-up is off by 35 deg, the refined
    estimate (x-axes stay horizontal) is not."""
    from src.utils.geometry import look_at_T_cw

    rng = np.random.default_rng(3)
    T_cws = []
    for _ in range(40):
        eye = np.r_[rng.uniform(-1, 1, 2), 1.3]
        yaw = rng.uniform(-0.6, 0.6)
        d = np.r_[
            np.cos(yaw) * np.cos(np.deg2rad(35)),
            np.sin(yaw) * np.cos(np.deg2rad(35)),
            -np.sin(np.deg2rad(35)),
        ]
        T_cws.append(look_at_T_cw(eye, eye + d, np.array([0.0, 0.0, 1.0])))
    T_cws = np.array(T_cws)
    image_up = -T_cws[:, 1, :3].mean(axis=0)
    image_up /= np.linalg.norm(image_up)
    assert np.rad2deg(np.arccos(image_up[2])) > 30
    up = estimate_camera_up(T_cws)
    assert np.rad2deg(np.arccos(np.clip(up[2], -1, 1))) < 1.0


def test_camera_path_lengths_per_recording_in_frame_order():
    from src.alignment import camera_path_lengths

    names = ["B_0002.jpg", "A_0001.jpg", "A_0002.jpg", "B_0001.jpg", "A_0003.jpg"]
    centers = np.array([[0, 3, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 1, 0]], float)
    lengths = camera_path_lengths(names, centers)
    assert lengths == {"A": 2.0, "B": 3.0}


def test_pose_outliers_flags_a_frame_that_jumps_away_from_its_neighbours():
    from src.alignment import pose_outliers
    from src.utils.geometry import look_at_T_cw

    up = np.array([0.0, 0.0, 1.0])

    def pose(x, y=0.0):
        return {
            "epoch": "A",
            "T_cw": look_at_T_cw(np.array([x, y, 1.3]), np.array([x, 3.0, 0.5]), up).tolist(),
        }

    poses = {f"A_{i:04d}.jpg": pose(0.05 * i) for i in range(1, 11)}  # 5 cm steps sideways
    poses["A_0006.jpg"] = pose(0.30, 0.6)  # registered 60 cm off the path
    assert pose_outliers(poses, 0.15, 2.0) == ["A_0006.jpg"]
    fast = {f"A_{i:04d}.jpg": pose(0.4 * i) for i in range(1, 6)}  # quick but smooth motion
    assert pose_outliers(fast, 0.15, 2.0) == []
