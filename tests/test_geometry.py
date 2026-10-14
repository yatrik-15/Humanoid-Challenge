"""Round-trip tests for the conventions in src/utils/geometry.py."""

import numpy as np
import pytest

from src.utils.geometry import (
    backproject_depth,
    backproject_pixels,
    camera_center,
    invert_pose,
    look_at_T_cw,
    pixel_index,
    pixel_rays,
    project_points,
    rotation_aligning,
    to_homogeneous_4x4,
)

K = np.array([[500.0, 0, 320.0], [0, 520.0, 240.0], [0, 0, 1]])


def random_pose(rng: np.random.Generator) -> np.ndarray:
    eye = rng.uniform(-3, 3, 3)
    target = rng.uniform(-0.5, 0.5, 3)
    return look_at_T_cw(eye, target, up=np.array([0, 0, 1.0]))


def test_invert_pose_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = random_pose(rng)
        np.testing.assert_allclose(T @ invert_pose(T), np.eye(4), atol=1e-10)


def test_3x4_padding():
    T = random_pose(np.random.default_rng(1))
    np.testing.assert_allclose(to_homogeneous_4x4(T[:3]), T)


def test_look_at_points_optical_axis_at_target():
    eye, target = np.array([2.0, 1.0, 1.5]), np.array([0.0, 0.0, 0.5])
    T_cw = look_at_T_cw(eye, target, up=np.array([0, 0, 1.0]))
    np.testing.assert_allclose(camera_center(T_cw), eye, atol=1e-10)
    uv, z = project_points(target[None], K, T_cw)
    np.testing.assert_allclose(uv[0], [K[0, 2], K[1, 2]], atol=1e-8)
    assert z[0] == pytest.approx(np.linalg.norm(target - eye))


def test_look_at_world_up_is_image_up():
    """A point above the target must project to a smaller row v (image y points down)."""
    T_cw = look_at_T_cw(np.array([3.0, 0, 1.0]), np.zeros(3), up=np.array([0, 0, 1.0]))
    uv, _ = project_points(np.array([[0, 0, 0.0], [0, 0, 0.5]]), K, T_cw)
    assert uv[1, 1] < uv[0, 1]


def test_project_backproject_roundtrip():
    rng = np.random.default_rng(2)
    for _ in range(10):
        T_cw = random_pose(rng)
        X = rng.uniform(-0.5, 0.5, (100, 3))
        uv, z = project_points(X, K, T_cw)
        assert np.all(z > 0)
        X_rec = backproject_pixels(uv[:, 0], uv[:, 1], z, K, T_cw)
        np.testing.assert_allclose(X_rec, X, atol=1e-9)


def test_backproject_depth_map_skips_invalid_and_indexes_row_col():
    T_cw = look_at_T_cw(np.array([0, -2.0, 0]), np.zeros(3), up=np.array([0, 0, 1.0]))
    depth = np.zeros((480, 640))
    depth[100, 200] = 2.0  # row 100, col 200
    X, idx = backproject_depth(depth, K, T_cw)
    assert X.shape == (1, 3)
    assert tuple(idx[0]) == (100, 200)
    uv, z = project_points(X, K, T_cw)
    # COLMAP convention: centre of pixel (row 100, col 200) is (u, v) = (200.5, 100.5)
    np.testing.assert_allclose(uv[0], [200.5, 100.5], atol=1e-9)
    assert z[0] == pytest.approx(2.0)
    assert tuple(pixel_index(uv)[0]) == (100, 200)


def test_pixel_rays_hit_distance_is_z_depth():
    """A point at t along pixel_rays must project back to that pixel with z-depth t."""
    T_cw = random_pose(np.random.default_rng(5))
    H, W = 48, 64
    Ks = np.array([[60.0, 0, 32.0], [0, 62.0, 24.0], [0, 0, 1]])
    o, d = pixel_rays(Ks, T_cw, H, W)
    t = 2.5
    X = o + t * d
    uv, z = project_points(X, Ks, T_cw)
    np.testing.assert_allclose(z, t, atol=1e-9)
    idx = pixel_index(uv)
    rows, cols = np.mgrid[0:H, 0:W]
    np.testing.assert_array_equal(idx[:, 0], rows.ravel())
    np.testing.assert_array_equal(idx[:, 1], cols.ravel())


def test_rotation_aligning():
    rng = np.random.default_rng(3)
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = rotation_aligning(a, b)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0)
        np.testing.assert_allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-10)
    a = np.array([0, 0, 1.0])
    np.testing.assert_allclose(rotation_aligning(a, -a) @ a, -a, atol=1e-10)


def test_pycolmap_convention_matches():
    """pycolmap cam_from_world == our T_cw; projection_center == our camera_center."""
    pycolmap = pytest.importorskip("pycolmap")
    T_cw = random_pose(np.random.default_rng(4))
    rig = pycolmap.Rigid3d(pycolmap.Rotation3d(T_cw[:3, :3]), T_cw[:3, 3])
    np.testing.assert_allclose(to_homogeneous_4x4(rig.matrix()), T_cw, atol=1e-10)
    cam = pycolmap.Camera.create_from_model_name(1, "PINHOLE", 500.0, 640, 480)
    cam.params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    X = np.array([[0.1, -0.2, 0.3]])
    uv_ours, _ = project_points(X, K, T_cw)
    X_c = rig * X[0]
    uv_colmap = cam.img_from_cam(X_c)
    np.testing.assert_allclose(uv_ours[0], np.asarray(uv_colmap).ravel()[:2], atol=1e-8)


def test_in_box_handles_yaw_margin_and_footprint_only():
    from src.utils.geometry import in_box

    box = {"center": [1.0, 2.0, 0.5], "extents": [0.4, 0.2, 1.0], "yaw_rad": np.pi / 2}  # long side along y
    pts = np.array([[1.0, 2.15, 0.5], [1.15, 2.0, 0.5], [1.0, 2.0, 1.2]])
    assert in_box(box, pts).tolist() == [True, False, False]
    assert in_box(box, pts, margin=0.06).tolist() == [True, True, False]
    assert in_box(box, pts[:, :2]).tolist() == [True, False, True]  # footprint ignores height
