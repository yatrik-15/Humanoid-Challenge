"""Unit tests for fusion pre-processing helpers."""

import numpy as np
import pytest

pytest.importorskip("open3d")

from src.fusion import erode_depth_edges, free_space_points
from src.utils.geometry import look_at_T_cw, project_points


def test_erode_depth_edges_removes_band_around_jump_and_holes():
    d = np.full((20, 20), 2.0, np.float32)
    d[:, 10:] = 3.0  # vertical discontinuity between cols 9 and 10
    d[5, 3] = 0.0  # an invalid pixel
    out = erode_depth_edges(d, rel_jump=0.05, erode_px=1)
    assert np.all(out[:, 8:12] == 0)  # cols 9,10 flagged, dilated by 1
    assert np.all(out[:, :7][out[:, :7] > 0] == 2.0)
    assert out[4:7, 2:5].max() == 0  # hole dilated by 1
    assert out[15, 2] == 2.0 and out[15, 15] == 3.0


def test_erode_keeps_smooth_slanted_surface():
    d = np.tile(np.linspace(1.0, 1.5, 50, dtype=np.float32), (30, 1))  # 1% steps
    out = erode_depth_edges(d, rel_jump=0.05, erode_px=2)
    assert np.all(out[:, :] == d)


def test_free_space_points_lie_strictly_in_front_of_surface():
    K = np.array([[50.0, 0, 20], [0, 50.0, 15], [0, 0, 1]])
    T_cw = look_at_T_cw(np.array([0, -3.0, 1.0]), np.zeros(3), np.array([0, 0, 1.0]))
    depth = np.full((30, 40), 2.5, np.float32)
    cfg = {
        "carve_pixel_stride": 4,
        "voxel_size": 0.02,
        "sdf_trunc_multiplier": 6.0,
        "carve_step_rel": 0.5,
        "block_resolution": 16,
        "depth_min": 0.1,
        "depth_max": 5.0,
    }
    P = free_space_points(depth, K, T_cw, cfg)
    assert len(P) > 0
    _, z = project_points(P, K, T_cw)
    assert z.min() >= 0.1 - 1e-9
    assert z.max() < 2.5 - 0.02 * 6.0 + 1e-9


def test_integrate_uses_colmap_pixel_centres():
    """Fusing exact depth with K unchanged must beat K shifted by half a pixel.

    Pins down the convention in utils/geometry.py: Open3D's integrate treats pixel (r, c)
    as centred at (c + 0.5, r + 0.5), like COLMAP, so no K offset is needed.
    """
    pytest.importorskip("open3d")
    from src.fusion import DepthRenderer, fuse
    from tests.synthetic import Box, Scene, default_K, orbit_poses, render

    W, H = 160, 120
    K = default_K(W, H)
    scene = Scene([Box((-0.3, -0.3, 0.0), (0.3, 0.3, 0.6))], floor_extent=2.0)
    poses = orbit_poses((0, 0, 0), 2.4, 1.2, np.arange(0, 360, 20), target=(0, 0, 0.2))
    probe = orbit_poses((0, 0, 0), 2.2, 1.0, [35.0], target=(0, 0, 0.2))[0]
    truth, _ = render(scene, K, probe, H, W)
    cfg = {
        "voxel_size": 0.02,
        "block_resolution": 16,
        "block_count": 20000,
        "depth_min": 0.1,
        "depth_max": 6.0,
        "sdf_trunc_multiplier": 6.0,
        "raycast_weight_threshold": 1.0,
        "carve_free_space": False,
        "carve_pixel_stride": 4,
        "carve_step_rel": 0.5,
        "edge_rel_jump": 0.05,
        "edge_erode_px": 1,
    }

    def median_error(shift: float) -> float:
        Ks = K.copy()
        Ks[:2, 2] += shift
        frames = ((*render(scene, K, T, H, W), Ks, T) for T in poses)
        depth = DepthRenderer(fuse(frames, cfg), cfg).render(K, probe, W, H)
        ok = (depth > 0) & (truth > 0)
        return float(np.median(np.abs(depth[ok] - truth[ok])))

    err = median_error(0.0)
    assert err < 0.005
    assert err < 0.5 * min(median_error(-0.5), median_error(0.5))


def test_snap_to_floor_flattens_floor_but_keeps_upright_object():
    """A floor rendered 8 % too deep is put back on z = 0; a box face standing on it is not."""
    from src.fusion import snap_to_floor
    from src.utils.geometry import look_at_T_cw

    H, W = 120, 160
    K = np.array([[150.0, 0, W / 2], [0, 150.0, H / 2], [0, 0, 1]])
    eye = np.array([0.0, 0.0, 1.3])
    T_cw = look_at_T_cw(eye, np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    R_wc = T_cw[:3, :3].T
    v, u = np.mgrid[0:H, 0:W]
    rays_w = np.stack([(u + 0.5 - W / 2) / 150.0, (v + 0.5 - H / 2) / 150.0, np.ones((H, W))], -1) @ R_wc.T
    t_floor = -eye[2] / rays_w[..., 2]
    t_wall = 1.5 / rays_w[..., 1]  # vertical face at y = 1.5, 0.4 m tall
    wall_z = eye[2] + t_wall * rays_w[..., 2]
    on_wall = (t_wall > 0) & (wall_z > 0) & (wall_z < 0.4) & ((t_wall < t_floor) | (t_floor <= 0))
    depth = np.where(t_floor > 0, t_floor * 1.08, 0.0)
    depth[on_wall] = t_wall[on_wall]

    out, mask = snap_to_floor(depth, K, T_cw, band=(-0.4, 0.2), max_normal_deg=30.0, step=2)
    floor_px = (t_floor > 0) & ~on_wall
    inner = np.zeros_like(mask)
    inner[4:-4, 4:-4] = True
    snapped_floor = mask & floor_px & inner
    assert snapped_floor.sum() > 0.8 * (floor_px & inner).sum()
    np.testing.assert_allclose(out[snapped_floor], t_floor[snapped_floor], rtol=1e-5)
    assert not mask[on_wall & inner].any()
