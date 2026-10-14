"""Synthetic end-to-end test of render-and-compare change detection.

Scene A: floor, walls, box1 (removed), box2 (static), box4 (static, but never observed in B,
so it must come out unknown rather than removed), box5 (moved) and box6 (a tall removed
pole whose top is far from any surface, which needs free-space carving to be confirmed).
Scene B: floor, walls, box2, box4, box5 at a new place and box3 (added).
A and B use different camera orbits, so the static parts only see a viewpoint change.

Runs the real Open3D fusion, mesh rendering, src.diff and src.cluster, plus an ablation
showing that the free-space check is what suppresses the box4 false positive.
"""

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from src.cluster import cluster_points, colors_in_box, hue_sat_histogram, pair_moved
from src.diff import accumulate_changes
from src.fusion import DepthRenderer, fuse, grid_cloud, grid_lookup
from tests.synthetic import Box, Scene, default_K, orbit_poses, pixel_rays, render

W, H = 160, 120
K = default_K(W, H)

FUSION = {
    "voxel_size": 0.02,
    "block_resolution": 16,
    "block_count": 50000,
    "depth_min": 0.1,
    "depth_max": 6.0,
    "sdf_trunc_multiplier": 6.0,
    "raycast_weight_threshold": 1.0,
    "extract_weight_threshold": 2.0,
    "carve_free_space": True,
    "carve_pixel_stride": 4,
    "carve_step_rel": 0.5,
    "edge_rel_jump": 0.05,
    "edge_erode_px": 1,
}
DIFF = {
    "tau_rel": 0.05,
    "tau_abs_min": 0.03,
    "min_views": 3,
    "evidence_voxel_size": 0.03,
    "free_tsdf_min": 0.3,
    "view_group_radius_rel": 0.05,
    "view_group_angle_deg": 10.0,
}
CLUSTER = {
    "dbscan_eps": 0.07,
    "dbscan_min_points": 5,
    "min_frontal_area": 0.005,
    "min_cluster_points": 10,
    "min_cluster_thickness": 0.10,
    "min_planarity": 0.15,
    "sheet_aspect": 3.0,
    "min_surface_gap": 0.06,
}
MOVED = {
    "max_volume_ratio": 2.0,
    "max_extent_ratio": 2.0,
    "min_appearance_sim": 0.3,
    "max_centroid_distance": 3.0,
    "hist_bins": 16,
    "appearance_box_margin": 0.02,
}

BOX1 = Box((-1.2, 0.6, 0.0), (-0.85, 0.95, 0.35), (0.9, 0.1, 0.1))  # removed
BOX2 = Box((-0.2, -0.2, 0.0), (0.2, 0.2, 0.5), (0.5, 0.5, 0.5))  # static
BOX3 = Box((0.7, 0.7, 0.0), (0.9, 0.9, 0.5), (0.1, 0.2, 0.9))  # added
BOX4 = Box((0.6, -1.3, 0.0), (0.9, -1.0, 0.3), (0.9, 0.9, 0.1))  # unseen in B
BOX5_A = Box((-1.1, -1.0, 0.0), (-0.8, -0.7, 0.25), (0.1, 0.8, 0.1))  # moved: before
BOX5_B = Box((-0.3, -1.4, 0.0), (0.0, -1.1, 0.25), (0.1, 0.8, 0.1))  # moved: after
BOX6 = Box((1.0, -0.2, 0.0), (1.15, -0.05, 1.2), (0.6, 0.3, 0.8))  # tall, removed

SCENE_A = Scene([BOX1, BOX2, BOX4, BOX5_A, BOX6], floor_extent=2.0)
POSES_A = orbit_poses((0, 0, 0), 2.6, 1.3, np.arange(0, 360, 15), target=(0, 0, 0.1))
POSES_B = orbit_poses((0.1, 0.1, 0), 2.3, 1.1, np.arange(7, 367, 15), target=(0, 0, 0.2))


def center(b: Box) -> np.ndarray:
    return (np.array(b.lo) + np.array(b.hi)) / 2


def rays_through_box(T_cw: np.ndarray, box: Box) -> np.ndarray:
    """(H, W) bool: pixel rays that intersect the box volume (slab test)."""
    o, d = pixel_rays(K, T_cw, H, W)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (np.array(box.lo) - o) / d
        t2 = (np.array(box.hi) - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return ((tmax >= tmin) & (tmax > 0)).reshape(H, W)


# B's geometry for rendering omits box4, and every B ray through box4's volume is blanked:
# B never observes that volume, yet sees the floor BEHIND it from other angles. Whether
# box4 exists in B is therefore unknowable -> must be UNKNOWN. This is the case where a
# naive "B sees farther than A" rule produces a false REMOVED.
SCENE_B_RENDER = Scene([BOX2, BOX3, BOX5_B], floor_extent=2.0)


def run_detector(gA, gB, renders, diff_cfg):
    lookA = grid_lookup(gA, FUSION["voxel_size"])
    lookB = grid_lookup(gB, FUSION["voxel_size"])
    ev = accumulate_changes(renders, lookA, lookB, diff_cfg, scene_scale=2.5)
    from scipy.spatial import cKDTree

    tA = cKDTree(grid_cloud(gA, 2.0)["points"])
    tB = cKDTree(grid_cloud(gB, 2.0)["points"])
    removed = cluster_points(ev["removed"]["points"], ev["removed"]["views"], CLUSTER, 0.03, tB)
    added = cluster_points(ev["added"]["points"], ev["added"]["views"], CLUSTER, 0.03, tA)
    return ev, removed, added


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(0)

    def frames(scene, poses, hide):
        for T in poses:
            d, c = render(scene, K, T, H, W)
            d = d * (1 + rng.normal(0, 0.005, d.shape)).astype(np.float32)
            if hide:
                d[rays_through_box(T, BOX4)] = 0.0
            yield d, c, K, T

    gA = fuse(frames(SCENE_A, POSES_A, False), FUSION)
    gB = fuse(frames(SCENE_B_RENDER, POSES_B, True), FUSION)
    rA, rB = DepthRenderer(gA, FUSION), DepthRenderer(gB, FUSION)
    renders = [(rA.render(K, T, W, H), rB.render(K, T, W, H), K, T) for T in POSES_A + POSES_B]

    ev, removed, added = run_detector(gA, gB, renders, DIFF)
    clouds = {"A": grid_cloud(gA, 2.0), "B": grid_cloud(gB, 2.0)}
    for c in removed:
        c["hist"] = hue_sat_histogram(
            colors_in_box(clouds["A"]["points"], clouds["A"]["colors"], c, 0.02), 16
        )
    for c in added:
        c["hist"] = hue_sat_histogram(
            colors_in_box(clouds["B"]["points"], clouds["B"]["colors"], c, 0.02), 16
        )
    pairs = pair_moved(removed, added, MOVED)

    _, removed_nocheck, _ = run_detector(gA, gB, renders, dict(DIFF, free_space_check=False))
    return {
        "ev": ev,
        "removed": removed,
        "added": added,
        "pairs": pairs,
        "clouds": clouds,
        "removed_nocheck": removed_nocheck,
    }


def _near(clusters, box, tol=0.15):
    return [c for c in clusters if np.linalg.norm(c["center"][:2] - center(box)[:2]) < tol]


def test_static_background_coincides(result):
    """Stage-4-style check on synthetic data: A and B clouds overlap on static surfaces."""
    from scipy.spatial import cKDTree

    A, B = result["clouds"]["A"]["points"], result["clouds"]["B"]["points"]
    assert np.median(cKDTree(B).query(A)[0]) < 2 * FUSION["voxel_size"]


def test_removed_box_detected(result):
    assert len(_near(result["removed"], BOX1)) == 1


def test_added_box_detected(result):
    assert len(_near(result["added"], BOX3)) == 1


def test_tall_removed_object_detected_full_height(result):
    """Needs free-space carving: the pole's top is ~1 m from any surface seen in B."""
    c = _near(result["removed"], BOX6)
    assert len(c) == 1
    assert c[0]["extents"][2] > 0.8, c[0]["extents"]


def test_moved_box_paired(result):
    r = _near(result["removed"], BOX5_A)
    a = _near(result["added"], BOX5_B)
    assert len(r) == 1 and len(a) == 1
    ri = next(i for i, c in enumerate(result["removed"]) if c is r[0])
    ai = next(i for i, c in enumerate(result["added"]) if c is a[0])
    assert any(p["removed_idx"] == ri and p["added_idx"] == ai for p in result["pairs"])


def test_unobserved_region_is_not_a_change(result):
    """box4 exists in both epochs but B never measured it: not REMOVED, but UNKNOWN."""
    assert _near(result["removed"], BOX4, tol=0.3) == []
    assert _near(result["added"], BOX4, tol=0.3) == []
    unk = result["ev"]["unknown"]["points"]
    assert np.any(np.linalg.norm(unk[:, :2] - center(BOX4)[:2], axis=1) < 0.3)


def test_ablation_without_free_space_check_box4_is_false_positive(result):
    """Sensitivity check: the box4 test above only passes BECAUSE of the free-space check."""
    assert len(_near(result["removed_nocheck"], BOX4, tol=0.3)) >= 1


def test_static_box_under_viewpoint_change_not_reported(result):
    assert _near(result["removed"], BOX2, tol=0.3) == []
    assert _near(result["added"], BOX2, tol=0.3) == []


def test_no_extra_clusters(result):
    """Exactly the planted changes: 3 removed (box1, box5_A, box6), 2 added (box3, box5_B)."""
    assert len(result["removed"]) == 3, [c["center"] for c in result["removed"]]
    assert len(result["added"]) == 2, [c["center"] for c in result["added"]]
