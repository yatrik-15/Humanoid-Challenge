"""TSDF fusion of each recording into its own Open3D VoxelBlockGrid, and mesh rendering.

Both grids share the aligned world frame and voxel size. `integrate` takes T_cw as the
extrinsic and uses the same pixel-centre convention as COLMAP, so K is passed unchanged.
Stored TSDF values are normalized to [-1, 1] and positive in free space.

Rendering does not use VoxelBlockGrid.ray_cast: with open3d 0.20 on Windows it hangs for
some range_map_down_factor values and returns different hits on repeated calls for others.
Instead the grid's zero-level mesh is ray-cast with RaycastingScene. The mesh covers every
allocated block, so objects that were later removed are still rendered.

Conventions: src/utils/geometry.py. Depth maps are float32 z-depth in world units.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c

from src.diff import VoxelLookup
from src.utils.geometry import invert_pose, pixel_rays, transform_points

DEPTH_SCALE = 1.0  # depth maps are already in world units


def make_grid(cfg: dict) -> o3d.t.geometry.VoxelBlockGrid:
    """Empty TSDF grid with tsdf, weight and color attributes."""
    return o3d.t.geometry.VoxelBlockGrid(
        ("tsdf", "weight", "color"),
        (o3c.float32, o3c.float32, o3c.float32),
        ((1), (1), (3)),
        cfg["voxel_size"],
        cfg["block_resolution"],
        cfg["block_count"],
        o3c.Device("CPU:0"),
    )


def integrate_frame(
    grid: o3d.t.geometry.VoxelBlockGrid,
    depth: np.ndarray,
    color: np.ndarray,
    K: np.ndarray,
    T_cw: np.ndarray,
    cfg: dict,
) -> None:
    """Integrate one z-depth map (H, W) float32 and RGB (H, W, 3) float32 in [0, 1].

    Open3D only allocates blocks near observed surfaces, so empty space far from any
    surface would stay unallocated (= "never observed"). With cfg["carve_free_space"],
    blocks along a subsample of the rays (camera -> just before the surface) are also
    allocated; integrate() then records those voxels as observed free space (tsdf = +1).
    This is what lets the change detector confirm REMOVED for parts of objects that are
    far from the surfaces seen behind them in the other epoch.
    """
    depth = erode_depth_edges(depth, cfg["edge_rel_jump"], cfg["edge_erode_px"])
    Ki = o3c.Tensor(np.asarray(K, dtype=np.float64), o3c.float64)
    ext = o3c.Tensor(np.asarray(T_cw, dtype=np.float64), o3c.float64)
    d_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth, dtype=np.float32)))
    c_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(color, dtype=np.float32)))
    blocks = grid.compute_unique_block_coordinates(
        d_img, Ki, ext, DEPTH_SCALE, cfg["depth_max"], cfg["sdf_trunc_multiplier"]
    )
    if cfg.get("carve_free_space", False):
        free = free_space_points(depth, K, T_cw, cfg)
        if len(free):
            pcd = o3d.t.geometry.PointCloud(o3c.Tensor(free.astype(np.float32)))
            extra = grid.compute_unique_block_coordinates(pcd, cfg["sdf_trunc_multiplier"])
            merged = np.unique(np.vstack([blocks.numpy(), extra.numpy()]), axis=0)
            blocks = o3c.Tensor(merged.astype(np.int32))
    grid.integrate(
        blocks, d_img, c_img, Ki, Ki, ext, DEPTH_SCALE, cfg["depth_max"], cfg["sdf_trunc_multiplier"]
    )


def erode_depth_edges(depth: np.ndarray, rel_jump: float, erode_px: int) -> np.ndarray:
    """Invalidate (set 0) pixels within `erode_px` of a depth discontinuity or invalid pixel.

    A discontinuity is a 4-neighbour pair whose depths differ by more than
    rel_jump * min(depth). Rays just beside an object's silhouette would otherwise carve
    the object's edge as free space (projective TSDF), and monocular depth is least
    reliable there ("flying pixels").
    """
    if erode_px <= 0:
        return depth
    import cv2

    d = depth.astype(np.float32)
    bad = d <= 0
    for a, b, sl_a, sl_b in (
        (d[:, :-1], d[:, 1:], (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        (d[:-1, :], d[1:, :], (slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        jump = np.abs(a - b) > rel_jump * np.minimum(a, b)
        bad[sl_a] |= jump
        bad[sl_b] |= jump
    kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    bad = cv2.dilate(bad.astype(np.uint8), kernel) > 0
    out = d.copy()
    out[bad] = 0.0
    return out


def free_space_points(depth: np.ndarray, K: np.ndarray, T_cw: np.ndarray, cfg: dict) -> np.ndarray:
    """World points sampled along rays of every `carve_pixel_stride`-th valid pixel, from
    depth_min to (depth - truncation distance), spaced `carve_step_rel` * block edge."""
    s = cfg["carve_pixel_stride"]
    sub = depth[::s, ::s]
    rows, cols = np.nonzero(sub > 0)
    if len(rows) == 0:
        return np.zeros((0, 3))
    z_end = sub[rows, cols] - cfg["voxel_size"] * cfg["sdf_trunc_multiplier"]
    step = cfg["carve_step_rel"] * cfg["voxel_size"] * cfg["block_resolution"]
    n_steps = int(np.ceil(cfg["depth_max"] / step))
    zs = cfg["depth_min"] + step * np.arange(n_steps)  # (S,)
    Kf = np.asarray(K, dtype=np.float64)
    x = (cols * s + 0.5 - Kf[0, 2]) / Kf[0, 0]
    y = (rows * s + 0.5 - Kf[1, 2]) / Kf[1, 1]
    Z = zs[None, :] * np.ones((len(rows), 1))  # (P, S)
    keep = Z < z_end[:, None]
    P_cam = np.stack([x[:, None] * Z, y[:, None] * Z, Z], axis=-1)[keep]
    return transform_points(invert_pose(T_cw), P_cam)


def fuse(frames: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], cfg: dict):
    """Fuse (depth, color, K, T_cw) frames into a new grid."""
    grid = make_grid(cfg)
    for depth, color, K, T_cw in frames:
        integrate_frame(grid, depth, color, K, T_cw, cfg)
    return grid


class DepthRenderer:
    """Renders z-depth of a grid's zero-level surface (all allocated blocks) from any pose."""

    def __init__(self, grid: o3d.t.geometry.VoxelBlockGrid, cfg: dict):
        mesh = grid.extract_triangle_mesh(cfg["raycast_weight_threshold"])
        self.num_triangles = int(mesh.triangle.indices.shape[0])
        self.scene = o3d.t.geometry.RaycastingScene()
        if self.num_triangles:
            self.scene.add_triangles(
                o3c.Tensor(mesh.vertex.positions.numpy().astype(np.float32)),
                o3c.Tensor(mesh.triangle.indices.numpy().astype(np.uint32)),
            )
        self.depth_min = cfg["depth_min"]
        self.depth_max = cfg["depth_max"]

    def render(self, K: np.ndarray, T_cw: np.ndarray, width: int, height: int) -> np.ndarray:
        """(H, W) float32 z-depth, 0 = no hit or outside [depth_min, depth_max]."""
        if not self.num_triangles:
            return np.zeros((height, width), np.float32)
        o, d = pixel_rays(K, T_cw, height, width)
        rays = o3c.Tensor(np.concatenate([o, d], axis=1).astype(np.float32))
        t = self.scene.cast_rays(rays)["t_hit"].numpy().reshape(height, width)
        ok = np.isfinite(t) & (t >= self.depth_min) & (t <= self.depth_max)
        return np.where(ok, t, 0.0).astype(np.float32)


def grid_lookup(grid: o3d.t.geometry.VoxelBlockGrid, voxel_size: float) -> VoxelLookup:
    """Free-space lookup over observed voxels (weight > 0) of a grid."""
    coords, idx = grid.voxel_coordinates_and_flattened_indices()
    idx = idx.numpy().ravel()
    weight = grid.attribute("weight").reshape((-1,)).numpy()[idx]
    obs = weight > 0
    tsdf = grid.attribute("tsdf").reshape((-1,)).numpy()[idx][obs]
    ijk = np.round(coords.numpy()[obs] / voxel_size).astype(np.int64)
    return VoxelLookup(ijk, tsdf, weight[obs], voxel_size)


def grid_cloud(grid: o3d.t.geometry.VoxelBlockGrid, weight_threshold: float) -> dict[str, np.ndarray]:
    """Surface point cloud (points, colors in [0, 1]) extracted at the TSDF zero crossing."""
    pcd = grid.extract_point_cloud(weight_threshold)
    pts = pcd.point.positions.numpy().astype(np.float64)
    cols = pcd.point.colors.numpy().astype(np.float64) if "colors" in pcd.point else np.zeros_like(pts)
    return {"points": pts, "colors": cols}


def fuse_depths(out_dir: Path, cfg: dict) -> dict[str, dict]:
    """Stage entry: fuse aligned depth maps of each epoch into grid A and grid B.

    Inputs:  align/poses.json, depth/<frame>.npy (z-depth, 0 = invalid, only frames that
             passed alignment), sfm/undistorted/images/<frame>.
    Outputs: fusion/grid_{A,B}.npz (VoxelBlockGrid.save), fusion/cloud_{A,B}.npz,
             fusion/cloud_{A,B}.ply, fusion/fusion_report.json.
    """
    import cv2

    poses = json.load(open(out_dir / "align" / "poses.json"))
    depth_dir = out_dir / "depth"
    img_dir = out_dir / "sfm" / "undistorted" / "images"
    fdir = out_dir / "fusion"
    fdir.mkdir(parents=True, exist_ok=True)

    report = {}
    for epoch in ("A", "B"):
        names = sorted(
            n
            for n, p in poses.items()
            if p["epoch"] == epoch and (depth_dir / f"{Path(n).stem}.npy").exists()
        )
        if not names:
            raise RuntimeError(f"No aligned depth maps for epoch {epoch}; cannot fuse.")

        def frames(names):
            for n in names:
                depth = np.load(depth_dir / f"{Path(n).stem}.npy")
                bgr = cv2.imread(str(img_dir / n), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                if rgb.shape[:2] != depth.shape:
                    raise ValueError(f"{n}: image {rgb.shape[:2]} vs depth {depth.shape}")
                yield depth, rgb, np.array(poses[n]["K"]), np.array(poses[n]["T_cw"])

        grid = fuse(frames(names), cfg)
        grid.save(str(fdir / f"grid_{epoch}.npz"))
        cloud = grid_cloud(grid, cfg["extract_weight_threshold"])
        np.savez_compressed(fdir / f"cloud_{epoch}.npz", **cloud)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud["points"]))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(cloud["colors"], 0, 1))
        o3d.io.write_point_cloud(str(fdir / f"cloud_{epoch}.ply"), pcd)
        report[epoch] = {
            "frames_fused": len(names),
            "blocks": int(grid.hashmap().size()),
            "cloud_points": int(len(cloud["points"])),
        }
        print(
            f"  [{epoch}] fused {len(names)} frames, {report[epoch]['blocks']} blocks, "
            f"{report[epoch]['cloud_points']} surface points"
        )

    report["background_overlap"] = cloud_overlap(fdir, cfg)
    with open(fdir / "fusion_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def cloud_overlap(fdir: Path, cfg: dict) -> dict:
    """How well the static surfaces of A and B coincide (a check on the shared frame).

    For each point of cloud A, the distance to the nearest point of cloud B (and vice
    versa). Changes affect only a small fraction of points, so the median reflects the
    static background; it should be on the order of the voxel size.
    """
    from scipy.spatial import cKDTree

    A = np.load(fdir / "cloud_A.npz")["points"]
    B = np.load(fdir / "cloud_B.npz")["points"]
    dAB = cKDTree(B).query(A)[0]
    dBA = cKDTree(A).query(B)[0]
    vs = cfg["voxel_size"]
    return {
        "median_A_to_B": float(np.median(dAB)),
        "median_B_to_A": float(np.median(dBA)),
        "frac_A_within_2_voxels": float(np.mean(dAB < 2 * vs)),
        "frac_B_within_2_voxels": float(np.mean(dBA < 2 * vs)),
        "voxel_size": vs,
    }
