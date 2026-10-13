"""Gravity alignment and metric scale.

RANSAC finds the dominant horizontal plane in the sparse cloud (the floor, or a bed or
table top when the floor is not in view) and a similarity X' = s R X + t moves it to z = 0
with +z up. RANSAC thresholds are relative to the scene scale L (median distance of the
sparse points to the centroid of the camera centres), so they do not depend on the
arbitrary SfM scale.

Scale modes (alignment.scale_mode):
    metric_depth   median ratio of a metric monocular depth model to SfM depth over many
                   frames; approximate metres (default)
    camera_height  assumed camera height above the plane; only valid when the plane
                   really is the floor
    known          user-supplied metres per SfM unit
    normalized     L maps to a fixed number of scene units; not metric

Camera up in world coordinates is -R_wc[:, 1] (OpenCV cameras have +y pointing down).

Writes align/sparse, align/poses.json, align/alignment.json
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

from src.utils.geometry import camera_center, rotation_aligning, to_homogeneous_4x4


@dataclass
class FloorFit:
    """Result of the floor RANSAC. normal is unit length and oriented towards the cameras."""

    found: bool
    normal: np.ndarray  # (3,) plane normal n, plane: n . X + d = 0
    d: float
    inlier_ratio: float
    below_ratio: float  # fraction of points clearly below the plane
    tilt_to_camera_up_deg: float


def estimate_camera_up(T_cws: np.ndarray) -> np.ndarray:
    """Mean camera 'up' direction in world coordinates (unit vector).

    For an OpenCV camera, image-up is the -y axis; in world that is -R_wc[:, 1] = -R_cw[1, :].
    """
    ups = np.array([-T[1, :3] for T in T_cws])
    up = ups.mean(axis=0)
    return up / np.linalg.norm(up)


def scene_scale(points: np.ndarray, cam_centers: np.ndarray) -> float:
    """Scene scale L: median distance from sparse points to the camera centroid."""
    return float(np.median(np.linalg.norm(points - cam_centers.mean(axis=0), axis=1)))


def fit_floor(
    points: np.ndarray,
    cam_centers: np.ndarray,
    up_hint: np.ndarray,
    cfg: dict,
    rng: np.random.Generator | None = None,
) -> FloorFit:
    """RANSAC floor plane search.

    A candidate plane is accepted only if (a) its normal is within `max_tilt_deg` of the
    camera-up hint, (b) the camera centroid is above it, and (c) at most `max_below_ratio`
    of the points lie below it by more than the inlier threshold (rejects table tops,
    which have floor points under them). Among accepted planes the one with the most
    inliers wins, then it is refined by least squares (SVD) on its inliers.
    """
    rng = rng or np.random.default_rng(cfg.get("seed", 0))
    L = scene_scale(points, cam_centers)
    thr = cfg["floor_ransac_threshold_rel"] * L
    cos_tilt = np.cos(np.deg2rad(cfg["max_tilt_deg"]))
    cam_c = cam_centers.mean(axis=0)
    n_pts = len(points)

    best = None  # (n_inliers, normal, d)
    for _ in range(cfg["ransac_iterations"]):
        p = points[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        if np.dot(n, up_hint) < 0:
            n = -n
        if np.dot(n, up_hint) < cos_tilt:
            continue
        d = -np.dot(n, p[0])
        if np.dot(n, cam_c) + d <= 0:
            continue
        dist = points @ n + d
        n_in = int(np.sum(np.abs(dist) < thr))
        if np.mean(dist < -thr) > cfg["max_below_ratio"]:
            continue
        if best is None or n_in > best[0]:
            best = (n_in, n, d)

    if best is None:
        return FloorFit(False, up_hint.copy(), 0.0, 0.0, 0.0, 0.0)

    _, n, d = best
    inliers = points[np.abs(points @ n + d) < thr]
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid)
    n_ref = vt[-1]
    if np.dot(n_ref, up_hint) < 0:
        n_ref = -n_ref
    d_ref = -float(np.dot(n_ref, centroid))
    dist = points @ n_ref + d_ref
    inlier_ratio = float(np.mean(np.abs(dist) < thr))
    found = inlier_ratio >= cfg["floor_min_inliers_ratio"]
    return FloorFit(
        found=found,
        normal=n_ref,
        d=d_ref,
        inlier_ratio=inlier_ratio,
        below_ratio=float(np.mean(dist < -thr)),
        tilt_to_camera_up_deg=float(np.rad2deg(np.arccos(np.clip(np.dot(n_ref, up_hint), -1, 1)))),
    )


def metric_depth_scale(model: pycolmap.Reconstruction, image_dir: Path, cfg: dict) -> tuple[float, dict]:
    """Metres per SfM unit from a METRIC monocular depth model (Depth Anything V2 Metric-Indoor).

    For every `metric_frame_stride`-th registered frame: ratio_i = median over the frame's
    sparse points of (predicted metric z-depth / SfM z-depth). Scale = median of the ratio_i.
    Rotation/translation do not change camera z-depth ratios, so this runs on the unaligned
    model. Single frames can be far off (close-ups of a blank wall), hence the median
    across frames rather than a joint fit.

    Returns:
        scale (metres per SfM unit), stats dict (per-frame ratios, spread) for alignment.json
    """
    import cv2
    import torch

    from src.depth import DepthModel, sparse_depth_samples
    from src.utils.geometry import pixel_index

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DepthModel(cfg["metric_depth_model_id"], device)
    ratios = {}
    ids = sorted(model.reg_image_ids(), key=lambda i: model.image(i).name)
    for image_id in ids[:: cfg["metric_frame_stride"]]:
        name = model.image(image_id).name
        rgb = cv2.cvtColor(cv2.imread(str(image_dir / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        pred = net.predict(rgb)
        uv, z = sparse_depth_samples(model, image_id)
        if len(z) < cfg["metric_min_points"]:
            continue
        rc = pixel_index(uv)
        inside = (rc[:, 0] >= 0) & (rc[:, 0] < pred.shape[0]) & (rc[:, 1] >= 0) & (rc[:, 1] < pred.shape[1])
        d, z = pred[rc[inside, 0], rc[inside, 1]], z[inside]
        keep = (d > 0) & (z > 0)
        if keep.sum() < cfg["metric_min_points"]:
            continue
        ratios[name] = float(np.median(d[keep] / z[keep]))
    net.close()
    if not ratios:
        raise RuntimeError("metric_depth scale: no frame had enough sparse points")
    r = np.array(list(ratios.values()))
    s = float(np.median(r))
    q25, q75 = np.percentile(r, [25, 75])
    stats = {
        "model_id": cfg["metric_depth_model_id"],
        "num_frames": int(len(r)),
        "ratio_min": float(r.min()),
        "ratio_max": float(r.max()),
        "ratio_iqr_rel": float((q75 - q25) / s),
        "per_frame": {k: round(v, 6) for k, v in ratios.items()},
    }
    print(
        f"  metric_depth scale: {s:.5f} m/unit from {len(r)} frames "
        f"(min {r.min():.5f}, max {r.max():.5f}, IQR/median {stats['ratio_iqr_rel']:.3f})"
    )
    return s, stats


def compute_similarity(
    floor: FloorFit,
    points: np.ndarray,
    cam_centers: np.ndarray,
    up_hint: np.ndarray,
    cfg: dict,
    metric_scale: float = None,
) -> tuple[float, np.ndarray, np.ndarray, str, bool]:
    """Similarity (s, R, t) with X_new = s R X + t.

    After the transform: floor normal -> +z, floor at z = 0 (or, without a floor, the
    `fallback_floor_percentile` of heights along up_hint at z = 0), and the camera
    centroid projects to x = y = 0.

    Returns:
        s, R (3x3), t (3,), scale_mode_used, is_metric
    """
    normal = floor.normal if floor.found else up_hint
    R = rotation_aligning(normal, np.array([0.0, 0.0, 1.0]))
    rot_pts = points @ R.T
    rot_cams = cam_centers @ R.T
    if floor.found:
        z0 = -floor.d  # n . X = -d on the plane, and R maps n to +z
    else:
        z0 = float(np.percentile(rot_pts[:, 2], cfg["fallback_floor_percentile"]))

    mode = cfg["scale_mode"]
    if mode == "camera_height" and not floor.found:
        print("  WARNING: scale_mode=camera_height needs a floor; falling back to 'normalized'.")
        mode = "normalized"
    if mode == "camera_height":
        heights = rot_cams[:, 2] - z0
        s = cfg["assumed_camera_height_m"] / float(np.median(heights))
        is_metric = True  # approximate
    elif mode == "metric_depth":
        if metric_scale is None:
            raise ValueError("scale_mode=metric_depth needs metric_scale (see metric_depth_scale)")
        s = float(metric_scale)
        is_metric = True  # approximate (monocular metric depth)
    elif mode == "known":
        s = float(cfg["known_scale"])
        is_metric = True
    elif mode == "normalized":
        s = cfg["normalized_scene_scale"] / scene_scale(points, cam_centers)
        is_metric = False
    else:
        raise ValueError(f"Unknown scale_mode: {mode}")

    c = rot_cams.mean(axis=0)
    t = -s * np.array([c[0], c[1], z0])
    return s, R, t, mode, is_metric


def align_scene(out_dir: Path, config: dict) -> dict:
    """Fit the support plane, transform the undistorted model and export poses."""
    from src.sfm import export_poses

    src_model = out_dir / "sfm" / "undistorted" / "sparse"
    align_dir = out_dir / "align"
    (align_dir / "sparse").mkdir(parents=True, exist_ok=True)

    model = pycolmap.Reconstruction(src_model)
    points = np.array([p.xyz for p in model.points3D.values()])
    T_cws = np.array(
        [
            to_homogeneous_4x4(np.asarray(model.image(i).cam_from_world().matrix()))
            for i in model.reg_image_ids()
        ]
    )
    cams = np.array([camera_center(T) for T in T_cws])
    up_hint = estimate_camera_up(T_cws)

    floor = fit_floor(points, cams, up_hint, config)
    print(
        f"  Floor found: {floor.found}  inliers: {floor.inlier_ratio:.1%}  "
        f"below: {floor.below_ratio:.1%}  tilt vs camera-up: {floor.tilt_to_camera_up_deg:.1f} deg"
    )

    metric_scale, metric_stats = None, None
    if config["scale_mode"] == "metric_depth":
        metric_scale, metric_stats = metric_depth_scale(
            model, out_dir / "sfm" / "undistorted" / "images", config
        )
    s, R, t, mode, is_metric = compute_similarity(floor, points, cams, up_hint, config, metric_scale)
    err_before = model.compute_mean_reprojection_error()
    model.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t))
    err_after = model.compute_mean_reprojection_error()
    # A similarity transform must not change reprojection error (convention check).
    if abs(err_after - err_before) > 1e-6 * max(1.0, err_before):
        raise RuntimeError(f"Reprojection error changed under Sim3d: {err_before} -> {err_after}")
    model.write(align_dir / "sparse")

    new_cams = s * (cams @ R.T) + t
    report = {
        "floor_found": floor.found,
        "floor_inlier_ratio": round(floor.inlier_ratio, 4),
        "floor_below_ratio": round(floor.below_ratio, 4),
        "floor_tilt_to_camera_up_deg": round(floor.tilt_to_camera_up_deg, 2),
        "scale_mode": mode,
        "scale_is_metric": is_metric,
        "scale_note": {
            "camera_height": "APPROXIMATE metres from assumed camera height",
            "known": "metres from user-supplied scale",
            "metric_depth": "APPROXIMATE metres from monocular metric depth (median over frames)",
            "normalized": "scene units, not metric",
        }[mode],
        "scale": s,
        "R": R.tolist(),
        "t": t.tolist(),
        "camera_height_median": float(np.median(new_cams[:, 2])),
        "num_points3D": int(len(points)),
        "metric_depth_stats": metric_stats,
    }
    with open(align_dir / "alignment.json", "w") as f:
        json.dump(report, f, indent=2)
    export_poses(model, align_dir / "poses.json")
    print(
        f"  Scale mode: {mode} (metric={is_metric}), s={s:.4f}, "
        f"median camera height: {report['camera_height_median']:.3f}"
    )
    return report
