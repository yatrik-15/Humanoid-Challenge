"""
Per-frame alignment of relative (affine-invariant inverse-depth) predictions to SfM depth.

Depth Anything V2 (relative variant) predicts a quantity proportional to inverse depth up
to an unknown scale and shift: inv_depth ~= a * pred + b. For each frame we fit (a, b)
with RANSAC against the inverse z-depth of that frame's triangulated sparse points, then
refine by least squares on the inliers.

Conventions:
    - Sparse depth z: camera-frame z of a 3D point (aligned world frame), same units as
      the aligned world (approximate metres or scene units, see align/alignment.json).
    - Sparse sample location: COLMAP image coordinates (u, v); sampled at pixel
      floor(v), floor(u) of the prediction map (see utils.geometry.pixel_index).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DepthFit:
    """Result of aligning one frame. ok=False means the frame must not be fused."""

    ok: bool
    reason: str
    a: float
    b: float
    n_points: int
    inlier_ratio: float
    median_rel_residual: float  # median |z_aligned - z_sparse| / z_sparse over ALL points


def fit_inverse_depth_affine(
    pred: np.ndarray,
    z_sparse: np.ndarray,
    cfg: dict,
    rng: np.random.Generator | None = None,
) -> DepthFit:
    """Fit inv_z ~= a * pred + b with 2-point RANSAC.

    Args:
        pred: (N,) predicted relative inverse depth at the sparse points' pixels.
        z_sparse: (N,) sparse z-depths (> 0).
        cfg: depth.alignment config. Keys: min_sparse_points, ransac_inlier_threshold
            (relative: |a*pred+b - 1/z| < thr * median(1/z)), ransac_iterations,
            min_inlier_ratio, max_residual (median relative depth error).
    """
    rng = rng or np.random.default_rng(0)
    pred = np.asarray(pred, dtype=np.float64)
    z = np.asarray(z_sparse, dtype=np.float64)
    keep = np.isfinite(pred) & np.isfinite(z) & (z > 0)
    pred, z = pred[keep], z[keep]
    n = len(z)
    if n < cfg["min_sparse_points"]:
        return DepthFit(False, f"too few sparse points ({n})", 0.0, 0.0, n, 0.0, np.inf)

    y = 1.0 / z
    thr = cfg["ransac_inlier_threshold"] * float(np.median(y))
    best_inl = None
    for _ in range(cfg["ransac_iterations"]):
        i, j = rng.choice(n, 2, replace=False)
        if abs(pred[i] - pred[j]) < 1e-9:
            continue
        a = (y[i] - y[j]) / (pred[i] - pred[j])
        if a <= 0:  # inverse depth must increase with predicted disparity
            continue
        b = y[i] - a * pred[i]
        inl = np.abs(a * pred + b - y) < thr
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    if best_inl is None or best_inl.sum() < 2:
        return DepthFit(False, "RANSAC found no valid model", 0.0, 0.0, n, 0.0, np.inf)

    A = np.stack([pred[best_inl], np.ones(best_inl.sum())], axis=1)
    a, b = np.linalg.lstsq(A, y[best_inl], rcond=None)[0]
    inl = np.abs(a * pred + b - y) < thr
    inlier_ratio = float(inl.mean())

    inv = a * pred + b
    z_fit = np.where(inv > 0, 1.0 / np.maximum(inv, 1e-12), np.inf)
    med_res = float(np.median(np.abs(z_fit - z) / z))

    if a <= 0:
        return DepthFit(False, "non-positive scale", a, b, n, inlier_ratio, med_res)
    if inlier_ratio < cfg["min_inlier_ratio"]:
        return DepthFit(False, f"inlier ratio {inlier_ratio:.2f} too low", a, b, n, inlier_ratio, med_res)
    if med_res > cfg["max_residual"]:
        return DepthFit(False, f"median residual {med_res:.3f} too high", a, b, n, inlier_ratio, med_res)
    return DepthFit(True, "ok", float(a), float(b), n, inlier_ratio, med_res)


def apply_fit(pred_map: np.ndarray, fit: DepthFit, depth_max: float) -> np.ndarray:
    """Convert a relative prediction map to z-depth. Invalid / beyond depth_max -> 0."""
    inv = fit.a * pred_map.astype(np.float64) + fit.b
    depth = np.zeros_like(inv)
    valid = inv > 1.0 / depth_max
    depth[valid] = 1.0 / inv[valid]
    return depth.astype(np.float32)
