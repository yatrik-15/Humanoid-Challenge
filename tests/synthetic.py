"""
Analytic synthetic scenes for tests: a floor plane plus axis-aligned boxes, rendered to
exact z-depth maps by ray casting in numpy. No external renderer needed.

Conventions follow src/utils/geometry.py (T_cw world->camera, OpenCV camera, COLMAP
image coordinates: pixel (r, c) centre at (c + 0.5, r + 0.5)).
"""

from dataclasses import dataclass

import numpy as np

from src.utils.geometry import look_at_T_cw, pixel_rays  # noqa: F401 (re-exported for tests)


@dataclass
class Box:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    color: tuple[float, float, float] = (0.8, 0.2, 0.2)


@dataclass
class Scene:
    boxes: list[Box]
    floor_extent: float = 3.0  # floor is the square |x|,|y| <= extent at z = 0
    floor_color: tuple[float, float, float] = (0.6, 0.6, 0.6)
    walls: bool = True  # vertical walls at x = +extent and y = +extent
    closed_room: bool = False  # also walls at x = -extent and y = -extent
    textured: bool = False  # modulate colors with 3-D value noise (for SfM)


def value_noise(X: np.ndarray, cell: float, seed: int) -> np.ndarray:
    """Smooth 3-D value noise in [0, 1] at points X (N, 3): hashed lattice + trilinear blend."""
    g = X / cell
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    f = f * f * (3 - 2 * f)  # smoothstep

    def h(ijk):
        v = (ijk[:, 0] * 73856093) ^ (ijk[:, 1] * 19349663) ^ (ijk[:, 2] * 83492791) ^ (seed * 2654435761)
        v = (v ^ (v >> 13)) * 1274126177
        return ((v ^ (v >> 16)) & 0xFFFF) / 65535.0

    out = np.zeros(len(X))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (
                    (f[:, 0] if dx else 1 - f[:, 0])
                    * (f[:, 1] if dy else 1 - f[:, 1])
                    * (f[:, 2] if dz else 1 - f[:, 2])
                )
                out += w * h(i0 + np.array([dx, dy, dz]))
    return out


def texture(X: np.ndarray) -> np.ndarray:
    """Multi-scale solid texture in [0.35, 1.0] (non-repeating, so features match uniquely)."""
    n = 0.5 * value_noise(X, 0.015, 1) + 0.3 * value_noise(X, 0.05, 2) + 0.2 * value_noise(X, 0.2, 3)
    return 0.35 + 0.65 * n


def render(
    scene: Scene, K: np.ndarray, T_cw: np.ndarray, H: int, W: int, max_depth: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """Render exact z-depth (0 = no hit) and RGB in [0, 1] for the scene.

    Because ray directions have camera-z = 1, the ray parameter t equals z-depth.
    """
    o, d = pixel_rays(K, T_cw, H, W)
    t_best = np.full(len(o), np.inf)
    col = np.zeros((len(o), 3))
    E = scene.floor_extent

    def plane_hit(axis: int, value: float, bounds: list[tuple[int, float, float]], color):
        nonlocal t_best, col
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (value - o[:, axis]) / d[:, axis]
        p = o + t[:, None] * d
        ok = (t > 1e-6) & np.isfinite(t)
        for ax, lo, hi in bounds:
            ok &= (p[:, ax] >= lo) & (p[:, ax] <= hi)
        better = ok & (t < t_best)
        t_best[better] = t[better]
        col[better] = color

    plane_hit(2, 0.0, [(0, -E, E), (1, -E, E)], scene.floor_color)
    if scene.walls:
        plane_hit(0, E, [(1, -E, E), (2, 0, 2.5)], (0.7, 0.7, 0.5))
        plane_hit(1, E, [(0, -E, E), (2, 0, 2.5)], (0.5, 0.7, 0.7))
    if scene.closed_room:
        plane_hit(0, -E, [(1, -E, E), (2, 0, 2.5)], (0.7, 0.5, 0.6))
        plane_hit(1, -E, [(0, -E, E), (2, 0, 2.5)], (0.6, 0.6, 0.8))

    for b in scene.boxes:
        lo, hi = np.array(b.lo), np.array(b.hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - o) / d
            t2 = (hi - o) / d
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        ok = (tmax >= tmin) & (tmin > 1e-6)
        better = ok & (tmin < t_best)
        t_best[better] = tmin[better]
        col[better] = b.color

    hit = np.isfinite(t_best) & (t_best < max_depth)
    depth = np.where(hit, t_best, 0.0)
    if scene.textured and hit.any():
        X = o[hit] + t_best[hit, None] * d[hit]
        col[hit] *= texture(X)[:, None]
    return depth.reshape(H, W).astype(np.float32), col.reshape(H, W, 3).astype(np.float32)


def orbit_poses(center, radius: float, height: float, angles_deg, target=None) -> list[np.ndarray]:
    """T_cw for cameras on a horizontal circle looking at `target` (default: center)."""
    center = np.asarray(center, dtype=np.float64)
    target = center if target is None else np.asarray(target, dtype=np.float64)
    poses = []
    for a in np.deg2rad(np.asarray(angles_deg, dtype=np.float64)):
        eye = center + np.array([radius * np.cos(a), radius * np.sin(a), 0.0])
        eye[2] = height
        poses.append(look_at_T_cw(eye, target, np.array([0.0, 0.0, 1.0])))
    return poses


def default_K(W: int = 160, H: int = 120, f: float = 120.0) -> np.ndarray:
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
