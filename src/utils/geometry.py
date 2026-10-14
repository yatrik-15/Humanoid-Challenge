"""Geometry helpers. Conventions used throughout the repo:

- World frame: the shared SfM frame; after alignment +z is up.
- Camera frame: OpenCV (x right, y down, z forward).
- T_cw maps world to camera, X_c = T_cw @ X_w; it is pycolmap's
  Image.cam_from_world().matrix() padded to 4x4. T_wc = inv(T_cw); its translation is the
  camera centre.
- Image coordinates (u, v) follow COLMAP: u along columns, v along rows, (0, 0) is the
  top-left corner of the top-left pixel, so pixel (row r, col c) has its centre at
  (c + 0.5, r + 0.5). hloc shifts keypoints by +0.5 to match. Open3D's
  VoxelBlockGrid.integrate uses the same convention, so K is passed to it unchanged
  (test_integrate_uses_colmap_pixel_centres).
- Depth is camera-frame z, not ray length; 0 means invalid.
- K is the 3x3 pinhole matrix of the undistorted images.
"""

import numpy as np


def to_homogeneous_4x4(T: np.ndarray) -> np.ndarray:
    """Pad a 3x4 rigid transform to 4x4. A 4x4 input is returned as a float64 copy."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape == (4, 4):
        return T.copy()
    if T.shape != (3, 4):
        raise ValueError(f"Expected 3x4 or 4x4 transform, got {T.shape}")
    out = np.eye(4)
    out[:3, :] = T
    return out


def invert_pose(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform using R^T (exact for rotations)."""
    T = to_homogeneous_4x4(T)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def camera_center(T_cw: np.ndarray) -> np.ndarray:
    """Camera centre in world coordinates: C = -R^T t."""
    return invert_pose(T_cw)[:3, 3]


def transform_points(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply a 4x4 (or 3x4) rigid transform to (N, 3) points."""
    T = to_homogeneous_4x4(T)
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    return X @ T[:3, :3].T + T[:3, 3]


def project_points(X_w: np.ndarray, K: np.ndarray, T_cw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into an image.

    Returns:
        uv: (N, 2) pixel coordinates (u = column, v = row). Undefined where depth <= 0.
        depth: (N,) camera-frame z. Points behind the camera have depth <= 0.
    """
    X_c = transform_points(T_cw, X_w)
    z = X_c[:, 2]
    safe_z = np.where(np.abs(z) < 1e-12, 1e-12, z)
    uv_h = X_c @ np.asarray(K, dtype=np.float64).T
    uv = uv_h[:, :2] / safe_z[:, None]
    return uv, z


def backproject_pixels(
    u: np.ndarray, v: np.ndarray, depth: np.ndarray, K: np.ndarray, T_cw: np.ndarray
) -> np.ndarray:
    """Lift pixels with z-depth to world points.

    Args:
        u, v: (N,) pixel coordinates (column, row).
        depth: (N,) camera-frame z (must be > 0 to be meaningful).
        K: 3x3 intrinsics. T_cw: world->camera.

    Returns:
        (N, 3) world points.
    """
    K = np.asarray(K, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    z = np.asarray(depth, dtype=np.float64).ravel()
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    X_c = np.stack([x, y, z], axis=1)
    return transform_points(invert_pose(T_cw), X_c)


def backproject_depth(
    depth: np.ndarray, K: np.ndarray, T_cw: np.ndarray, mask: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a full depth map (H, W) to world points.

    Pixels with depth <= 0 (invalid) are always excluded; `mask` further restricts.

    Returns:
        X_w: (N, 3) world points.
        idx: (N, 2) integer (row, col) of each returned pixel.
    """
    valid = depth > 0
    if mask is not None:
        valid &= mask.astype(bool)
    rows, cols = np.nonzero(valid)
    X_w = backproject_pixels(cols + 0.5, rows + 0.5, depth[rows, cols], K, T_cw)
    return X_w, np.stack([rows, cols], axis=1)


def pixel_index(uv: np.ndarray) -> np.ndarray:
    """(N, 2) image coordinates (u, v) -> (N, 2) integer (row, col) of the containing pixel."""
    uv = np.asarray(uv, dtype=np.float64)
    return np.stack([np.floor(uv[:, 1]), np.floor(uv[:, 0])], axis=1).astype(np.int64)


def pixel_rays(K: np.ndarray, T_cw: np.ndarray, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """World-frame ray origins and directions through every pixel centre, row-major (H*W, 3).

    Directions are scaled so their CAMERA-frame z component is 1, so a hit distance t
    along the ray (in units of the direction length) equals z-depth.
    """
    K = np.asarray(K, dtype=np.float64)
    rows, cols = np.mgrid[0:H, 0:W]
    x = (cols + 0.5 - K[0, 2]) / K[0, 0]
    y = (rows + 0.5 - K[1, 2]) / K[1, 1]
    d_cam = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
    T_wc = invert_pose(T_cw)
    d_w = d_cam @ T_wc[:3, :3].T
    o_w = np.broadcast_to(T_wc[:3, 3], d_w.shape)
    return o_w, d_w


def look_at_T_cw(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build T_cw for an OpenCV camera at `eye` looking at `target`.

    `up` is the approximate world up direction; the camera's -y axis is aligned to it.
    Used by the synthetic tests and the overview render.
    """
    eye = np.asarray(eye, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - eye
    z /= np.linalg.norm(z)
    # OpenCV: y points down (~ -up) and x = y cross z = z cross up.
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("up vector is parallel to the viewing direction")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_wc = np.stack([x, y, z], axis=1)  # columns = camera axes in world
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = eye
    return invert_pose(T_wc)


def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation R (3x3) with R @ a_hat = b_hat (Rodrigues formula)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate about any axis perpendicular to a
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
