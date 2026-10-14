"""Projected change boxes on the video frames: side_by_side.mp4."""

import json
from pathlib import Path

import numpy as np

from src.utils.geometry import camera_center, project_points
from src.viz.report import action_color, change_boxes, shown_in


def draw_box(img: np.ndarray, box: dict, K: np.ndarray, T_cw: np.ndarray, color, text: str) -> None:
    """Draw a projected 3-D box (12 edges) if all its corners are in front of the camera."""
    import cv2

    C = np.array(box["corners"])
    uv, z = project_points(C, K, T_cw)
    if np.any(z <= 0):
        return
    H, W = img.shape[:2]
    if uv[:, 0].max() < 0 or uv[:, 0].min() >= W or uv[:, 1].max() < 0 or uv[:, 1].min() >= H:
        return
    # corners are ordered by (dx, dy, dz) in {-,+}^3 (see cluster.gravity_obb)
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    bgr = tuple(int(c) for c in color[::-1])
    pts = np.round(uv).astype(int)
    for a, b in edges:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), bgr, 2, cv2.LINE_AA)
    top = pts[np.argmin(pts[:, 1])]
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    x = int(np.clip(top[0] - tw // 2, 0, max(0, W - tw - 4)))
    y = int(np.clip(top[1] - 6, th + 4, H - base))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base), (20, 20, 20), -1)  # label plate
    cv2.putText(img, text, (x + 2, y - 1), font, scale, bgr, 1, cv2.LINE_AA)


def write_side_by_side(out_dir: Path, report: dict, cfg: dict, path: Path) -> None:
    """For every B frame, the A frame with the closest camera pose, both with projected boxes."""
    import cv2
    import imageio.v3 as iio

    poses = json.load(open(out_dir / "align" / "poses.json"))
    img_dir = out_dir / "sfm" / "undistorted" / "images"
    A = sorted(n for n, p in poses.items() if p["epoch"] == "A")
    B = sorted(n for n, p in poses.items() if p["epoch"] == "B")
    if not A or not B:
        return
    cA = np.array([camera_center(np.array(poses[n]["T_cw"])) for n in A])
    zA = np.array([np.array(poses[n]["T_cw"])[2, :3] for n in A])
    boxes = list(change_boxes(report))
    frames = []
    for nb in B:
        T_b = np.array(poses[nb]["T_cw"])
        # pose distance: centre distance + viewing-direction difference (scene-relative weight)
        d = np.linalg.norm(cA - camera_center(T_b), axis=1) + cfg["pair_angle_weight"] * (1 - zA @ T_b[2, :3])
        na = A[int(np.argmin(d))]
        tiles = []
        for name, side in ((na, "A"), (nb, "B")):
            img = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
            K, T = np.array(poses[name]["K"]), np.array(poses[name]["T_cw"])
            for box, action, text, which in boxes:
                if shown_in(action, which, side):
                    draw_box(img, box, K, T, action_color(cfg, action), text)
            cv2.putText(
                img,
                f"{'BEFORE' if side == 'A' else 'AFTER'}  {name}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(img)
        h = min(t.shape[0] for t in tiles)
        tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
        frame = cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB)
        s = cfg["mp4_max_width"] / frame.shape[1]
        if s < 1:
            frame = cv2.resize(frame, (int(frame.shape[1] * s) // 2 * 2, int(frame.shape[0] * s) // 2 * 2))
        else:
            frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
        frames.append(frame)
    iio.imwrite(str(path), np.stack(frames), fps=cfg["mp4_fps"], plugin="pyav", codec="libx264")
