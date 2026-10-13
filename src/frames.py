"""Frame extraction from the two videos.

Frames are sampled at a target rate, rotated upright using the video's display matrix,
resized, and filtered for motion blur. They are named A_0001.jpg / B_0001.jpg and listed in
frames/manifest.csv and manifest.json (frame_name, source_video, epoch, timestamp_sec,
blur_score, width, height).
"""

import csv
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

# display-matrix rotation (degrees, mod 360) -> cv2 rotation that makes the frame upright.
# Android phones report -90 (= 270) for portrait video, which needs a clockwise turn.
_UPRIGHT_ROTATE = {
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian; higher is sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_longest(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale so the longest side is at most max_dim (never upscales)."""
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def display_rotation_deg(video_path: Path) -> int:
    """Rotation stored in the video's display matrix, in degrees mod 360.

    Phones store frames in sensor orientation and only flag the upright orientation, which
    imageio's reader ignores; PyAV exposes it as VideoFrame.rotation.
    """
    import av

    with av.open(str(video_path)) as container:
        return int(round(next(container.decode(video=0)).rotation)) % 360


def blur_keep_mask(scores: np.ndarray, rel_threshold: float, window: int) -> np.ndarray:
    """Keep frame i if scores[i] >= rel_threshold * median of the scores around it.

    Laplacian variance depends on scene texture as much as on blur: a plain tiled floor
    scores ~50 where a patterned bed sheet scores ~400 at the same sharpness, so a fixed
    threshold throws away sharp low-texture frames. Texture changes slowly along a video,
    while motion blur shows up as a short dip against the neighbouring frames.
    """
    half = window // 2
    local = np.array([np.median(scores[max(0, i - half) : i + half + 1]) for i in range(len(scores))])
    return scores >= rel_threshold * local


def extract_from_video(video_path: Path, epoch: str, frames_dir: Path, cfg: dict) -> list[dict]:
    """Sample, orient, resize and blur-filter one video; returns manifest rows."""
    fps = iio.immeta(video_path, plugin="pyav").get("fps", 30.0)
    rotation = display_rotation_deg(video_path)
    if rotation not in (0, *_UPRIGHT_ROTATE):
        raise NotImplementedError(f"display rotation {rotation} deg is not a multiple of 90")
    step = max(1, int(round(fps / cfg["fps"])))
    print(f"  [{epoch}] {video_path.name}: {fps:.1f} fps, display rotation {rotation} deg")

    candidates = []  # (bgr, timestamp); blur is judged after all frames are scored
    for idx, rgb in enumerate(iio.imiter(video_path, plugin="pyav")):
        if idx % step:
            continue
        bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
        if rotation:
            bgr = cv2.rotate(bgr, _UPRIGHT_ROTATE[rotation])
        candidates.append((resize_longest(bgr, cfg["max_dimension"]), idx / fps))

    scores = np.array([sharpness(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)) for bgr, _ in candidates])
    keep = blur_keep_mask(scores, cfg["blur_rel_threshold"], cfg["blur_window"])

    rows = []
    for (bgr, t), score, ok in zip(candidates, scores, keep):
        if not ok:
            continue
        name = f"{epoch}_{len(rows) + 1:04d}.jpg"
        cv2.imwrite(str(frames_dir / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(
            {
                "frame_name": name,
                "source_video": video_path.name,
                "epoch": epoch,
                "timestamp_sec": round(t, 3),
                "blur_score": round(float(score), 1),
                "width": bgr.shape[1],
                "height": bgr.shape[0],
            }
        )
    print(
        f"  [{epoch}] kept {len(rows)}/{len(candidates)} frames "
        f"(sharpness min/median/max {scores.min():.0f}/{np.median(scores):.0f}/{scores.max():.0f})"
    )
    return rows


def extract_frames(video_a: Path, video_b: Path, out_dir: Path, cfg: dict) -> Path:
    """Extract frames of both videos into out_dir/frames and write the manifests."""
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in [*frames_dir.glob("A_*.jpg"), *frames_dir.glob("B_*.jpg")]:
        stale.unlink()  # a re-run may keep fewer frames; never mix runs

    rows = extract_from_video(video_a, "A", frames_dir, cfg) + extract_from_video(
        video_b, "B", frames_dir, cfg
    )
    with open(frames_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(frames_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"  {len(rows)} frames -> {frames_dir}")
    return frames_dir
