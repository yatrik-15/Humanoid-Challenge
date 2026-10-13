"""
Render a synthetic textured room as two 'phone videos' (A = before, B = after) with known
changes and known camera poses, for an end-to-end pipeline test before real data exists.

Usage:  python -m tests.make_synthetic_videos [--out data/synth]
Writes: <out>_A.mp4, <out>_B.mp4, <out>_gt.json (true poses T_cw in the synthetic world
        frame, K, changes with true boxes). This GT is exact BY CONSTRUCTION; it is not a
        substitute for hand-labelled ground truth on real recordings.
"""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from tests.synthetic import Box, Scene, orbit_poses, render

W, H, F = 640, 480, 520.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])

STATIC = [
    Box((-0.6, 1.2, 0.0), (0.6, 1.8, 0.75), (0.75, 0.55, 0.35)),  # table
    Box((1.5, -0.5, 0.0), (2.2, 0.9, 0.45), (0.35, 0.45, 0.70)),  # sofa
    Box((-2.2, -2.2, 0.0), (-1.6, -1.6, 1.6), (0.55, 0.55, 0.50)),  # cabinet
]
REMOVED = Box((-1.4, 0.2, 0.0), (-0.9, 0.7, 0.9), (0.85, 0.20, 0.20))  # "chair", A only
ADDED = Box((0.4, -1.4, 0.0), (0.8, -1.0, 0.35), (0.20, 0.35, 0.85))  # "box", B only
MOVED_A = Box((0.6, 0.1, 0.0), (0.95, 0.45, 0.5), (0.20, 0.75, 0.25))  # "bin", before
MOVED_B = Box((-0.5, -0.9, 0.0), (-0.15, -0.55, 0.5), (0.20, 0.75, 0.25))  # after

SCENE_A = Scene(
    STATIC + [REMOVED, MOVED_A],
    floor_extent=2.5,
    closed_room=True,
    textured=True,
    floor_color=(0.65, 0.6, 0.55),
)
SCENE_B = Scene(
    STATIC + [ADDED, MOVED_B],
    floor_extent=2.5,
    closed_room=True,
    textured=True,
    floor_color=(0.65, 0.6, 0.55),
)


def path(radius, height, start_deg, sweep_deg, n, wobble, target):
    """Hand-held-like path: an arc with small radial/height wobble, looking at `target`."""
    poses = []
    angs = np.linspace(start_deg, start_deg + sweep_deg, n)
    for k, a in enumerate(angs):
        r = radius + wobble * np.sin(k * 0.7)
        h = height + 0.5 * wobble * np.cos(k * 0.9)
        tgt = np.array(target) + np.array([0.15 * np.sin(k * 0.3), 0.15 * np.cos(k * 0.4), 0.0])
        poses += orbit_poses((0, 0, 0), r, h, [a], target=tgt)
    return poses


def box_json(b: Box, label: str) -> dict:
    lo, hi = np.array(b.lo), np.array(b.hi)
    return {"label": label, "center": ((lo + hi) / 2).tolist(), "extents": (hi - lo).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    poses = {
        "A": path(1.7, 1.30, 0, 300, args.frames, 0.08, (0, 0, 0.4)),
        "B": path(1.5, 1.25, 40, 290, args.frames, 0.10, (0, 0, 0.3)),
    }
    gt = {
        "scene": out.name,
        "frame": "gt_world",
        "video_fps": 4,
        "match_radius": 0.3,
        "K": K.tolist(),
        "width": W,
        "height": H,
        "poses": {},
        "changes": [
            {"action": "removed", **box_json(REMOVED, "chair")},
            {
                "action": "added",
                "label": "box",
                **{k: v for k, v in box_json(ADDED, "box").items() if k != "label"},
            },
            {
                "action": "moved",
                "label": "bin",
                "before": box_json(MOVED_A, "bin"),
                "after": box_json(MOVED_B, "bin"),
            },
        ],
    }
    for epoch, scene in (("A", SCENE_A), ("B", SCENE_B)):
        frames = []
        for i, T in enumerate(poses[epoch]):
            _, rgb = render(scene, K, T, H, W)
            frames.append((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
            gt["poses"][f"{epoch}_{i:03d}"] = T.tolist()
        iio.imwrite(str(out) + f"_{epoch}.mp4", np.stack(frames), fps=4, plugin="pyav", codec="libx264")
        print(f"wrote {out}_{epoch}.mp4 ({len(frames)} frames)")
    with open(str(out) + "_gt.json", "w") as f:
        json.dump(gt, f, indent=1)
    print(f"wrote {out}_gt.json")


if __name__ == "__main__":
    main()
