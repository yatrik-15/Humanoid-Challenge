"""Package a pipeline run as an example folder: examples/<name>/.

Copies the report and visual outputs, draws a before / after still of the first change,
writes the folder's README.md (GitHub shows it when the folder is opened) and, with
--videos, re-encodes the two input videos small enough to ship (same resolution, 10 fps,
H.264): the pipeline samples 2 fps, so the run can be repeated on exactly these files.

    python scripts/package_example.py out/hero hero --videos A.mp4 B.mp4 --rrd
    python scripts/package_example.py out/static static --no-anim
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.viz import action_color, change_boxes, draw_box  # noqa: E402
from src.viz.report import shown_in  # noqa: E402

VIEWER = (
    "https://app.rerun.io/version/0.39.0/?url="
    "https://raw.githubusercontent.com/yatrik-15/Humanoid-Challenge/main/examples/{}/changes.rrd"
)


def encode_inputs(videos: list[Path], dst: Path) -> None:
    """Re-encode the before / after videos to input/A.mp4 and input/B.mp4 (rotation applied).

    Phone videos carry metadata such as the GPS position where they were filmed; it is dropped.
    """
    import imageio_ffmpeg

    (dst / "input").mkdir(parents=True, exist_ok=True)
    for src, epoch in zip(videos, ("A", "B")):
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf", "fps=10",
             "-c:v", "libx264", "-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p", "-an",
             "-map_metadata", "-1", "-movflags", "+faststart", str(dst / "input" / f"{epoch}.mp4")],
            check=True,
        )  # fmt: skip


def copy_outputs(run: Path, dst: Path, with_rrd: bool, with_gif: bool) -> None:
    files = ["report.md", "report.json", "floor_map.png", "scene_3d.png"]
    files += ["scene_3d.gif"] if with_gif else []
    files += ["changes.rrd"] if with_rrd else []
    for f in files:
        if (run / f).exists():
            shutil.copy2(run / f, dst / f)
    if (run / "nav" / "route.json").exists():
        (dst / "nav").mkdir(exist_ok=True)
        for f in ["route.json", "map_A.pgm", "map_A.yaml", "map_B.pgm", "map_B.yaml"]:
            shutil.copy2(run / "nav" / f, dst / "nav" / f)


def write_still(run: Path, dst: Path, cfg: dict) -> None:
    """side_by_side.jpg: the before / after views the semantic stage picked for the first change
    (its best unoccluded view, and the same place in the other recording), every change box
    drawn where it applies."""
    import cv2

    report = json.load(open(run / "report.json"))
    if not report["changes"] or len(report["changes"][0]["views"]) != 2:
        return
    poses = json.load(open(run / "align" / "poses.json"))
    tiles = []
    for v in sorted(report["changes"][0]["views"], key=lambda v: v["epoch"]):
        side, frame = v["epoch"], v["frame"]
        img = cv2.imread(str(run / "sfm" / "undistorted" / "images" / frame))
        K, T = np.array(poses[frame]["K"]), np.array(poses[frame]["T_cw"])
        for box, action, text, which in change_boxes(report):
            if shown_in(action, which, side):
                draw_box(img, box, K, T, action_color(cfg, action), text)
        title = f"{'BEFORE' if side == 'A' else 'AFTER'}  {frame}"
        cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    out = np.hstack(tiles)
    s = 1600 / out.shape[1]
    out = cv2.resize(out, (1600, int(out.shape[0] * s))) if s < 1 else out
    cv2.imwrite(str(dst / "side_by_side.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])


def write_readme(dst: Path, name: str) -> None:
    """The report without its own visuals section, plus the visuals that ship in this folder."""
    md = (dst / "report.md").read_text(encoding="utf-8").split("\n## Visuals")[0].rstrip() + "\n"
    (dst / "report.md").write_text(md, encoding="utf-8", newline="\n")
    L = ["", "## Visuals", ""]
    if (dst / "scene_3d.gif").exists() or (dst / "scene_3d.png").exists():
        g = "scene_3d.gif" if (dst / "scene_3d.gif").exists() else "scene_3d.png"
        L += ["Both recordings fused in 3-D, seen from one virtual camera (left before, right after):", ""]
        L += [f"![3-D overview]({g})", ""]
    if (dst / "side_by_side.jpg").exists():
        L += ["Best before / after view of the first change, with every change box projected in:", ""]
        L += ["![before and after](side_by_side.jpg)", ""]
    if (dst / "floor_map.png").exists():
        L += ["Top-down map (filled = there now, dashed = gone or old position):", ""]
        L += ['<img src="floor_map.png" width="520" alt="top-down change map">', ""]
    if (dst / "changes.rrd").exists():
        L += [
            f"Interactive 3-D in the browser: [open in the Rerun web viewer]({VIEWER.format(name)}), "
            "or locally `rerun changes.rrd`.",
            "",
        ]
    if (dst / "nav").exists():
        L += ["Route check and ROS `map_server` occupancy maps (before / after): [`nav/`](nav).", ""]
    if (dst / "input").exists():
        L += [
            "Input videos: [`input/A.mp4`](input/A.mp4) (before), [`input/B.mp4`](input/B.mp4) (after).",
            "",
        ]
    else:
        L += ["The input videos of this pair are not included (size); the outputs above come from them.", ""]
    (dst / "README.md").write_text(md + "\n".join(L), encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Package a pipeline run as examples/<name>/")
    ap.add_argument("run", type=Path, help="pipeline output dir, e.g. out/hero")
    ap.add_argument("name", help="example name: examples/<name>/")
    ap.add_argument("--videos", nargs=2, type=Path, metavar=("A", "B"), help="re-encode these as input/")
    ap.add_argument("--rrd", action="store_true", help="include the Rerun recording")
    ap.add_argument("--no-anim", action="store_true", help="no gif or still (pairs without a change)")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))["viz"]
    dst = REPO / "examples" / args.name
    dst.mkdir(parents=True, exist_ok=True)
    if args.videos:
        encode_inputs(args.videos, dst)
    copy_outputs(args.run, dst, args.rrd, not args.no_anim)
    if not args.no_anim:
        write_still(args.run, dst, cfg)
    write_readme(dst, args.name)
    print(f"examples/{args.name}: {sorted(p.name for p in dst.iterdir())}")


if __name__ == "__main__":
    main()
