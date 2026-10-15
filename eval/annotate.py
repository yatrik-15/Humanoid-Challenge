"""Hand-label ground truth by clicking each changed object in two or more frames.

The clicked centres are triangulated from the SfM camera poses (align/poses.json) alone;
the change detector's output is never read, so the labels stay independent of it.

When the clicked views are too close together (viewing rays less than --min_angle_deg
apart, e.g. a recording made from nearly one spot), triangulation cannot fix the distance
along the rays. The point is then taken where each click ray first meets the recording's
own fused surface (fusion/grid_<epoch>.npz, the epoch the object is clicked in) and the
median of those hits is saved. The entry records which method was used.

Interactive (opens a window; keys: n/p = next/previous frame, click = mark centre,
u = undo last click on this frame, enter = triangulate and save, q = quit):
    python eval/annotate.py --out out/change --scene change --action removed --label keyboard --epoch A

Non-interactive (pixel coordinates read off any image viewer, on sfm/undistorted/images/):
    python eval/annotate.py --out out/change --scene change --action removed --label keyboard \
        --epoch A --clicks A_0030.jpg:410,1180 A_0041.jpg:520,1150

Moved objects: annotate twice with the same --label, once with --part before --epoch A and
once with --part after --epoch B; the two halves are merged into one "moved" entry.

Pixel (u, v) = (column, row) in the undistorted image as shown by an image viewer
(integer pixel index); converted to COLMAP coordinates by +0.5 (utils/geometry.py).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.geometry import project_points  # noqa: E402


def triangulate(Ks: list[np.ndarray], T_cws: list[np.ndarray], uvs: list[np.ndarray]) -> np.ndarray:
    """Linear (DLT) triangulation of one world point from >= 2 views.

    Args:
        Ks: 3x3 intrinsics per view; T_cws: 4x4 world->camera per view;
        uvs: (2,) COLMAP image coordinates per view.
    Returns:
        (3,) world point.
    """
    rows = []
    for K, T, uv in zip(Ks, T_cws, uvs):
        P = K @ T[:3, :]
        rows.append(uv[0] * P[2] - P[0])
        rows.append(uv[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def reprojection_errors(X: np.ndarray, Ks, T_cws, uvs) -> list[float]:
    """Pixel distance between each click and the projection of the triangulated point."""
    errs = []
    for K, T, uv in zip(Ks, T_cws, uvs):
        uv_hat, z = project_points(X[None], K, T)
        errs.append(float(np.linalg.norm(uv_hat[0] - uv)) if z[0] > 0 else float("inf"))
    return errs


def ray_angle_deg(X: np.ndarray, T_cws: list[np.ndarray]) -> float:
    """Largest angle between the rays from the cameras to X (triangulation baseline)."""
    from src.utils.geometry import camera_center

    d = [(X - camera_center(T)) / np.linalg.norm(X - camera_center(T)) for T in T_cws]
    return max(float(np.degrees(np.arccos(np.clip(a @ b, -1, 1)))) for a in d for b in d)


def surface_hits(out: Path, epoch: str, Ks, T_cws, uvs) -> np.ndarray:
    """Where each click ray first meets the fused surface of `epoch` (rows of NaN = no hit)."""
    import open3d as o3d
    import open3d.core as o3c
    import yaml

    from src.fusion import DepthRenderer
    from src.utils.geometry import invert_pose

    cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))["fusion"]
    grid = o3d.t.geometry.VoxelBlockGrid.load(str(out / "fusion" / f"grid_{epoch}.npz"))
    scene = DepthRenderer(grid, cfg).scene
    hits = []
    for K, T, uv in zip(Ks, T_cws, uvs):
        T_wc = invert_pose(T)
        d = T_wc[:3, :3] @ (np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]))
        o = T_wc[:3, 3]
        t = scene.cast_rays(o3c.Tensor(np.r_[o, d][None].astype(np.float32)))["t_hit"].numpy()[0]
        hits.append(o + t * d if np.isfinite(t) else np.full(3, np.nan))
    return np.array(hits)


def save_entry(
    gt_path: Path,
    scene: str,
    action: str,
    label: str,
    part: str,
    X: np.ndarray,
    clicks: dict[str, tuple[float, float]],
    errs: list[float],
    method: str = "triangulated",
) -> dict:
    """Add/merge one change into eval/gt/<scene>.json (format of eval/run_eval.py)."""
    gt = (
        json.loads(gt_path.read_text())
        if gt_path.exists()
        else {"scene": scene, "frame": "pipeline", "changes": []}
    )
    src = {
        "clicks": {k: list(v) for k, v in clicks.items()},
        "reproj_err_px": [round(e, 2) for e in errs],
        "method": method,
    }
    xyz = np.round(X, 4).tolist()
    if action == "moved":
        entry = next((c for c in gt["changes"] if c["action"] == "moved" and c["label"] == label), None)
        if entry is None:
            entry = {"action": "moved", "label": label}
            gt["changes"].append(entry)
        entry[f"centroid_{part}"] = xyz
        entry[f"annotation_{part}"] = src
    else:
        gt["changes"] = [c for c in gt["changes"] if not (c["action"] == action and c["label"] == label)]
        entry = {"action": action, "label": label, "centroid": xyz, "annotation": src}
        gt["changes"].append(entry)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(gt, indent=2))
    return entry


def interactive_clicks(img_dir: Path, names: list[str]) -> dict[str, tuple[float, float]]:
    """Matplotlib window to click the object centre in several frames. Returns name -> (u, v) pixel."""
    import matplotlib.pyplot as plt

    clicks: dict[str, tuple[float, float]] = {}
    state = {"i": 0, "done": False}
    fig, ax = plt.subplots(figsize=(6, 10))

    def show():
        ax.clear()
        name = names[state["i"]]
        ax.imshow(plt.imread(img_dir / name))
        if name in clicks:
            ax.plot(*clicks[name], "r+", ms=25, mew=2)
        ax.set_title(
            f"{name} ({state['i'] + 1}/{len(names)})  clicked in {len(clicks)} frames\n"
            "n/p: next/prev  click: mark centre  u: undo  enter: save  q: quit",
            fontsize=9,
        )
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes is ax and ev.button == 1 and fig.canvas.toolbar.mode == "":
            clicks[names[state["i"]]] = (float(ev.xdata), float(ev.ydata))
            show()

    def on_key(ev):
        if ev.key == "n":
            state["i"] = min(state["i"] + 1, len(names) - 1)
        elif ev.key == "p":
            state["i"] = max(state["i"] - 1, 0)
        elif ev.key == "u":
            clicks.pop(names[state["i"]], None)
        elif ev.key == "enter":
            state["done"] = True
            plt.close(fig)
            return
        elif ev.key == "q":
            plt.close(fig)
            return
        show()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()
    if not state["done"]:
        sys.exit("quit without saving")
    return clicks


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand-label one change by clicking it in 2+ frames")
    ap.add_argument("--out", required=True, help="pipeline run dir, e.g. out/change (needs align/poses.json)")
    ap.add_argument("--scene", required=True, help="GT file name: eval/gt/<scene>.json")
    ap.add_argument("--action", required=True, choices=["added", "removed", "moved"])
    ap.add_argument("--label", required=True, help="object name, e.g. keyboard")
    ap.add_argument(
        "--epoch",
        required=True,
        choices=["A", "B"],
        help="frames to click in: A for removed / moved-before, B for added / moved-after",
    )
    ap.add_argument("--part", choices=["before", "after"], help="for --action moved")
    ap.add_argument("--clicks", nargs="*", help="non-interactive: FRAME:u,v (pixel column,row)")
    ap.add_argument(
        "--min_angle_deg",
        type=float,
        default=5.0,
        help="below this triangulation angle use the fused surface hit instead (see module doc)",
    )
    ap.add_argument(
        "--max_reproj_px",
        type=float,
        default=15.0,
        help="refuse to save if any click reprojects worse than this (inconsistent clicks)",
    )
    args = ap.parse_args()
    if args.action == "moved" and not args.part:
        ap.error("--action moved needs --part before|after")

    out = Path(args.out)
    poses = json.loads((out / "align" / "poses.json").read_text())
    img_dir = out / "sfm" / "undistorted" / "images"
    names = sorted(n for n, p in poses.items() if p["epoch"] == args.epoch)

    if args.clicks:
        clicks = {}
        for c in args.clicks:
            name, uv = c.split(":")
            u, v = (float(x) for x in uv.split(","))
            if name not in poses:
                sys.exit(f"{name} is not a registered frame of this run")
            clicks[name] = (u, v)
    else:
        clicks = interactive_clicks(img_dir, names)
    if len(clicks) < 2:
        sys.exit("need clicks in at least 2 frames")

    Ks = [np.array(poses[n]["K"]) for n in clicks]
    Ts = [np.array(poses[n]["T_cw"]) for n in clicks]
    uvs = [np.array(clicks[n]) + 0.5 for n in clicks]  # pixel index -> COLMAP coordinates
    X = triangulate(Ks, Ts, uvs)
    errs = reprojection_errors(X, Ks, Ts, uvs)
    for n, e in zip(clicks, errs):
        print(f"  {n}: reprojection error {e:.1f} px")
    if max(errs) > args.max_reproj_px:
        sys.exit(f"clicks are inconsistent (max error {max(errs):.1f} px > {args.max_reproj_px}); not saved")
    method = "triangulated"
    angle = ray_angle_deg(X, Ts)
    if angle < args.min_angle_deg:
        hits = surface_hits(out, args.epoch, Ks, Ts, uvs)
        ok = hits[np.all(np.isfinite(hits), axis=1)]
        if len(ok) == 0:
            sys.exit(f"rays only {angle:.1f} deg apart and no click ray hits the fused surface; not saved")
        X = np.median(ok, axis=0)
        spread = float(np.max(np.linalg.norm(ok - X, axis=1)))
        method = "surface"
        print(
            f"  rays only {angle:.1f} deg apart: using the fused surface ({len(ok)}/{len(hits)} hits, "
            f"max {spread:.2f} m from their median)"
        )
    else:
        print(f"  triangulated (rays up to {angle:.1f} deg apart)")
    entry = save_entry(
        REPO / "eval" / "gt" / f"{args.scene}.json",
        args.scene,
        args.action,
        args.label,
        args.part,
        X,
        clicks,
        errs,
        method,
    )
    print(f"  saved: {json.dumps({k: v for k, v in entry.items() if not k.startswith('annotation')})}")


if __name__ == "__main__":
    main()
