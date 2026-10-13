"""Reports and visualizations.

Writes into the run directory:
    report.json       changes with id, action, label, description, boxes, centroid,
                      displacement, label confidence and supporting view groups
    report.md         human-readable summary
    changes.rrd       Rerun recording: clouds A and B, camera frustums, change boxes,
                      arrows for moved objects, unknown evidence in grey
    side_by_side.mp4  each B frame next to the nearest-pose A frame, with projected boxes
    floor_map.png     top-down map of obstacles, changes and camera paths

Rerun transforms are applied as X_parent = mat3x3 @ X + translation.
"""

import json
from pathlib import Path

import numpy as np

from src.utils.geometry import camera_center, invert_pose, project_points

ACTION_ORDER = ("moved", "removed", "added")


def _color(cfg: dict, action: str) -> list[int]:
    return list(cfg[f"bbox_color_{action}"])


def build_report(out_dir: Path) -> dict:
    """Merge geometry (cluster/changes.json) and semantics (semantics/labels.json)."""
    changes = json.load(open(out_dir / "cluster" / "changes.json"))
    labels_path = out_dir / "semantics" / "labels.json"
    labels = {l["id"]: l for l in json.load(open(labels_path))} if labels_path.exists() else {}
    # Labels from an older clustering run would attach to the wrong change; ignore those.
    by_id = {ch["id"]: ch["action"] for ch in changes}
    labels = {i: l for i, l in labels.items() if by_id.get(i) == l.get("action")}
    align = json.load(open(out_dir / "align" / "alignment.json"))
    diff = json.load(open(out_dir / "diff" / "diff_report.json"))
    items = []
    for ch in changes:
        lab = labels.get(ch["id"], {})
        item = {
            "id": ch["id"],
            "action": ch["action"],
            "label": lab.get("object", "label_unavailable"),
            "label_confidence": lab.get("confidence", 0.0),
            "description": lab.get("description", ""),
            "spatial_summary": lab.get("spatial_summary", ""),
            "supporting_view_groups": ch["support_views"],
            "num_evidence_voxels": ch["num_voxels"],
        }
        if ch["action"] == "moved":
            item.update(
                {
                    "box_before": ch["box_before"],
                    "box_after": ch["box_after"],
                    "centroid_before": ch["box_before"]["center"],
                    "centroid_after": ch["box_after"]["center"],
                    "displacement": ch["displacement"],
                    "distance": ch["distance"],
                }
            )
        else:
            item.update({"box": ch["box"], "centroid": ch["box"]["center"]})
        item["views"] = lab.get("views", [])
        items.append(item)
    items.sort(key=lambda c: (ACTION_ORDER.index(c["action"]), c["id"]))
    return {
        "units": "approx. metres" if align["scale_is_metric"] else "scene units (not metric)",
        "scale_note": align["scale_note"],
        "world_frame": (
            "+z up; z=0 is the dominant horizontal support plane found by RANSAC (the floor, or a "
            "bed/table top when the floor is not visible)"
        )
        if align["floor_found"]
        else "gravity from camera-up (no floor plane found)",
        "counts": {a: sum(c["action"] == a for c in items) for a in ACTION_ORDER},
        "unknown_evidence_voxels": diff["unknown_voxels"],
        "changes": items,
        "navigation": _route_summary(out_dir),
    }


def _route_summary(out_dir: Path) -> dict | None:
    """nav/route.json without the polylines, or None when the nav stage was skipped."""
    path = out_dir / "nav" / "route.json"
    if not path.exists():
        return None
    route = json.load(open(path))
    return {k: v for k, v in route.items() if k not in ("route", "blocked_points", "replanned")}


def write_markdown(report: dict, path: Path) -> None:
    L = [
        "# What changed in the room",
        "",
        f"Units: **{report['units']}** ({report['scale_note']}). World frame: {report['world_frame']}.",
        "",
        f"Detected: **{report['counts']['moved']} moved, {report['counts']['removed']} removed, "
        f"{report['counts']['added']} added**. Regions observed in only one recording are reported "
        f"as UNKNOWN ({report['unknown_evidence_voxels']} evidence voxels), never as changes.",
        "",
        "| id | action | object | label conf. | where (centroid x, y, z) | size / displacement | view groups |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in report["changes"]:
        if c["action"] == "moved":
            where = f"{_fmt(c['centroid_before'])} → {_fmt(c['centroid_after'])}"
            size = f"moved {c['distance']:.2f}"
        else:
            where = _fmt(c["centroid"])
            size = " x ".join(f"{e:.2f}" for e in c["box"]["extents"])
        L.append(
            f"| {c['id']} | {c['action']} | {c['label']} | {c['label_confidence']:.2f} | {where} | {size} | "
            f"{c['supporting_view_groups']} |"
        )
    L += ["", "## Descriptions", ""]
    for c in report["changes"]:
        L.append(
            f"- **#{c['id']} {c['action']} — {c['label']}**: {c['description']}"
            + (f" {c['spatial_summary']}" if c["spatial_summary"] else "")
        )
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _fmt(v) -> str:
    return "(" + ", ".join(f"{x:.2f}" for x in v) + ")"


def _boxes(report: dict):
    """Yield (box dict, action, label text, which) for every box in the report."""
    for c in report["changes"]:
        if c["action"] == "moved":
            yield c["box_before"], "moved", f"#{c['id']} {c['label']} (before)", "before"
            yield c["box_after"], "moved", f"#{c['id']} {c['label']} (after)", "after"
        else:
            yield c["box"], c["action"], f"#{c['id']} {c['label']} ({c['action']})", None


def write_rerun(out_dir: Path, report: dict, cfg: dict, path: Path) -> None:
    import rerun as rr

    rr.init("what_changed", spawn=False)
    rr.save(str(path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for e, _col in (("A", [80, 140, 255]), ("B", [255, 170, 60])):
        c = np.load(out_dir / "fusion" / f"cloud_{e}.npz")
        pts = c["points"]
        if len(pts) > cfg["max_cloud_points"]:
            idx = np.random.default_rng(0).choice(len(pts), cfg["max_cloud_points"], replace=False)
            pts, cols = pts[idx], c["colors"][idx]
        else:
            cols = c["colors"]
        rr.log(
            f"world/cloud_{e}",
            rr.Points3D(pts, colors=(np.clip(cols, 0, 1) * 255).astype(np.uint8), radii=cfg["point_radius"]),
            static=True,
        )
    ev = np.load(out_dir / "diff" / "evidence.npz")
    if len(ev["unknown_points"]):
        rr.log(
            "world/unknown",
            rr.Points3D(ev["unknown_points"], colors=[150, 150, 150], radii=cfg["point_radius"]),
            static=True,
        )
    for k in ("added", "removed"):
        if len(ev[f"{k}_points"]):
            rr.log(
                f"world/evidence_{k}",
                rr.Points3D(ev[f"{k}_points"], colors=_color(cfg, k), radii=cfg["point_radius"]),
                static=True,
            )
    boxes = list(_boxes(report))
    if boxes:
        rr.log(
            "world/changes",
            rr.Boxes3D(
                centers=[b["center"] for b, *_ in boxes],
                half_sizes=[np.array(b["extents"]) / 2 for b, *_ in boxes],
                rotation_axis_angles=[
                    rr.RotationAxisAngle(axis=[0, 0, 1], radians=b["yaw_rad"]) for b, *_ in boxes
                ],
                colors=[_color(cfg, a) for _, a, *_ in boxes],
                labels=[t for _, _, t, _ in boxes],
            ),
            static=True,
        )
    moved = [c for c in report["changes"] if c["action"] == "moved"]
    if moved:
        rr.log(
            "world/moved_arrows",
            rr.Arrows3D(
                origins=[c["centroid_before"] for c in moved],
                vectors=[c["displacement"] for c in moved],
                colors=[_color(cfg, "moved")] * len(moved),
            ),
            static=True,
        )
    poses = json.load(open(out_dir / "align" / "poses.json"))
    for name, p in sorted(poses.items()):
        T_wc = invert_pose(np.array(p["T_cw"]))
        ent = f"world/cameras/{p['epoch']}/{Path(name).stem}"
        rr.log(ent, rr.Transform3D(translation=T_wc[:3, 3], mat3x3=T_wc[:3, :3]), static=True)
        rr.log(
            ent,
            rr.Pinhole(
                image_from_camera=np.array(p["K"]),
                resolution=[p["width"], p["height"]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=cfg["frustum_size"],
            ),
            static=True,
        )
    rr.log(
        "report",
        rr.TextDocument(
            (out_dir / "report.md").read_text(encoding="utf-8"), media_type=rr.MediaType.MARKDOWN
        ),
        static=True,
    )


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
    cv2.putText(
        img, text, (int(top[0]), max(12, int(top[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA
    )


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
    boxes = list(_boxes(report))
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
                show = (
                    (action == "removed" and side == "A")
                    or (action == "added" and side == "B")
                    or (action == "moved" and ((which == "before") == (side == "A")))
                )
                if show:
                    draw_box(img, box, K, T, _color(cfg, action), text)
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


def write_floor_map(out_dir: Path, report: dict, cfg: dict, path: Path) -> None:
    """Top-down map: B obstacles (height band), change footprints, camera paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    B = np.load(out_dir / "fusion" / "cloud_B.npz")["points"]
    lo, hi = cfg["obstacle_z_band"]
    obst = B[(B[:, 2] > lo) & (B[:, 2] < hi)]
    poses = json.load(open(out_dir / "align" / "poses.json"))
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(obst):
        ax.scatter(obst[:, 0], obst[:, 1], s=0.3, c="0.55", label="obstacles now (B)")
    for e, col in (("A", "tab:blue"), ("B", "tab:orange")):
        C = np.array(
            [camera_center(np.array(p["T_cw"])) for n, p in sorted(poses.items()) if p["epoch"] == e]
        )
        if len(C):
            ax.plot(C[:, 0], C[:, 1], "-", color=col, lw=1, label=f"camera path {e}")
    for box, action, text, which in _boxes(report):
        corners = np.array(box["corners"])
        foot = corners[[0, 2, 6, 4], :2]  # dz=-0.5 corners in cyclic order
        col = np.array(_color(cfg, action)) / 255
        present_now = action == "added" or which == "after"  # removed / moved-before: gone now
        ax.add_patch(
            Polygon(
                foot,
                closed=True,
                fill=present_now,
                alpha=0.4 if present_now else 1,
                edgecolor=col,
                facecolor=col,
                ls="-" if present_now else "--",
                lw=2,
            )
        )
        ax.text(*np.array(box["center"][:2]), text, fontsize=8, color=col * 0.7)
    for c in report["changes"]:
        if c["action"] == "moved":
            ax.annotate(
                "",
                xy=c["centroid_after"][:2],
                xytext=c["centroid_before"][:2],
                arrowprops=dict(arrowstyle="->", color=np.array(_color(cfg, "moved")) / 255, lw=2),
            )
    route_path = out_dir / "nav" / "route.json"
    if route_path.exists():
        route = json.load(open(route_path))
        blocked = np.array(route["blocked_points"]).reshape(-1, 2)
        if len(blocked):
            ax.plot(blocked[:, 0], blocked[:, 1], "x", color="red", ms=5, label="route A, blocked now")
        if route.get("replanned"):
            p = np.array(route["replanned"])
            ax.plot(p[:, 0], p[:, 1], "--", color="tab:green", lw=2, label="new route (A*)")
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({report['units']})")
    ax.set_ylabel(f"y ({report['units']})")
    ax.set_title("Top-down change map (filled = there now, dashed = gone / old position)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def generate_visualizations(out_dir: Path, cfg: dict) -> dict:
    """Stage entry."""
    report = build_report(out_dir)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, out_dir / "report.md")
    print(f"  report.json / report.md: {report['counts']}")
    if cfg["rerun_enabled"]:
        write_rerun(out_dir, report, cfg, out_dir / "changes.rrd")
        print(f"  Rerun recording: {out_dir / 'changes.rrd'}")
    if cfg["mp4_enabled"]:
        write_side_by_side(out_dir, report, cfg, out_dir / "side_by_side.mp4")
        print(f"  Side-by-side video: {out_dir / 'side_by_side.mp4'}")
    if cfg["floor_map_enabled"]:
        write_floor_map(out_dir, report, cfg, out_dir / "floor_map.png")
        print(f"  Floor map: {out_dir / 'floor_map.png'}")
    return report
