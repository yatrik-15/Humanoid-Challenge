"""Reports and visualizations.

Writes into the run directory:
    report.json       changes with id, action, label, description, boxes, centroid,
                      displacement, label confidence and supporting view groups
    report.md         human-readable summary
    changes.rrd       Rerun recording: clouds A and B, camera frustums, change boxes,
                      arrows for moved objects, unknown evidence in grey
    side_by_side.mp4  each B frame next to the nearest-pose A frame, with projected boxes
    floor_map.png     top-down map of obstacles, changes and camera paths
    scene_3d.png/gif  fused A and B rendered from one elevated virtual camera (gif: orbit)

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
            "label_check": lab.get("label_check"),
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
    checked = [c for c in report["changes"] if c.get("label_check")]
    if checked:
        L += [
            "",
            "## Label check (SigLIP)",
            "",
            "Score gain of each name from the view without the object to the view with it. Flags only; "
            "labels and detections above are unchanged.",
            "",
            "| id | VLM label | verdict | VLM label gain | best names (gain) |",
            "|---|---|---|---|---|",
        ]
        for c in checked:
            chk = c["label_check"]
            top = ", ".join(f"{n} ({g:+.3f})" for n, g in chk.get("top", []))
            gain = f"{chk['vlm_label_score']:+.3f}" if "vlm_label_score" in chk else "–"
            L.append(f"| {c['id']} | {c['label']} | {chk['verdict']} | {gain} | {top} |")
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
        name = "object" if c["label"] == "label_unavailable" else c["label"]
        if c["action"] == "moved":
            yield c["box_before"], "moved", f"#{c['id']} {name} (before)", "before"
            yield c["box_after"], "moved", f"#{c['id']} {name} (after)", "after"
        else:
            yield c["box"], c["action"], f"#{c['id']} {name} ({c['action']})", None


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


def render_points(
    pts: np.ndarray,
    cols: np.ndarray,
    K: np.ndarray,
    T_cw: np.ndarray,
    W: int,
    H: int,
    voxel: float,
    fill_px: int = 0,
) -> np.ndarray:
    """Z-buffered splat render of a coloured cloud (RGB uint8). Each point is drawn as a
    square about one voxel wide at its depth, so the surface reads as closed."""
    uv, z = project_points(pts, K, T_cw)
    ok = z > 0.05
    uv, z, cols = uv[ok], z[ok], cols[ok]
    rmax = 12
    r = np.clip(np.round(0.6 * K[0, 0] * voxel / z), 1, rmax).astype(int)
    u0, v0 = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    idx, depth, src = [], [], []
    for dy in range(-rmax, rmax + 1):
        for dx in range(-rmax, rmax + 1):
            m = np.maximum(abs(dx), abs(dy)) <= r
            u, v = u0[m] + dx, v0[m] + dy
            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            idx.append(v[inside] * W + u[inside])
            depth.append(z[m][inside])
            src.append(np.flatnonzero(m)[inside])
    idx, depth, src = np.concatenate(idx), np.concatenate(depth), np.concatenate(src)
    order = np.lexsort((depth, idx))  # nearest point first within each pixel
    first = order[np.r_[True, idx[order][1:] != idx[order][:-1]]]
    img = np.full((H * W, 3), 32, np.uint8)
    img[idx[first]] = (np.clip(cols[src[first]], 0, 1) * 255).astype(np.uint8)
    hit = np.zeros(H * W, np.uint8)
    hit[idx[first]] = 255
    return fill_small_holes(img.reshape(H, W, 3), hit.reshape(H, W), fill_px)


def fill_small_holes(img: np.ndarray, hit: np.ndarray, size: int) -> np.ndarray:
    """Inpaint background pixels enclosed by surface within `size` px. A plane oblique to
    the voxel grid leaves regular rows of missing surface points; this closes them without
    growing the silhouette."""
    import cv2

    if size <= 0:
        return img
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    holes = cv2.morphologyEx(hit, cv2.MORPH_CLOSE, k) & ~hit
    if not holes.any():
        return img
    return cv2.inpaint(img, holes, 3, cv2.INPAINT_TELEA)


def overview_camera(out_dir: Path, report: dict, angle_deg: float, W: int, H: int, pts: np.ndarray):
    """(K, T_cw) of an elevated virtual camera behind the recorded paths, looking at the
    changes (or the scene centre), rotated by `angle_deg` about the vertical axis. The focal
    length and principal point are fitted so that 98 % of `pts` fill the image."""
    from src.utils.geometry import look_at_T_cw

    poses = json.load(open(out_dir / "align" / "poses.json"))
    C = np.array([camera_center(np.array(p["T_cw"])) for p in poses.values()])
    fwd = np.array([np.array(p["T_cw"])[2, :3] for p in poses.values()]).mean(axis=0)
    boxes = [b for b, *_ in _boxes(report)]
    if boxes:
        target = np.mean([b["center"] for b in boxes], axis=0)
    else:  # scene centre: about 1.5 camera-path lengths in front of the cameras
        target = C.mean(axis=0) + fwd / np.linalg.norm(fwd) * 1.5
    back = C.mean(axis=0) - target
    back[2] = 0
    dist = max(np.linalg.norm(back), 1.0)
    back /= dist
    a = np.deg2rad(angle_deg)
    back = np.array([back[0] * np.cos(a) - back[1] * np.sin(a), back[0] * np.sin(a) + back[1] * np.cos(a), 0])
    eye = target + back * dist * 1.2 + np.array([0, 0, 1.0]) * dist * 0.8
    T = look_at_T_cw(eye, target, np.array([0.0, 0.0, 1.0]))
    Xc = pts @ T[:3, :3].T + T[:3, 3]
    Xc = Xc[Xc[:, 2] > 0.05]
    n = Xc[:, :2] / Xc[:, 2:]
    lo, hi = np.percentile(n, 1, axis=0), np.percentile(n, 99, axis=0)
    f = 0.92 * min(W / (hi[0] - lo[0]), H / (hi[1] - lo[1]))
    K = np.array(
        [[f, 0, W / 2 - f * (lo[0] + hi[0]) / 2], [0, f, H / 2 - f * (lo[1] + hi[1]) / 2], [0, 0, 1]]
    )
    return K, T


def render_overview(out_dir: Path, report: dict, cfg: dict, angle_deg: float, W: int, H: int, voxel: float):
    """BEFORE | AFTER renders from one virtual camera, with change boxes and camera paths.

    Points are splatted at twice the resolution and averaged down (anti-aliased edges
    instead of square blocks); boxes and labels are drawn afterwards at full sharpness."""
    import cv2

    ss = 2

    clouds = {e: np.load(out_dir / "fusion" / f"cloud_{e}.npz") for e in ("A", "B")}
    K, T = overview_camera(out_dir, report, angle_deg, W, H, clouds["B"]["points"])
    poses = json.load(open(out_dir / "align" / "poses.json"))
    tiles = []
    for e in ("A", "B"):
        c = clouds[e]
        Ks = K.copy()
        Ks[:2] *= ss  # exact for COLMAP corner-origin pixel coordinates
        big = render_points(c["points"], c["colors"], Ks, T, W * ss, H * ss, voxel, fill_px=9)
        img = cv2.cvtColor(cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        C = np.array([camera_center(np.array(p["T_cw"])) for p in poses.values() if p["epoch"] == e])
        uv, z = project_points(C, K, T)
        col = (255, 140, 80) if e == "A" else (60, 170, 255)
        for (u, v), zz in zip(uv, z):
            if zz > 0:
                cv2.circle(img, (int(u), int(v)), 3, col, -1, cv2.LINE_AA)
        for box, action, text, which in _boxes(report):
            show = (
                (action == "removed" and e == "A")
                or (action == "added" and e == "B")
                or (action == "moved" and ((which == "before") == (e == "A")))
            )
            if show:
                draw_box(img, box, K, T, _color(cfg, action), text)
        title = "BEFORE (A)" if e == "A" else "AFTER (B)"
        cv2.putText(img, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    return cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB)


def write_overview(out_dir: Path, report: dict, cfg: dict, voxel: float) -> None:
    """scene_3d.png (one view) and scene_3d.gif (orbit) of the fused before/after scenes."""
    import imageio.v3 as iio

    W, H = cfg["overview_size"]
    iio.imwrite(str(out_dir / "scene_3d.png"), render_overview(out_dir, report, cfg, 0.0, W, H, voxel))
    span = cfg["overview_orbit_deg"]
    angles = np.concatenate([np.linspace(-span, span, 13), np.linspace(span, -span, 13)[1:-1]])
    w, h = W // 2, H // 2
    frames = [render_overview(out_dir, report, cfg, a, w, h, voxel) for a in angles]
    write_gif(frames, out_dir / "scene_3d.gif", duration_ms=120)


def write_gif(frames: list[np.ndarray], path: Path, duration_ms: int) -> None:
    """Animated GIF with ONE palette for all frames and no dithering: per-frame palettes
    make colours flicker, and dithering adds speckle noise to smooth surfaces."""
    from PIL import Image

    sample = np.vstack([f[::4, ::4] for f in frames])
    palette = Image.fromarray(sample).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)


def generate_visualizations(out_dir: Path, cfg: dict, voxel: float = 0.02) -> dict:
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
    if cfg["overview_enabled"]:
        write_overview(out_dir, report, cfg, voxel)
        print(f"  3-D overview: {out_dir / 'scene_3d.png'}, {out_dir / 'scene_3d.gif'}")
    append_visuals(out_dir / "report.md")
    return report


def append_visuals(md_path: Path) -> None:
    """Add a section showing the visual outputs that sit next to report.md."""
    d = md_path.parent
    L = ["", "## Visuals", ""]
    if (d / "scene_3d.gif").exists() or (d / "scene_3d.png").exists():
        name = "scene_3d.gif" if (d / "scene_3d.gif").exists() else "scene_3d.png"
        L += ["Both recordings fused in 3-D, seen from one virtual camera (left before, right after):", ""]
        L += [f"![3-D overview]({name})", ""]
    if (d / "floor_map.png").exists():
        L += ["Top-down map (filled = there now, dashed = gone or old position):", ""]
        L += ['<img src="floor_map.png" width="520" alt="top-down change map">', ""]
    if (d / "side_by_side.mp4").exists():
        L += ["Every after-frame next to the closest before-frame: [side_by_side.mp4](side_by_side.mp4)", ""]
    if (d / "changes.rrd").exists():
        L += ["Interactive 3-D: `rerun changes.rrd`", ""]
    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n".join(L))
