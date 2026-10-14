"""Reports and visualizations.

Writes into the run directory:
    report.json       changes with id, action, label, description, boxes, centroid,
                      displacement, label confidence and supporting view groups
    report.md         human-readable summary
    changes.rrd       Rerun recording that opens on BEFORE / AFTER views: surface meshes,
                      camera paths, change boxes, arrows for moved objects (frustums, points
                      and evidence voxels logged but hidden by default)
    side_by_side.mp4  each B frame next to the nearest-pose A frame, with projected boxes
    floor_map.png     top-down map: seen-empty / obstacle / never observed, changes, paths
    scene_3d.png/gif  A and B surface meshes from one elevated virtual camera, changed
                      surfaces tinted (gif: orbit)

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


def simplify_mesh(mesh: dict, max_triangles: int) -> dict:
    """Quadric decimation to at most `max_triangles` (vertex colours are kept)."""
    import open3d as o3d

    if len(mesh["triangles"]) <= max_triangles:
        return mesh
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh["vertices"]),
        o3d.utility.Vector3iVector(mesh["triangles"].astype(np.int32)),
    )
    m.vertex_colors = o3d.utility.Vector3dVector(mesh["colors"])
    m = m.simplify_quadric_decimation(max_triangles)
    m.compute_vertex_normals()
    return {
        "vertices": np.asarray(m.vertices),
        "triangles": np.asarray(m.triangles, np.int64),
        "colors": np.clip(np.asarray(m.vertex_colors), 0, 1),
        "normals": np.asarray(m.vertex_normals),
    }


def log_extras(out_dir: Path, e: str, s: str, cfg: dict, poses: dict, ev, rng) -> None:
    """Surface points, every camera frustum and the evidence voxels of one recording."""
    import rerun as rr

    names = sorted(n for n, p in poses.items() if p["epoch"] == e)
    c = np.load(out_dir / "fusion" / f"cloud_{e}.npz")
    pts, cols = c["points"], c["colors"]
    if len(pts) > cfg["max_cloud_points"]:
        idx = rng.choice(len(pts), cfg["max_cloud_points"], replace=False)
        pts, cols = pts[idx], cols[idx]
    rr.log(
        f"world/{s}/points",
        rr.Points3D(pts, colors=(np.clip(cols, 0, 1) * 255).astype(np.uint8), radii=cfg["point_radius"]),
        static=True,
    )
    for n in names:
        T_wc = invert_pose(np.array(poses[n]["T_cw"]))
        ent = f"world/{s}/cameras/{Path(n).stem}"
        rr.log(ent, rr.Transform3D(translation=T_wc[:3, 3], mat3x3=T_wc[:3, :3]), static=True)
        rr.log(
            ent,
            rr.Pinhole(
                image_from_camera=np.array(poses[n]["K"]),
                resolution=[poses[n]["width"], poses[n]["height"]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=cfg["frustum_size"],
            ),
            static=True,
        )
    k = "removed" if e == "A" else "added"
    if len(ev[f"{k}_points"]):
        rr.log(
            f"world/{s}/evidence/{k}",
            rr.Points3D(ev[f"{k}_points"], colors=_color(cfg, k), radii=cfg["point_radius"]),
            static=True,
        )
    if len(ev["unknown_points"]):
        rr.log(
            f"world/{s}/evidence/unknown",
            rr.Points3D(ev["unknown_points"], colors=[150, 150, 150], radii=cfg["point_radius"]),
            static=True,
        )


def write_rerun(out_dir: Path, report: dict, cfg: dict, path: Path, meshes: dict[str, dict]) -> None:
    """Rerun recording that opens on a ready layout: BEFORE and AFTER 3-D views side by side,
    both starting from the overview camera, and the report next to them.

    world/before (A): surface mesh, camera path, removed boxes and old positions of moves
    world/after  (B): surface mesh, camera path, added boxes and new positions of moves
    Meshes are simplified to `rerun_mesh_triangles` so the file loads quickly in the web
    viewer. With `rerun_extras`, every camera frustum, the fused surface points and the
    evidence voxels (unknown in grey) are also logged, hidden by default."""
    import rerun as rr
    import rerun.blueprint as rrb

    eye, target = overview_pose(out_dir, report, 0.0)
    side = {"A": "before", "B": "after"}

    def view(e: str) -> rrb.Spatial3DView:
        s = side[e]
        return rrb.Spatial3DView(
            name=f"{s.upper()} ({e})",
            origin="/world",
            contents=[
                f"+ /world/{s}/**",
                "+ /world/moves/**",
                f"- /world/{s}/cameras/**",
                f"- /world/{s}/points/**",
                f"- /world/{s}/evidence/**",
            ],
            background=[32, 32, 32],
            eye_controls=rrb.EyeControls3D(position=eye, look_target=target, eye_up=[0.0, 0.0, 1.0]),
        )

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            view("A"),
            view("B"),
            rrb.TextDocumentView(origin="/report", name="Report"),
            column_shares=[2, 2, 1],
        ),
        collapse_panels=True,
    )
    rr.init("what_changed", spawn=False)
    rr.save(str(path), default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    poses = json.load(open(out_dir / "align" / "poses.json"))
    boxes = list(_boxes(report))
    ev = np.load(out_dir / "diff" / "evidence.npz")
    rng = np.random.default_rng(0)
    for e, s in side.items():
        m = simplify_mesh(meshes[e], cfg["rerun_mesh_triangles"])
        rr.log(
            f"world/{s}/surface",
            rr.Mesh3D(
                vertex_positions=m["vertices"].astype(np.float32),
                triangle_indices=m["triangles"].astype(np.uint32),
                vertex_normals=m["normals"].astype(np.float32),
                vertex_colors=(m["colors"] * 255).astype(np.uint8),
            ),
            static=True,
        )
        if cfg["rerun_extras"]:
            log_extras(out_dir, e, s, cfg, poses, ev, rng)
        names = sorted(n for n, p in poses.items() if p["epoch"] == e)
        C = np.array([camera_center(np.array(poses[n]["T_cw"])) for n in names])
        col = [80, 140, 255] if e == "A" else [255, 170, 60]
        rr.log(
            f"world/{s}/camera_path",
            rr.LineStrips3D([C], colors=[col], radii=0.008, labels=[f"camera path {e}"]),
            static=True,
        )
        mine = [(b, a, t) for b, a, t, w in boxes if _shown(a, w, e)]
        if mine:
            rr.log(
                f"world/{s}/changes",
                rr.Boxes3D(
                    centers=[b["center"] for b, _, _ in mine],
                    half_sizes=[np.array(b["extents"]) / 2 for b, _, _ in mine],
                    rotation_axis_angles=[
                        rr.RotationAxisAngle(axis=[0, 0, 1], radians=b["yaw_rad"]) for b, _, _ in mine
                    ],
                    colors=[_color(cfg, a) for _, a, _ in mine],
                    labels=[t for _, _, t in mine],
                    radii=0.01,
                ),
                static=True,
            )
    moved = [c for c in report["changes"] if c["action"] == "moved"]
    if moved:
        rr.log(
            "world/moves",
            rr.Arrows3D(
                origins=[c["centroid_before"] for c in moved],
                vectors=[c["displacement"] for c in moved],
                colors=[_color(cfg, "moved")] * len(moved),
                radii=0.015,
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


def vertical_structure(points: np.ndarray, grid, min_span: float, min_points: int) -> np.ndarray:
    """(nx, ny) bool: cells whose points span at least `min_span` in height. Walls, furniture
    sides and objects do; a floor or bed top does not, wherever z = 0 is and even if the
    gravity estimate is tilted by a few tens of degrees."""
    ij = grid.index(points)
    flat = ij[:, 0] * grid.ny + ij[:, 1]
    n = grid.nx * grid.ny
    zmax = np.full(n, -np.inf)
    zmin = np.full(n, np.inf)
    np.maximum.at(zmax, flat, points[:, 2])
    np.minimum.at(zmin, flat, points[:, 2])
    count = np.bincount(flat, minlength=n)
    return ((zmax - zmin >= min_span) & (count >= min_points)).reshape(grid.nx, grid.ny)


def write_floor_map(
    out_dir: Path, report: dict, cfg: dict, path: Path, floor_height_range=(1.1, 1.9)
) -> None:
    """Top-down map of the after-room (B) that separates what was seen from what was not:
    white = surface seen with nothing standing on it (or walked), grey = obstacle now,
    hatched = never observed by either recording. Obstacles are the points in
    `obstacle_z_band` when z = 0 is reliably the floor, otherwise cells with at least
    `floor_map_min_span` of vertical structure (the map title says which). Change
    footprints (filled = there now, dashed = gone or old position), move arrows, both
    camera paths and, when the floor is in view, the route check."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon, Rectangle
    from scipy.ndimage import binary_dilation

    from src.navigation import Grid2D, densify, inflate, occupancy

    clouds = {e: np.load(out_dir / "fusion" / f"cloud_{e}.npz")["points"] for e in ("A", "B")}
    poses = json.load(open(out_dir / "align" / "poses.json"))
    paths = {
        e: np.array([camera_center(np.array(p["T_cw"])) for n, p in sorted(poses.items()) if p["epoch"] == e])
        for e in ("A", "B")
    }
    cell = cfg["floor_map_cell"]
    allxy = np.vstack([clouds["A"][:, :2], clouds["B"][:, :2]] + [p[:, :2] for p in paths.values() if len(p)])
    grid = Grid2D.around(allxy, cell, 0.3)

    seen = np.zeros((grid.nx, grid.ny), bool)
    for pts in clouds.values():
        ij = grid.index(pts)
        seen[ij[:, 0], ij[:, 1]] = True
    seen = binary_dilation(seen, iterations=1)  # surface points are ~2 cm apart; close gaps
    walked = np.zeros_like(seen)
    for p in paths.values():
        if len(p) > 1:
            ij = grid.index(densify(p[:, :2], cell / 2))
            walked[ij[:, 0], ij[:, 1]] = True
    known = seen | inflate(walked, int(np.ceil(cfg["floor_map_walked_radius"] / cell)))
    from src.alignment import floor_is_reliable

    floor_ok = floor_is_reliable(json.load(open(out_dir / "align" / "alignment.json")), floor_height_range)
    if floor_ok:
        obst = occupancy(clouds["B"], grid, tuple(cfg["obstacle_z_band"]), 3)
    else:
        obst = vertical_structure(clouds["B"], grid, cfg["floor_map_min_span"], 3)

    obstacle_rgb = (0.42, 0.42, 0.45)
    rgba = np.zeros((grid.ny, grid.nx, 4))
    rgba[known.T] = (1.0, 1.0, 1.0, 1.0)
    rgba[obst.T] = (*obstacle_rgb, 1.0)
    extent = (grid.x0, grid.x0 + grid.nx * cell, grid.y0, grid.y0 + grid.ny * cell)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#ececec")
    ax.add_patch(
        Rectangle(
            (extent[0], extent[2]),
            extent[1] - extent[0],
            extent[3] - extent[2],
            hatch="////",
            fill=False,
            edgecolor="#cfcfcf",
            lw=0,
            zorder=0,
        )
    )
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest", zorder=1)

    path_cols = {"A": "tab:blue", "B": "tab:orange"}
    for e, C in paths.items():
        if len(C):
            ax.plot(C[:, 0], C[:, 1], "-", color=path_cols[e], lw=1.5, zorder=3)
            ax.plot(C[-1, 0], C[-1, 1], "o", color=path_cols[e], ms=4, zorder=3)

    anchors = []
    for box, action, text, which in _boxes(report):
        foot = np.array(box["corners"])[[0, 2, 6, 4], :2]  # bottom corners in cyclic order
        col = np.array(_color(cfg, action)) / 255
        present_now = action == "added" or which == "after"  # removed / moved-before: gone now
        ax.add_patch(
            Polygon(
                foot,
                closed=True,
                fill=present_now,
                alpha=0.45 if present_now else 1,
                edgecolor=col,
                facecolor=col,
                ls="-" if present_now else "--",
                lw=2,
                zorder=4,
            )
        )
        anchors.append((np.array(box["center"][:2]), text, col))
    for c in report["changes"]:
        if c["action"] == "moved":
            ax.annotate(
                "",
                xy=c["centroid_after"][:2],
                xytext=c["centroid_before"][:2],
                zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=np.array(_color(cfg, "moved")) / 255, lw=2),
            )
    route_handles = []
    route_path = out_dir / "nav" / "route.json"
    if route_path.exists():
        route = json.load(open(route_path))
        blocked = np.array(route["blocked_points"]).reshape(-1, 2)
        if len(blocked):
            ax.plot(blocked[:, 0], blocked[:, 1], "x", color="red", ms=5, zorder=5)
            route_handles.append(Line2D([], [], color="red", marker="x", ls="", label="route A, blocked now"))
        if route.get("replanned"):
            p = np.array(route["replanned"])
            ax.plot(p[:, 0], p[:, 1], "--", color="tab:green", lw=2, zorder=5)
            route_handles.append(Line2D([], [], color="tab:green", ls="--", label="new route (A*)"))

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    place_labels(fig, ax, anchors)

    handles = [
        Patch(facecolor="white", edgecolor="0.6", label="seen, nothing standing (or walked)"),
        Patch(facecolor=obstacle_rgb, label="obstacle now (B)"),
        Patch(facecolor="#ececec", edgecolor="#bdbdbd", hatch="////", label="never observed"),
        Line2D([], [], color="tab:blue", label="camera path A (dot = end)"),
        Line2D([], [], color="tab:orange", label="camera path B (dot = end)"),
    ]
    for a in ACTION_ORDER:
        if any(c["action"] == a for c in report["changes"]):
            handles.append(Patch(facecolor=np.array(_color(cfg, a)) / 255, alpha=0.6, label=a))
    ax.legend(
        handles=handles + route_handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False
    )
    ax.set_xlabel(f"x ({report['units']})")
    ax.set_ylabel(f"y ({report['units']})")
    rule = (
        f"obstacles: {cfg['obstacle_z_band'][0]:g}-{cfg['obstacle_z_band'][1]:g} m above the floor"
        if floor_ok
        else f"floor not reliably found: obstacles = cells with >= {cfg['floor_map_min_span']:g} m of vertical structure"
    )
    ax.set_title(
        "Top-down map after the change (filled = there now, dashed = gone / old position)\n" + rule,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def place_labels(fig, ax, anchors: list[tuple[np.ndarray, str, np.ndarray]]) -> None:
    """Labels beside their boxes, pushed apart vertically so none overlap, each with a
    leader line to its box centre."""
    if not anchors:
        return
    fig.canvas.draw()
    to_px, to_data = ax.transData.transform, ax.transData.inverted().transform
    h = 15.0  # label height in pixels at fontsize 8, dpi 130, with its frame
    xmid = np.mean(ax.get_xlim())
    items = []
    for xy, text, col in anchors:
        px = to_px(xy)
        right = xy[0] <= xmid  # put the label on the side with more room
        items.append(
            {
                "anchor": px,
                "x": px[0] + (45 if right else -45),
                "y": px[1],
                "right": right,
                "w": 5.6 * len(text) + 8,
                "text": text,
                "col": col,
            }
        )
    for _ in range(300):  # repel in y the labels whose boxes overlap in x
        moved = False
        items.sort(key=lambda it: it["y"])
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                ax0, ax1 = (a["x"], a["x"] + a["w"]) if a["right"] else (a["x"] - a["w"], a["x"])
                bx0, bx1 = (b["x"], b["x"] + b["w"]) if b["right"] else (b["x"] - b["w"], b["x"])
                if min(ax1, bx1) > max(ax0, bx0) and abs(b["y"] - a["y"]) < h:
                    push = (h - abs(b["y"] - a["y"])) / 2 + 0.5
                    a["y"] -= push
                    b["y"] += push
                    moved = True
        if not moved:
            break
    for it in items:
        ax.annotate(
            it["text"],
            xy=to_data(it["anchor"]),
            xytext=to_data((it["x"], it["y"])),
            ha="left" if it["right"] else "right",
            va="center",
            fontsize=8,
            color=it["col"] * 0.75,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=it["col"], lw=0.8, alpha=0.9),
            arrowprops=dict(arrowstyle="-", color=it["col"], lw=0.8),
        )


def load_meshes(out_dir: Path, weight_threshold: float) -> dict[str, dict]:
    """Zero-level surface of each recording's TSDF grid: vertices, triangles, vertex colours
    (0-1) and vertex normals. A closed, coloured surface reads as a room, where the 2 cm
    surface points read as a scatter of dots."""
    import open3d as o3d

    meshes = {}
    for e in ("A", "B"):
        grid = o3d.t.geometry.VoxelBlockGrid.load(str(out_dir / "fusion" / f"grid_{e}.npz"))
        m = grid.extract_triangle_mesh(weight_threshold).to_legacy()
        m.compute_vertex_normals()
        meshes[e] = {
            "vertices": np.asarray(m.vertices, np.float64),
            "triangles": np.asarray(m.triangles, np.int64),
            "colors": np.clip(np.asarray(m.vertex_colors, np.float64), 0, 1),
            "normals": np.asarray(m.vertex_normals, np.float64),
        }
    return meshes


def inside_box(box: dict, X: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """True for points X (N, 3) inside a yaw-only box grown by `margin`."""
    c, s = np.cos(box["yaw_rad"]), np.sin(box["yaw_rad"])
    local = (X[:, :2] - np.array(box["center"][:2])) @ np.array([[c, -s], [s, c]])
    half = np.array(box["extents"]) / 2 + margin
    return (
        (np.abs(local[:, 0]) <= half[0])
        & (np.abs(local[:, 1]) <= half[1])
        & (np.abs(X[:, 2] - box["center"][2]) <= half[2])
    )


class MeshView:
    """Ray-cast renderer for one mesh: interpolated vertex colours, soft shading, and the
    3-D hit point of every pixel (used to tint the surface inside change boxes)."""

    BACKGROUND = np.array([32, 32, 32], np.float64)

    def __init__(self, mesh: dict):
        import open3d as o3d
        import open3d.core as o3c

        self.mesh = mesh
        self.scene = o3d.t.geometry.RaycastingScene()
        if len(mesh["triangles"]):
            self.scene.add_triangles(
                o3c.Tensor(mesh["vertices"].astype(np.float32)),
                o3c.Tensor(mesh["triangles"].astype(np.uint32)),
            )

    def render(
        self, K: np.ndarray, T_cw: np.ndarray, W: int, H: int, tints: list[tuple[dict, list[int]]] = ()
    ) -> np.ndarray:
        """(H, W, 3) uint8 RGB. `tints`: (box, rgb) pairs; surface inside a box is blended
        towards that colour, so the changed object itself lights up."""
        import open3d.core as o3c

        from src.utils.geometry import pixel_rays

        img = np.tile(self.BACKGROUND, (H * W, 1))
        if not len(self.mesh["triangles"]):
            return img.reshape(H, W, 3).astype(np.uint8)
        o, d = pixel_rays(K, T_cw, H, W)
        ans = self.scene.cast_rays(o3c.Tensor(np.concatenate([o, d], axis=1).astype(np.float32)))
        t = ans["t_hit"].numpy()
        hit = np.isfinite(t)
        tri = self.mesh["triangles"][ans["primitive_ids"].numpy()[hit]]
        uv = ans["primitive_uvs"].numpy()[hit]
        w = np.c_[1 - uv.sum(axis=1), uv]  # barycentric weights of the three vertices
        col = np.einsum("nk,nkc->nc", w, self.mesh["colors"][tri])
        n = np.einsum("nk,nkc->nc", w, self.mesh["normals"][tri])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        view = d[hit] / np.linalg.norm(d[hit], axis=1, keepdims=True)
        # headlight plus a little light from above: shapes read without hiding the colours
        shade = 0.55 + 0.35 * np.abs(np.sum(n * view, axis=1)) + 0.10 * np.clip(n[:, 2], 0, 1)
        rgb = np.clip(col * shade[:, None], 0, 1) * 255
        X = o[hit] + t[hit, None] * d[hit]
        for box, color in tints:
            m = inside_box(box, X, margin=0.02)
            rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(color, np.float64)
        img[hit] = rgb
        return img.reshape(H, W, 3).astype(np.uint8)


def overview_pose(out_dir: Path, report: dict, angle_deg: float):
    """(eye, target) of an elevated virtual camera behind the recorded paths, looking at the
    changes (or the scene centre), rotated by `angle_deg` about the vertical axis."""
    poses = json.load(open(out_dir / "align" / "poses.json"))
    C = np.array([camera_center(np.array(p["T_cw"])) for p in poses.values()])
    fwd = np.array([np.array(p["T_cw"])[2, :3] for p in poses.values()]).mean(axis=0)
    boxes = [b for b, *_ in _boxes(report)]
    if boxes:
        target = np.mean([b["center"] for b in boxes], axis=0)
    else:  # scene centre: about 1.5 m in front of the cameras
        target = C.mean(axis=0) + fwd / np.linalg.norm(fwd) * 1.5
    back = C.mean(axis=0) - target
    back[2] = 0
    dist = max(np.linalg.norm(back), 1.0)
    back /= dist
    a = np.deg2rad(angle_deg)
    back = np.array([back[0] * np.cos(a) - back[1] * np.sin(a), back[0] * np.sin(a) + back[1] * np.cos(a), 0])
    eye = target + back * dist * 1.2 + np.array([0, 0, 1.0]) * dist * 0.8
    return eye, target


def overview_camera(out_dir: Path, report: dict, angle_deg: float, W: int, H: int, pts: np.ndarray):
    """(K, T_cw) for `overview_pose`; focal length and principal point are fitted so that
    98 % of `pts` fill the image."""
    from src.utils.geometry import look_at_T_cw

    eye, target = overview_pose(out_dir, report, angle_deg)
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


def _shown(action: str, which: str | None, epoch: str) -> bool:
    """Is this box drawn in this recording's view? Removed and old positions in A, added and
    new positions in B."""
    return (
        (action == "removed" and epoch == "A")
        or (action == "added" and epoch == "B")
        or (action == "moved" and ((which == "before") == (epoch == "A")))
    )


def render_overview(
    out_dir: Path, report: dict, cfg: dict, views: dict[str, MeshView], angle_deg: float, W: int, H: int
) -> np.ndarray:
    """BEFORE | AFTER from one virtual camera: shaded meshes rendered at twice the size and
    averaged down, changed surfaces tinted, then camera paths, boxes and labels."""
    import cv2

    ss = 2
    pts = views["B"].mesh["vertices"]
    K, T = overview_camera(out_dir, report, angle_deg, W, H, pts)
    Ks = K.copy()
    Ks[:2] *= ss  # exact for COLMAP corner-origin pixel coordinates
    poses = json.load(open(out_dir / "align" / "poses.json"))
    boxes = list(_boxes(report))
    tiles = []
    for e in ("A", "B"):
        tints = [(b, _color(cfg, a)) for b, a, _, w in boxes if _shown(a, w, e)]
        big = views[e].render(Ks, T, W * ss, H * ss, tints)
        img = cv2.cvtColor(cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        C = np.array(
            [camera_center(np.array(p["T_cw"])) for n, p in sorted(poses.items()) if p["epoch"] == e]
        )
        uv, z = project_points(C, K, T)
        path = np.round(uv[z > 0]).astype(np.int32)
        col = (255, 140, 80) if e == "A" else (60, 170, 255)
        if len(path) > 1:
            cv2.polylines(img, [path], False, col, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(path[-1]), 4, col, -1, cv2.LINE_AA)
        for box, action, text, which in boxes:
            if _shown(action, which, e):
                draw_box(img, box, K, T, _color(cfg, action), text)
        title = "BEFORE (A)" if e == "A" else "AFTER (B)"
        cv2.putText(img, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    return cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB)


def write_overview(out_dir: Path, report: dict, cfg: dict, meshes: dict[str, dict]) -> None:
    """scene_3d.png (one view) and scene_3d.gif (a slow orbit) of the fused recordings."""
    import imageio.v3 as iio

    views = {e: MeshView(m) for e, m in meshes.items()}
    W, H = cfg["overview_size"]
    iio.imwrite(str(out_dir / "scene_3d.png"), render_overview(out_dir, report, cfg, views, 0.0, W, H))
    span, n = cfg["overview_orbit_deg"], 18
    angles = np.concatenate([np.linspace(-span, span, n), np.linspace(span, -span, n)[1:-1]])
    w, h = W // 2, H // 2
    frames = [render_overview(out_dir, report, cfg, views, a, w, h) for a in angles]
    write_gif(frames, out_dir / "scene_3d.gif", duration_ms=110)


def write_gif(frames: list[np.ndarray], path: Path, duration_ms: int) -> None:
    """Animated GIF with ONE palette for all frames and no dithering: per-frame palettes
    make colours flicker, and dithering adds speckle noise to smooth surfaces."""
    from PIL import Image

    sample = np.vstack([f[::4, ::4] for f in frames])
    palette = Image.fromarray(sample).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)


def generate_visualizations(
    out_dir: Path, cfg: dict, fusion_cfg: dict, floor_height_range=(1.1, 1.9)
) -> dict:
    """Stage entry."""
    report = build_report(out_dir)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, out_dir / "report.md")
    print(f"  report.json / report.md: {report['counts']}")
    meshes = None
    if cfg["rerun_enabled"] or cfg["overview_enabled"]:
        meshes = load_meshes(out_dir, fusion_cfg["extract_weight_threshold"])
    if cfg["rerun_enabled"]:
        write_rerun(out_dir, report, cfg, out_dir / "changes.rrd", meshes)
        print(f"  Rerun recording: {out_dir / 'changes.rrd'}")
    if cfg["mp4_enabled"]:
        write_side_by_side(out_dir, report, cfg, out_dir / "side_by_side.mp4")
        print(f"  Side-by-side video: {out_dir / 'side_by_side.mp4'}")
    if cfg["floor_map_enabled"]:
        write_floor_map(out_dir, report, cfg, out_dir / "floor_map.png", floor_height_range)
        print(f"  Floor map: {out_dir / 'floor_map.png'}")
    if cfg["overview_enabled"]:
        write_overview(out_dir, report, cfg, meshes)
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
