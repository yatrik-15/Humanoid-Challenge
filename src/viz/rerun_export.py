"""changes.rrd: Rerun recording that opens on a BEFORE / AFTER layout.

Rerun transforms are applied as X_parent = mat3x3 @ X + translation."""

import json
from pathlib import Path

import numpy as np

from src.utils.geometry import camera_center, invert_pose
from src.viz.render import overview_pose, simplify_mesh
from src.viz.report import action_color, change_boxes, shown_in


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
            rr.Points3D(ev[f"{k}_points"], colors=action_color(cfg, k), radii=cfg["point_radius"]),
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
    boxes = list(change_boxes(report))
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
        mine = [(b, a, t) for b, a, t, w in boxes if shown_in(a, w, e)]
        if mine:
            rr.log(
                f"world/{s}/changes",
                rr.Boxes3D(
                    centers=[b["center"] for b, _, _ in mine],
                    half_sizes=[np.array(b["extents"]) / 2 for b, _, _ in mine],
                    rotation_axis_angles=[
                        rr.RotationAxisAngle(axis=[0, 0, 1], radians=b["yaw_rad"]) for b, _, _ in mine
                    ],
                    colors=[action_color(cfg, a) for _, a, _ in mine],
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
                colors=[action_color(cfg, "moved")] * len(moved),
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
