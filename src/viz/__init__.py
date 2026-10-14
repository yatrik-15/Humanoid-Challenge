"""Reports and visualizations of a run.

Writes into the run directory:
    report.json, report.md   the changes with labels, boxes, centroids, displacement (report)
    changes.rrd              Rerun recording, BEFORE / AFTER layout (rerun_export)
    side_by_side.mp4         B frames next to the nearest-pose A frames, boxes projected (images)
    floor_map.png            seen / occupied / never-observed map with the changes (floor_map)
    scene_3d.png / .gif      both rooms from one virtual camera, changes tinted (render)
"""

import json
from pathlib import Path

from src.viz.floor_map import vertical_structure, write_floor_map
from src.viz.images import draw_box, write_side_by_side
from src.viz.render import MeshView, load_meshes, render_overview, simplify_mesh, write_overview
from src.viz.report import (
    ACTION_ORDER,
    action_color,
    append_visuals,
    build_report,
    change_boxes,
    display_label,
    write_markdown,
)
from src.viz.rerun_export import write_rerun

__all__ = [
    "ACTION_ORDER",
    "MeshView",
    "action_color",
    "append_visuals",
    "build_report",
    "change_boxes",
    "display_label",
    "draw_box",
    "generate_visualizations",
    "load_meshes",
    "render_overview",
    "simplify_mesh",
    "vertical_structure",
    "write_floor_map",
    "write_markdown",
    "write_overview",
    "write_rerun",
    "write_side_by_side",
]


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
