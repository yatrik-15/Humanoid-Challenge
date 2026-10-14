"""report.json / report.md: the detected changes with their labels, and helpers shared by the
other outputs (box colours, which boxes each recording shows)."""

import json
from pathlib import Path

ACTION_ORDER = ("moved", "removed", "added")


def action_color(cfg: dict, action: str) -> list[int]:
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
        "navigation": route_summary(out_dir),
        "camera_path_length": align.get("camera_path_length"),
        "camera_path_short": align.get("camera_path_short", []),
    }


def route_summary(out_dir: Path) -> dict | None:
    """nav/route.json without the polylines, or None when the nav stage was skipped."""
    path = out_dir / "nav" / "route.json"
    if not path.exists():
        return None
    route = json.load(open(path))
    return {k: v for k, v in route.items() if k not in ("route", "blocked_points", "replanned")}


def display_label(c: dict) -> str:
    """Label as shown to people: "(no label)" when the VLM gave no valid answer."""
    return "(no label)" if c["label"] == "label_unavailable" else c["label"]


def path_warning(report: dict) -> list[str]:
    """Markdown lines warning that a recording's camera hardly moved (empty if fine)."""
    short = report.get("camera_path_short") or []
    if not short:
        return []
    paths = report["camera_path_length"]
    which = " and ".join(f"{paths[e]:.2f} m in {e}" for e in short)
    return [
        f"> **Note:** the camera moved only {which}. Changes seen from few distinct viewpoints may "
        "not be confirmed; walk sideways past the area when recording.",
        "",
    ]


def write_markdown(report: dict, path: Path) -> None:
    source = report["scale_note"].replace("APPROXIMATE metres from ", "scale from ")
    L = [
        "# What changed in the room",
        "",
        f"Units: **{report['units']}** ({source}). World frame: {report['world_frame']}.",
        "",
        f"Detected: **{report['counts']['moved']} moved, {report['counts']['removed']} removed, "
        f"{report['counts']['added']} added**. Regions observed in only one recording are reported "
        f"as UNKNOWN ({report['unknown_evidence_voxels']} evidence voxels), never as changes.",
        "",
        *path_warning(report),
        "| id | action | object | label conf. | where (centroid x, y, z) | size / displacement | view groups |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in report["changes"]:
        if c["action"] == "moved":
            where = f"{fmt_xyz(c['centroid_before'])} → {fmt_xyz(c['centroid_after'])}"
            size = f"moved {c['distance']:.2f}"
        else:
            where = fmt_xyz(c["centroid"])
            size = " x ".join(f"{e:.2f}" for e in c["box"]["extents"])
        L.append(
            f"| {c['id']} | {c['action']} | {display_label(c)} | {c['label_confidence']:.2f} | {where} | {size} | "
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
            L.append(f"| {c['id']} | {display_label(c)} | {chk['verdict']} | {gain} | {top} |")
    L += ["", "## Descriptions", ""]
    for c in report["changes"]:
        L.append(
            f"- **#{c['id']} {c['action']} — {display_label(c)}**: {c['description']}"
            + (f" {c['spatial_summary']}" if c["spatial_summary"] else "")
        )
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def fmt_xyz(v) -> str:
    return "(" + ", ".join(f"{x:.2f}" for x in v) + ")"


def change_boxes(report: dict):
    """Yield (box dict, action, label text, which) for every box in the report."""
    for c in report["changes"]:
        name = "object" if c["label"] == "label_unavailable" else c["label"]  # box labels stay short
        if c["action"] == "moved":
            yield c["box_before"], "moved", f"#{c['id']} {name} (before)", "before"
            yield c["box_after"], "moved", f"#{c['id']} {name} (after)", "after"
        else:
            yield c["box"], c["action"], f"#{c['id']} {name} ({c['action']})", None


def shown_in(action: str, which: str | None, epoch: str) -> bool:
    """Is this box drawn in this recording's view? Removed and old positions in A, added and
    new positions in B."""
    return (
        (action == "removed" and epoch == "A")
        or (action == "added" and epoch == "B")
        or (action == "moved" and ((which == "before") == (epoch == "A")))
    )


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
