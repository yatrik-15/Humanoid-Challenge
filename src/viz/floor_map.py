"""floor_map.png: top-down map of seen, occupied and never-observed space with the changes."""

import json
from pathlib import Path

import numpy as np

from src.utils.geometry import camera_center
from src.viz.report import ACTION_ORDER, action_color, change_boxes


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
    for box, action, text, which in change_boxes(report):
        foot = np.array(box["corners"])[[0, 2, 6, 4], :2]  # bottom corners in cyclic order
        col = np.array(action_color(cfg, action)) / 255
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
                arrowprops=dict(arrowstyle="-|>", color=np.array(action_color(cfg, "moved")) / 255, lw=2),
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
            handles.append(Patch(facecolor=np.array(action_color(cfg, a)) / 255, alpha=0.6, label=a))
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
