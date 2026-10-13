"""Route check: can a robot still follow the route walked in the first recording?

The camera path of recording A is a route a person actually walked. Both fused clouds are
turned into 2-D occupancy grids (points in the obstacle height band), inflated by the robot
radius, and A's route is replayed on B's grid. Route cells that are now occupied are
blocked; they are attributed to the detected changes whose footprint they fall in, and a
new route between the same endpoints is planned with A* on B's grid.

Only meaningful when z = 0 is the floor: in recordings where the dominant plane is a bed or
table the "obstacle band" is relative to that surface, so the stage is skipped unless the
floor was found and the cameras sit at a plausible handheld height above it.

Reads  fusion/cloud_{A,B}.npz, align/{alignment,poses}.json, cluster/changes.json
Writes nav/route.json
"""

import heapq
import json
from pathlib import Path

import numpy as np

from src.utils.geometry import camera_center


class Grid2D:
    """Square cells of size `cell` covering [x0, x0 + nx * cell) x [y0, y0 + ny * cell)."""

    def __init__(self, x0: float, y0: float, nx: int, ny: int, cell: float):
        self.x0, self.y0, self.nx, self.ny, self.cell = x0, y0, nx, ny, cell

    @classmethod
    def around(cls, xy: np.ndarray, cell: float, margin: float) -> "Grid2D":
        lo = xy.min(axis=0) - margin
        hi = xy.max(axis=0) + margin
        n = np.ceil((hi - lo) / cell).astype(int) + 1
        return cls(float(lo[0]), float(lo[1]), int(n[0]), int(n[1]), cell)

    def index(self, xy: np.ndarray) -> np.ndarray:
        """(N, 2) world xy -> (N, 2) integer (ix, iy), clipped to the grid."""
        ij = np.floor((np.asarray(xy)[:, :2] - [self.x0, self.y0]) / self.cell).astype(int)
        return np.clip(ij, 0, [self.nx - 1, self.ny - 1])

    def centers(self, ij: np.ndarray) -> np.ndarray:
        return (np.asarray(ij) + 0.5) * self.cell + [self.x0, self.y0]


def occupancy(points: np.ndarray, grid: Grid2D, z_band: tuple[float, float], min_points: int) -> np.ndarray:
    """(nx, ny) bool: cells with at least min_points points inside the height band."""
    lo, hi = z_band
    pts = points[(points[:, 2] > lo) & (points[:, 2] < hi)]
    counts = np.zeros((grid.nx, grid.ny), dtype=int)
    if len(pts):
        ij = grid.index(pts)
        np.add.at(counts, (ij[:, 0], ij[:, 1]), 1)
    return counts >= min_points


def inflate(occ: np.ndarray, radius_cells: int) -> np.ndarray:
    """Dilate occupied cells by a disk, so the robot can be planned as a point."""
    from scipy.ndimage import binary_dilation

    r = radius_cells
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return binary_dilation(occ, structure=(xx**2 + yy**2) <= r * r)


def astar(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """8-connected A* on a bool grid (True = blocked). Returns the cell path or None."""
    nx, ny = blocked.shape
    steps = [(dx, dy, np.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    g = {start: 0.0}
    parent = {}
    heap = [(np.hypot(goal[0] - start[0], goal[1] - start[1]), start)]
    done = set()
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        if cur in done:
            continue
        done.add(cur)
        for dx, dy, cost in steps:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny) or blocked[nxt]:
                continue
            if dx and dy and (blocked[cur[0] + dx, cur[1]] or blocked[cur[0], cur[1] + dy]):
                continue  # no corner cutting
            ng = g[cur] + cost
            if ng < g.get(nxt, np.inf):
                g[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(heap, (ng + np.hypot(goal[0] - nxt[0], goal[1] - nxt[1]), nxt))
    return None


def densify(xy: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline so consecutive points are at most `step` apart."""
    out = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t for t in np.arange(1, n + 1) / n]
    return np.array(out)


def footprint_contains(box: dict, xy: np.ndarray, margin: float) -> np.ndarray:
    """True for xy points inside the box footprint grown by `margin`."""
    c, s = np.cos(box["yaw_rad"]), np.sin(box["yaw_rad"])
    local = (xy - np.array(box["center"][:2])) @ np.array([[c, -s], [s, c]])
    half = np.array(box["extents"][:2]) / 2 + margin
    return np.all(np.abs(local) <= half, axis=1)


def present_boxes(changes: list[dict]) -> list[tuple[int, dict]]:
    """(change id, box) for objects that exist in B: added ones and moved ones' new place."""
    out = []
    for ch in changes:
        if ch["action"] == "added":
            out.append((ch["id"], ch["box"]))
        elif ch["action"] == "moved":
            out.append((ch["id"], ch["box_after"]))
    return out


def check_route(
    route_xy: np.ndarray,
    occ_A: np.ndarray,
    occ_B: np.ndarray,
    grid: Grid2D,
    changes: list[dict],
    cfg: dict,
) -> dict:
    """Replay route_xy on the inflated occupancy of A and B; replan on B if it is blocked."""
    r_cells = int(np.ceil(cfg["robot_radius"] / grid.cell))
    blocked_A, blocked_B = inflate(occ_A, r_cells), inflate(occ_B, r_cells)
    route = densify(route_xy, grid.cell / 2)
    ij = grid.index(route)
    # The person stood at the endpoints, so they are free even if the inflation says otherwise.
    for end in (ij[0], ij[-1]):
        x, y = end
        blocked_B[max(0, x - r_cells) : x + r_cells + 1, max(0, y - r_cells) : y + r_cells + 1] = False
    hit_B = blocked_B[ij[:, 0], ij[:, 1]]
    hit_A = blocked_A[ij[:, 0], ij[:, 1]]
    newly = hit_B & ~hit_A  # blocked now, free before: caused by a change, not by A's map noise

    blockers = []
    for cid, box in present_boxes(changes):
        if np.any(newly & footprint_contains(box, route, cfg["robot_radius"])):
            blockers.append(cid)

    result = {
        "route_length": float(np.sum(np.linalg.norm(np.diff(route_xy, axis=0), axis=1))),
        "route_cells_blocked_now": int(newly.sum()),
        "route_cells_blocked_in_A": int(hit_A.sum()),
        "blocked_by_changes": blockers,
        "route": np.round(route_xy, 3).tolist(),
        "blocked_points": np.round(route[newly], 3).tolist(),
    }
    if blockers or newly.sum() >= cfg["min_blocked_cells"]:
        path = astar(blocked_B, tuple(ij[0]), tuple(ij[-1]))
        if path is None:
            result["replanned"] = None
        else:
            p = grid.centers(np.array(path))
            result["replanned"] = np.round(p, 3).tolist()
            result["replanned_length"] = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    return result


def plan_routes(out_dir: Path, cfg: dict, obstacle_z_band) -> dict | None:
    """Stage entry: writes nav/route.json, or nothing if z = 0 is not the floor."""
    align = json.load(open(out_dir / "align" / "alignment.json"))
    lo, hi = cfg["camera_height_range"]
    cam_h = align["camera_height_median"]
    if not align["floor_found"] or not (lo <= cam_h <= hi):
        (out_dir / "nav" / "route.json").unlink(missing_ok=True)
        print(
            f"  skipped: floor_found={align['floor_found']}, camera height {cam_h:.2f} "
            f"outside [{lo}, {hi}], so z = 0 is probably not the floor"
        )
        return None

    poses = json.load(open(out_dir / "align" / "poses.json"))
    route = np.array(
        [camera_center(np.array(p["T_cw"]))[:2] for n, p in sorted(poses.items()) if p["epoch"] == "A"]
    )
    clouds = {e: np.load(out_dir / "fusion" / f"cloud_{e}.npz")["points"] for e in ("A", "B")}
    grid = Grid2D.around(np.vstack([clouds["A"][:, :2], clouds["B"][:, :2], route]), cfg["cell_size"], 0.5)
    occ = {e: occupancy(clouds[e], grid, obstacle_z_band, cfg["min_points_per_cell"]) for e in clouds}
    changes = json.load(open(out_dir / "cluster" / "changes.json"))
    result = check_route(route, occ["A"], occ["B"], grid, changes, cfg)

    (out_dir / "nav").mkdir(exist_ok=True)
    with open(out_dir / "nav" / "route.json", "w") as f:
        json.dump(result, f, indent=2)
    msg = f"  route walked in A: {result['route_length']:.2f} m, {result['route_cells_blocked_now']} cells now blocked"
    if result["blocked_by_changes"]:
        msg += f" by change(s) {result['blocked_by_changes']}"
    if "replanned" in result:
        msg += (
            f"; new route {result['replanned_length']:.2f} m"
            if result["replanned"]
            else "; no route around it"
        )
    print(msg)
    return result
