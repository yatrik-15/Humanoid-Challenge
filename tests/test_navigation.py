"""Tests for the route check: occupancy, A*, blocking attribution and replanning."""

import numpy as np

from src.navigation import Grid2D, astar, check_route, occupancy

CFG = {"robot_radius": 0.2, "min_points_per_cell": 3, "min_blocked_cells": 3}
BAND = (0.05, 1.8)


def box_points(x0, y0, x1, y1, height, step=0.02):
    xs, ys, zs = np.meshgrid(
        np.arange(x0, x1, step), np.arange(y0, y1, step), np.arange(0.1, height, step), indexing="ij"
    )
    return np.c_[xs.ravel(), ys.ravel(), zs.ravel()]


def scene(with_box: bool):
    """A 4 x 2 m room with walls; B optionally has a box in the middle of the corridor."""
    walls = np.vstack([box_points(0, -0.05, 4, 0, 1.0), box_points(0, 2, 4, 2.05, 1.0)])
    floor = np.c_[np.random.default_rng(0).uniform([0, 0], [4, 2], (5000, 2)), np.zeros(5000)]
    A = np.vstack([walls, floor])
    B = np.vstack([A, box_points(1.8, 0.7, 2.2, 1.3, 0.5)]) if with_box else A
    grid = Grid2D.around(np.vstack([A, B])[:, :2], 0.05, 0.2)
    occ_A = occupancy(A, grid, BAND, CFG["min_points_per_cell"])
    occ_B = occupancy(B, grid, BAND, CFG["min_points_per_cell"])
    return grid, occ_A, occ_B


ROUTE = np.array([[0.3, 1.0], [3.7, 1.0]])
BOX_CHANGE = {
    "id": 4,
    "action": "added",
    "box": {"center": [2.0, 1.0, 0.25], "extents": [0.4, 0.6, 0.5], "yaw_rad": 0.0},
}


def test_occupancy_ignores_the_floor():
    grid, occ_A, _ = scene(with_box=False)
    mid = grid.index(np.array([[2.0, 1.0]]))[0]
    wall = grid.index(np.array([[2.0, 2.02]]))[0]
    assert not occ_A[tuple(mid)]
    assert occ_A[tuple(wall)]


def test_astar_goes_around_a_wall():
    blocked = np.zeros((20, 20), bool)
    blocked[10, 0:15] = True
    path = astar(blocked, (2, 2), (18, 2))
    assert path[0] == (2, 2) and path[-1] == (18, 2)
    assert not any(blocked[p] for p in path)
    assert max(p[1] for p in path) >= 15  # had to go past the end of the wall
    blocked[10, :] = True
    assert astar(blocked, (2, 2), (18, 2)) is None


def test_added_box_blocks_route_and_is_blamed():
    grid, occ_A, occ_B = scene(with_box=True)
    r = check_route(ROUTE, occ_A, occ_B, grid, [BOX_CHANGE], CFG)
    assert r["blocked_by_changes"] == [4]
    assert r["route_cells_blocked_now"] > 0
    assert r["replanned"] is not None
    assert r["replanned_length"] > r["route_length"]
    # the new route keeps the robot radius away from the box
    p = np.array(r["replanned"])
    dx = np.maximum(np.abs(p[:, 0] - 2.0) - 0.2, 0)
    dy = np.maximum(np.abs(p[:, 1] - 1.0) - 0.3, 0)
    assert np.hypot(dx, dy).min() >= CFG["robot_radius"]


def test_unchanged_room_keeps_route():
    grid, occ_A, occ_B = scene(with_box=False)
    r = check_route(ROUTE, occ_A, occ_B, grid, [], CFG)
    assert r["route_cells_blocked_now"] == 0
    assert r["blocked_by_changes"] == []
    assert "replanned" not in r


def test_ros_map_orientation_and_values(tmp_path):
    from src.navigation import Grid2D, write_ros_map

    grid = Grid2D(x0=-1.0, y0=2.0, nx=4, ny=3, cell=0.5)
    occ = np.zeros((4, 3), bool)
    free = np.zeros((4, 3), bool)
    occ[3, 2] = True  # largest x, largest y -> top-right pixel
    free[0, 0] = True  # smallest x, smallest y -> bottom-left pixel
    write_ros_map(tmp_path / "m", occ, free, grid)
    raw = (tmp_path / "m.pgm").read_bytes()
    header, data = raw[:11], raw[11:]
    assert header == b"P5\n4 3\n255\n"
    img = np.frombuffer(data, np.uint8).reshape(3, 4)
    assert img[0, 3] == 0 and img[2, 0] == 254 and img[1, 1] == 205
    yaml_text = (tmp_path / "m.yaml").read_text()
    assert "origin: [-1.0000, 2.0000, 0.0]" in yaml_text and "resolution: 0.5" in yaml_text


def test_replanning_never_crosses_unknown_space():
    grid, occ_A, occ_B = scene(with_box=True)
    centres_y = grid.centers(np.c_[np.zeros(grid.ny, int), np.arange(grid.ny)])[:, 1]
    known = np.broadcast_to(centres_y < 1.3, (grid.nx, grid.ny)).copy()  # above y = 1.3 unobserved
    r = check_route(ROUTE, occ_A, occ_B, grid, [BOX_CHANGE], CFG, known_free=known)
    p = np.array(r["replanned"])
    assert p[:, 1].max() < 1.3  # went around below the box, not through the unknown side
    known[:] = False
    known[:, (centres_y > 0.7) & (centres_y < 1.3)] = True  # only the blocked corridor is known
    r = check_route(ROUTE, occ_A, occ_B, grid, [BOX_CHANGE], CFG, known_free=known)
    assert r["replanned"] is None
