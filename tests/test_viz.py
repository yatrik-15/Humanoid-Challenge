"""Mesh renderer used for the 3-D overview."""

import numpy as np

from src.utils.geometry import look_at_T_cw
from src.viz import MeshView, inside_box

K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
T = look_at_T_cw(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]))


def quad(z: float, color, half: float = 0.2) -> dict:
    v = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]])
    return {
        "vertices": v,
        "triangles": np.array([[0, 1, 2], [0, 2, 3]]),
        "colors": np.tile(np.array(color, float), (4, 1)),
        "normals": np.tile([0.0, 0.0, -1.0], (4, 1)),
    }


def merge(*meshes):
    out = {k: [] for k in ("vertices", "triangles", "colors", "normals")}
    off = 0
    for m in meshes:
        for k in ("vertices", "colors", "normals"):
            out[k].append(m[k])
        out["triangles"].append(m["triangles"] + off)
        off += len(m["vertices"])
    return {k: np.vstack(v) for k, v in out.items()}


def test_nearest_surface_wins_and_background_stays():
    mesh = merge(quad(2.0, [1, 0, 0], half=0.6), quad(1.0, [0, 1, 0]))  # far red, near green
    img = MeshView(mesh).render(K, T, 100, 100)
    r, g, b = img[50, 50].astype(int)
    assert g > 200 and r < 30 and b < 30  # near green, facing the camera: full headlight
    r, g, b = img[50, 25].astype(int)
    assert r > 150 and g < 30  # only the far red quad there
    assert tuple(img[0, 0]) == (32, 32, 32)  # background


def test_surface_inside_a_change_box_is_tinted():
    mesh = quad(1.0, [0.5, 0.5, 0.5])
    box = {"center": [0.0, 0.0, 1.0], "extents": [0.1, 0.1, 0.1], "yaw_rad": 0.0}
    img = MeshView(mesh).render(K, T, 100, 100, tints=[(box, [0, 200, 0])])
    assert img[50, 50, 1] > img[50, 50, 0] + 50  # tinted green inside the box
    assert abs(int(img[50, 20, 1]) - int(img[50, 20, 0])) < 5  # untouched outside it
    assert inside_box(box, np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]])).tolist() == [True, False]


def test_vertical_structure_marks_walls_not_a_tilted_floor():
    from src.navigation import Grid2D
    from src.viz import vertical_structure

    xs, ys = np.meshgrid(np.arange(0, 2, 0.02), np.arange(0, 2, 0.02))
    floor = np.c_[xs.ravel(), ys.ravel(), 0.3 + 0.4 * xs.ravel()]  # tilted 22 deg, offset from z = 0
    zs = np.arange(0, 1.0, 0.02)
    wall = np.array([[1.0, y, z] for y in np.arange(0, 2, 0.02) for z in zs])  # wall at x = 1
    grid = Grid2D(0.0, 0.0, 40, 40, 0.05)
    v = vertical_structure(np.vstack([floor, wall]), grid, 0.15, 3)
    assert v[20, :].all()  # the wall column
    assert not v[5, :].any() and not v[35, :].any()  # floor on both sides


def test_simplify_mesh_keeps_colours_and_respects_the_budget():
    from src.viz import simplify_mesh

    n = 60
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    v = np.c_[xs.ravel(), ys.ravel(), np.zeros(n * n)]
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    tris = np.vstack([np.c_[a, b, c], np.c_[a, c, d]])
    mesh = {
        "vertices": v,
        "triangles": tris,
        "colors": np.tile([0.2, 0.6, 0.9], (n * n, 1)),
        "normals": np.tile([0, 0, 1.0], (n * n, 1)),
    }
    small = simplify_mesh(mesh, 500)
    assert len(small["triangles"]) <= 500
    assert np.allclose(small["colors"], [0.2, 0.6, 0.9], atol=1e-6)
    assert simplify_mesh(mesh, 10**6) is mesh  # under budget: untouched
