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
