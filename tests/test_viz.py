"""Point-splat renderer used for the 3-D overview."""

import numpy as np

from src.utils.geometry import look_at_T_cw
from src.viz import render_points


def test_nearest_point_wins_and_points_behind_are_dropped():
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    T = look_at_T_cw(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]))
    pts = np.array([[0, 0, 2.0], [0, 0, 1.0], [0, 0, -1.0]])  # far red, near green, behind blue
    cols = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    img = render_points(pts, cols, K, T, 100, 100, voxel=0.02)
    assert tuple(img[50, 50]) == (0, 255, 0)
    assert not (img[..., 2] == 255).any()  # the point behind the camera is not drawn
    assert tuple(img[0, 0]) == (32, 32, 32)  # background
