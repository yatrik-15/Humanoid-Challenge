"""GT annotation helper: DLT triangulation round trip with known cameras."""

import numpy as np

from eval.annotate import reprojection_errors, triangulate
from src.utils.geometry import look_at_T_cw, project_points


def test_triangulation_round_trip():
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    X = np.array([0.3, 1.7, 0.4])
    Ts = [look_at_T_cw(np.array([x, -1.0, 1.2]), X, np.array([0, 0, 1.0])) for x in (-0.8, 0.1, 0.9)]
    uvs = [project_points(X[None], K, T)[0][0] for T in Ts]
    Xh = triangulate([K] * 3, Ts, uvs)
    assert np.allclose(Xh, X, atol=1e-6)
    assert max(reprojection_errors(Xh, [K] * 3, Ts, uvs)) < 1e-6
    # a 3-px click error must stay a small 3-D error at ~3 m distance
    noisy = [uv + np.array([3.0, -2.0]) * s for uv, s in zip(uvs, (1, -1, 1))]
    assert np.linalg.norm(triangulate([K] * 3, Ts, noisy) - X) < 0.05
