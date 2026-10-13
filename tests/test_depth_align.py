"""Tests for per-frame inverse-depth affine alignment (no model download needed)."""

import numpy as np
import pytest

from src.depth_align import apply_fit, fit_inverse_depth_affine

CFG = {
    "min_sparse_points": 20,
    "ransac_inlier_threshold": 0.1,
    "ransac_iterations": 500,
    "min_inlier_ratio": 0.4,
    "max_residual": 0.15,
}


def make(rng, n=300, a=0.8, b=0.05, noise=0.01, outlier_frac=0.0):
    z = rng.uniform(0.5, 4.0, n)
    pred = (1.0 / z - b) / a
    pred = pred * (1 + rng.normal(0, noise, n))
    n_out = int(outlier_frac * n)
    pred[:n_out] = rng.uniform(pred.min(), pred.max(), n_out)
    return pred, z


def test_recovers_affine_with_outliers():
    rng = np.random.default_rng(0)
    pred, z = make(rng, outlier_frac=0.3)
    fit = fit_inverse_depth_affine(pred, z, CFG, rng)
    assert fit.ok, fit.reason
    assert fit.a == pytest.approx(0.8, rel=0.03)
    assert fit.b == pytest.approx(0.05, abs=0.01)
    assert 0.6 < fit.inlier_ratio < 0.8


def test_rejects_too_few_points():
    rng = np.random.default_rng(1)
    pred, z = make(rng, n=10)
    fit = fit_inverse_depth_affine(pred, z, CFG, rng)
    assert not fit.ok and "too few" in fit.reason


def test_rejects_garbage_prediction():
    rng = np.random.default_rng(2)
    z = rng.uniform(0.5, 4.0, 300)
    pred = rng.uniform(0, 1, 300)  # unrelated to depth
    fit = fit_inverse_depth_affine(pred, z, CFG, rng)
    assert not fit.ok


def test_apply_fit_invalid_and_range():
    rng = np.random.default_rng(3)
    pred, z = make(rng, noise=0.0)
    fit = fit_inverse_depth_affine(pred, z, CFG, rng)
    d = apply_fit(pred.reshape(20, 15), fit, depth_max=3.0)
    zz = z.reshape(20, 15)
    np.testing.assert_allclose(d[zz < 2.9], zz[zz < 2.9], rtol=1e-4)
    assert np.all(d[zz > 3.1] == 0)
