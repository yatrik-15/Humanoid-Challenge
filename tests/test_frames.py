"""Blur filter: relative to neighbouring frames, so low-texture scenes are not discarded."""

import numpy as np

from src.frames import blur_keep_mask


def test_low_texture_video_is_kept():
    scores = np.array([50, 57, 53, 45, 48, 60, 59, 50, 43], float)  # sharp plain floor
    assert blur_keep_mask(scores, 0.5, 7).all()


def test_motion_blur_dip_is_dropped_and_texture_change_is_not():
    scores = np.array([70, 100, 130, 175, 20, 300, 350, 400, 410], float)
    keep = blur_keep_mask(scores, 0.5, 7)
    assert not keep[4]
    assert keep[[0, 1, 2, 3, 5, 6, 7, 8]].all()
