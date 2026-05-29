# tests/test_postprocess_grid.py
import numpy as np
from scripts.postprocess import tune_thresholds


def test_default_behaviour_unchanged():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=200)
    y = probs.argmax(1)
    thr = tune_thresholds(probs, y, 3)
    # default grid is bounded to [-0.10, 0.10]
    assert thr.shape == (3,)
    assert thr.min() >= -0.10 - 1e-9 and thr.max() <= 0.10 + 1e-9


def test_wide_grid_and_passes_accepted():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), size=300)
    y = (probs[:, 0] > 0.3).astype(int)  # imbalanced
    wide = tuple(np.round(np.arange(-0.30, 0.31, 0.02), 2))
    thr = tune_thresholds(probs, y, 4, grid=wide, max_passes=10)
    assert thr.shape == (4,)
    # the wider grid must be able to move a threshold beyond the old +/-0.10 cap
    assert thr.min() >= -0.30 - 1e-9 and thr.max() <= 0.30 + 1e-9
