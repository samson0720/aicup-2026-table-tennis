import numpy as np
import pandas as pd
import pytest

from scripts.stacker import stacked_oof


def test_stacker_no_rally_leakage():
    rng = np.random.default_rng(0)
    n_rallies = 200
    rally = np.repeat(np.arange(n_rallies), 1)
    match = (rally // 20).astype(int)  # 10 matches
    y = rng.integers(0, 3, size=n_rallies)
    probs1 = rng.dirichlet([1.0, 1.0, 1.0], size=n_rallies)
    probs2 = rng.dirichlet([1.0, 1.0, 1.0], size=n_rallies)
    base_oofs = {
        "m1": pd.DataFrame({"rally_uid": rally, **{f"p_{i}": probs1[:, i] for i in range(3)}}),
        "m2": pd.DataFrame({"rally_uid": rally, **{f"p_{i}": probs2[:, i] for i in range(3)}}),
    }
    labels = pd.DataFrame({"rally_uid": rally, "match": match, "y": y})
    out = stacked_oof(base_oofs, labels, target_kind="multiclass", n_classes=3, n_folds=5)
    assert len(out) == n_rallies
    assert out[[f"p_{i}" for i in range(3)]].sum(axis=1).between(0.99, 1.01).all()


def test_stacker_binary():
    rng = np.random.default_rng(1)
    n = 200
    rally = np.arange(n)
    match = (rally // 20).astype(int)
    y = rng.integers(0, 2, n)
    p1 = rng.random(n)
    p2 = rng.random(n)
    base = {
        "m1": pd.DataFrame({"rally_uid": rally, "p_1": p1}),
        "m2": pd.DataFrame({"rally_uid": rally, "p_1": p2}),
    }
    labels = pd.DataFrame({"rally_uid": rally, "match": match, "y": y})
    out = stacked_oof(base, labels, target_kind="binary", n_classes=1, n_folds=5)
    assert len(out) == n
    assert "p_1" in out.columns
    assert out["p_1"].between(0.0, 1.0).all()
