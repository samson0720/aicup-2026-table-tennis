import numpy as np
import pandas as pd

from scripts.train_catboost_baseline import (
    cat_feature_indices,
    prepare_x,
    fit_multiclass,
)


def test_cat_feature_indices_splits_continuous_from_categorical():
    cols = ["prefix_len", "scoreSelf", "strikeId", "handId", "act_cnt_3", "spin_entropy"]
    idx = cat_feature_indices(cols)
    # continuous + *_cnt_* + *_entropy excluded; only strikeId, handId remain
    assert [cols[i] for i in idx] == ["strikeId", "handId"]


def test_fit_multiclass_returns_aligned_proba():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.integers(0, 5, 200), "b": rng.integers(0, 3, 200)})
    y = pd.Series(rng.integers(0, 10, 200))
    cat_idx = [0, 1]
    xs = prepare_x(x, ["a", "b"])
    p = fit_multiclass(xs, y, xs.iloc[:20], list(range(10)), cat_idx, "sqrt", 0, 20)
    assert p.shape == (20, 10)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-3)
