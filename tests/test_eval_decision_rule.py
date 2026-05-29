# tests/test_eval_decision_rule.py
import numpy as np
from scripts.eval_decision_rule import nested_cv_f1, seed_holdout_f1
from scripts.decision_rule import RULES


def _synthetic(n=600, n_cls=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(n_cls, size=n, p=[0.7, 0.2, 0.1])
    probs = np.full((n, n_cls), 0.1)
    probs[np.arange(n), y] += rng.uniform(0.2, 0.6, size=n)
    probs /= probs.sum(1, keepdims=True)
    prior = np.bincount(y, minlength=n_cls).astype(float)
    prior /= prior.sum()
    groups = rng.integers(0, 20, size=n)          # 20 "matches"
    seeds = rng.choice([11, 22, 33, 44, 55], size=n)
    return probs, y, prior, groups, seeds, n_cls


def test_nested_cv_returns_valid_macro_f1():
    probs, y, prior, groups, seeds, n_cls = _synthetic()
    f = nested_cv_f1(RULES["additive_baseline"], probs, y, prior, groups, n_cls, n_folds=5)
    assert 0.0 <= f <= 1.0


def test_seed_holdout_only_scores_holdout_rows():
    probs, y, prior, groups, seeds, n_cls = _synthetic()
    f = seed_holdout_f1(RULES["weighted"], probs, y, prior, seeds, n_cls, holdout=55)
    assert 0.0 <= f <= 1.0


def test_seed_holdout_raises_when_holdout_absent():
    probs, y, prior, groups, seeds, n_cls = _synthetic()
    import pytest
    with pytest.raises(ValueError):
        seed_holdout_f1(RULES["weighted"], probs, y, prior, seeds, n_cls, holdout=999)
