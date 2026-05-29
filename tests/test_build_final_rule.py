# tests/test_build_final_rule.py
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from scripts.build_final_perrow import _nested_f1
from scripts.decision_rule import RULES
from scripts.postprocess import apply_thresholds, prior_correct, tune_thresholds


def test_nested_f1_baseline_matches_legacy_pipeline():
    """The refactored _nested_f1 with the additive_baseline rule must reproduce the
    legacy pipeline (full-y prior_correct + nested threshold tuning) exactly."""
    rng = np.random.default_rng(0)
    n, n_cls = 500, 3
    stk = rng.dirichlet(np.ones(n_cls), size=n)
    y = stk.argmax(1)
    groups = rng.integers(0, 15, size=n)

    got = _nested_f1(stk, y, groups, n_cls, rule_factory=RULES["additive_baseline"])

    full = np.bincount(y, minlength=n_cls).astype(float)
    full /= full.sum()
    corrected = prior_correct(stk, full)
    yhat = np.zeros(n, dtype=int)
    for tr, va in GroupKFold(n_splits=5).split(corrected, y, groups):
        thr = tune_thresholds(corrected[tr], y[tr], n_cls)
        yhat[va] = apply_thresholds(corrected[va], thr)
    legacy = f1_score(y, yhat, labels=list(range(n_cls)), average="macro", zero_division=0)

    assert abs(got - legacy) < 1e-9


def test_nested_f1_default_uses_prod_rule():
    rng = np.random.default_rng(1)
    n, n_cls = 400, 3
    stk = rng.dirichlet(np.ones(n_cls), size=n)
    y = stk.argmax(1)
    groups = rng.integers(0, 12, size=n)
    f = _nested_f1(stk, y, groups, n_cls)  # no rule_factory => PROD_RULE
    assert 0.0 <= f <= 1.0
