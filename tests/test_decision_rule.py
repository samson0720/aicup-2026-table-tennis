# tests/test_decision_rule.py
import numpy as np
from scripts.decision_rule import AdditiveThreshold, RULES


def _toy(n=400, n_cls=3, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(n_cls), size=n)
    y = probs.argmax(1)
    prior = np.bincount(y, minlength=n_cls).astype(float)
    prior /= prior.sum()
    return probs, y, prior, n_cls


def test_additive_baseline_recovers_known_pipeline():
    # Baseline = prior_correct + tune_thresholds(default grid, 2 passes).
    from scripts.postprocess import prior_correct, tune_thresholds, apply_thresholds
    probs, y, prior, n_cls = _toy()
    rule = AdditiveThreshold()  # defaults == current production rule
    rule.fit(probs, y, n_cls, prior)
    got = rule.predict(probs, n_cls, prior)
    corrected = prior_correct(probs, prior)
    thr = tune_thresholds(corrected, y, n_cls)
    expected = apply_thresholds(corrected, thr)
    assert np.array_equal(got, expected)


def test_wide_additive_is_registered_and_runs():
    probs, y, prior, n_cls = _toy()
    rule = RULES["additive_wide"]()
    rule.fit(probs, y, n_cls, prior)
    pred = rule.predict(probs, n_cls, prior)
    assert pred.shape == (probs.shape[0],)
    assert set(np.unique(pred)).issubset(set(range(n_cls)))


def test_registry_contains_all_four_rules():
    assert set(RULES) == {"additive_baseline", "additive_wide", "calibrated", "weighted"}


def test_calibrated_rule_outputs_valid_labels_and_is_fit_dependent():
    from scripts.decision_rule import CalibratedThreshold
    rng = np.random.default_rng(3)
    n, n_cls = 600, 4
    probs = rng.dirichlet(np.ones(n_cls), size=n)
    # make class 0 well-separated so isotonic has signal to calibrate
    y = (rng.random(n) < probs[:, 0]).astype(int)
    y = np.where(y == 1, 0, rng.integers(1, n_cls, size=n))
    prior = np.bincount(y, minlength=n_cls).astype(float)
    prior /= prior.sum()
    rule = CalibratedThreshold().fit(probs, y, n_cls, prior)
    pred = rule.predict(probs, n_cls, prior)
    assert pred.shape == (n,)
    assert set(np.unique(pred)).issubset(set(range(n_cls)))
    # an unfit rule must raise rather than silently mispredict
    import pytest
    with pytest.raises(Exception):
        CalibratedThreshold().predict(probs, n_cls, prior)


def test_weighted_rule_beats_plain_argmax_on_imbalanced_toy():
    from scripts.decision_rule import WeightedArgmax
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(7)
    n, n_cls = 900, 3
    # heavy class imbalance: class 0 dominates, plain argmax ignores 1 and 2
    y = rng.choice(n_cls, size=n, p=[0.8, 0.13, 0.07])
    probs = np.full((n, n_cls), 0.05)
    probs[np.arange(n), y] += rng.uniform(0.3, 0.7, size=n)
    probs /= probs.sum(1, keepdims=True)
    prior = np.bincount(y, minlength=n_cls).astype(float)
    prior /= prior.sum()
    rule = WeightedArgmax().fit(probs, y, n_cls, prior)
    pred = rule.predict(probs, n_cls, prior)
    plain = probs.argmax(1)
    f_rule = f1_score(y, pred, labels=range(n_cls), average="macro", zero_division=0)
    f_plain = f1_score(y, plain, labels=range(n_cls), average="macro", zero_division=0)
    assert f_rule >= f_plain
    assert set(np.unique(pred)).issubset(set(range(n_cls)))


def test_registry_now_has_weighted():
    from scripts.decision_rule import RULES
    assert set(RULES) == {"additive_baseline", "additive_wide", "calibrated", "weighted"}
