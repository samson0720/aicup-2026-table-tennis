import numpy as np
import pandas as pd

from scripts.postprocess import (
    prior_correct,
    tune_thresholds,
    apply_thresholds,
    phase_blend_server,
    build_server_pair_prior,
)


def test_prior_correct_amplifies_rare_class():
    # 3 classes, prior heavily skewed toward class 0.
    probs = np.array([[0.50, 0.30, 0.20],
                      [0.50, 0.30, 0.20],
                      [0.40, 0.35, 0.25]])
    prior = np.array([0.80, 0.15, 0.05])
    corrected = prior_correct(probs, prior)
    # argmax(probs) is 0; argmax(corrected) should NOT be 0 for at least one row.
    assert (corrected.argmax(1) != 0).any()
    # Output rows still sum to ~1.
    assert np.allclose(corrected.sum(1), 1.0, atol=1e-6)


def test_prior_correct_uniform_prior_is_noop():
    probs = np.random.default_rng(0).dirichlet([1, 1, 1, 1], size=20)
    uniform = np.full(4, 0.25)
    out = prior_correct(probs, uniform)
    assert np.allclose(out.argmax(1), probs.argmax(1))


def test_tune_thresholds_does_not_make_macro_f1_worse():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    n, k = 500, 5
    y = rng.integers(0, k, size=n)
    # Probabilities biased toward class 0.
    base = rng.dirichlet(np.full(k, 0.5), size=n)
    base[:, 0] += 0.3
    base = base / base.sum(1, keepdims=True)

    f1_argmax = f1_score(y, base.argmax(1), labels=list(range(k)),
                         average="macro", zero_division=0)
    thr = tune_thresholds(base, y, n_classes=k)
    yhat = apply_thresholds(base, thr)
    f1_tuned = f1_score(y, yhat, labels=list(range(k)),
                        average="macro", zero_division=0)
    assert f1_tuned + 1e-9 >= f1_argmax


def test_phase_blend_server_weights_per_phase():
    n = 30
    p_model = np.full(n, 0.5)
    p_prior = np.array([0.9] * 10 + [0.5] * 10 + [0.1] * 10)
    phase = np.array([0] * 10 + [1] * 10 + [2] * 10)
    weights = {0: 0.7, 1: 0.4, 2: 0.0}
    out = phase_blend_server(p_model, p_prior, phase, weights)
    assert np.allclose(out[:10], 0.7 * 0.9 + 0.3 * 0.5)
    assert np.allclose(out[10:20], 0.4 * 0.5 + 0.6 * 0.5)
    assert np.allclose(out[20:], 0.5)


def test_build_server_pair_prior_unseen_falls_back_to_global():
    # Train has two pairs; valid has one of those + one unseen pair.
    train = pd.DataFrame({
        "rally_uid": [1, 1, 2, 2, 3, 3, 4, 4],
        "strikeNumber": [1, 2, 1, 2, 1, 2, 1, 2],
        "gamePlayerId":      [10, 11, 10, 11, 20, 21, 10, 11],
        "gamePlayerOtherId": [11, 10, 11, 10, 21, 20, 11, 10],
        "serverGetPoint": [1, 1, 0, 0, 1, 1, 1, 1],
    })
    valid = pd.DataFrame({
        "rally_uid": [5, 6],
        "strikeNumber": [1, 1],
        "gamePlayerId":      [10, 99],  # 10-11 seen, 99-98 unseen
        "gamePlayerOtherId": [11, 98],
        "serverGetPoint": [0, 0],
    })
    out = build_server_pair_prior(train, valid, alpha=20.0)
    assert set(out.index) == {5, 6}
    # Unseen pair must equal global rate (== 0.75 -- 3 wins of 4 first-stroke rows).
    assert abs(out.loc[6] - 0.75) < 1e-9
    # Seen pair is smoothed toward 0.75: rate = (2 + 20*0.75)/(3 + 20) = 17/23
    expected = (2 + 20 * 0.75) / (3 + 20)
    assert abs(out.loc[5] - expected) < 1e-9
