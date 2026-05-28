import numpy as np

from scripts.seq_eval import honest_scores, monitor_score, warmup_cosine_lambda


def test_warmup_cosine_lambda_shape():
    fn = warmup_cosine_lambda(warmup_steps=10, total_steps=110)
    assert fn(0) < fn(9)
    assert abs(fn(10) - 1.0) < 1e-6
    assert fn(110) < 0.01


def _perfect_probs(y: np.ndarray, n_cls: int) -> np.ndarray:
    p = np.full((len(y), n_cls), 0.01)
    p[np.arange(len(y)), y] = 0.99
    return p


def test_honest_scores_perfect_predictions():
    rng = np.random.default_rng(0)
    ya = rng.integers(0, 19, 200)
    yp = rng.integers(0, 10, 200)
    ys = rng.integers(0, 2, 200)
    groups = rng.integers(0, 10, 200)
    s = honest_scores(
        _perfect_probs(ya, 19),
        _perfect_probs(yp, 10),
        ys * 0.99 + (1 - ys) * 0.01,
        ya,
        yp,
        ys,
        groups,
    )
    assert s["action_f1"] > 0.95
    assert s["point_f1"] > 0.95
    assert s["server_auc"] > 0.99
    assert 0.0 <= s["overall"] <= 1.0


def test_monitor_score_keys_and_value():
    rng = np.random.default_rng(1)
    ya = rng.integers(0, 19, 50)
    yp = rng.integers(0, 10, 50)
    ys = rng.integers(0, 2, 50)
    m = monitor_score(
        _perfect_probs(ya, 19),
        _perfect_probs(yp, 10),
        ys * 0.9 + 0.05,
        ya,
        yp,
        ys,
    )
    assert set(m) == {"action_f1", "point_f1", "server_auc", "overall"}
    assert m["action_f1"] > 0.9
