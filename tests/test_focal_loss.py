import numpy as np
from scripts.focal_loss import softmax, multiclass_focal_objective

class _FakeDataset:
    def __init__(self, y): self._y = y
    def get_label(self): return self._y

def test_softmax_rows_sum_to_one():
    z = np.random.randn(5, 4)
    p = softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)

def test_focal_gamma_zero_matches_ce_gradient():
    # gamma=0 reduces to softmax cross-entropy: grad = p - onehot
    n, k = 6, 3
    rng = np.random.default_rng(0)
    # raw laid out group-by-class: length k*n, reshape(k, n).T -> (n, k)
    raw = rng.normal(size=k * n)
    y = rng.integers(0, k, size=n)
    obj = multiclass_focal_objective(num_class=k, gamma=0.0)
    grad, hess = obj(raw, _FakeDataset(y))
    p = softmax(raw.reshape(k, n).T)              # (n, k)
    onehot = np.eye(k)[y]
    assert np.allclose(grad.reshape(k, n).T, p - onehot, atol=1e-6)
    assert (hess > 0).all()
