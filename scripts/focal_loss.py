"""Multiclass focal loss as a LightGBM custom objective (v4 L2).
Reference: maxhalford.github.io/blog/lightgbm-focal-loss/. CPU; data is small.
LightGBM lays multiclass raw scores out group-by-class (length n*num_class,
reshape(num_class, n).T -> (n, num_class)); grad/hess return the same layout."""
from __future__ import annotations

import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def multiclass_focal_objective(num_class: int, gamma: float = 2.0):
    def _obj(raw, dataset):
        y = dataset.get_label().astype(int)
        n = y.shape[0]
        p = softmax(raw.reshape(num_class, n).T)       # (n, num_class)
        onehot = np.eye(num_class)[y]
        pt = (p * onehot).sum(axis=1, keepdims=True)    # prob of true class, (n,1)
        mod = (1.0 - pt) ** gamma                       # focal modulating factor
        grad = mod * (p - onehot)                       # (n, num_class)
        # Hessian uses the STANDARD softmax second derivative p*(1-p), NOT scaled
        # by the focal modulating factor. Scaling the hessian by `mod` shrinks leaf
        # step sizes so much that trees stop splitting ("no positive gain") and the
        # model underfits to near-prior (degenerate argmax, macro-F1 ~0.04). Keeping
        # the stable softmax hessian lets the focal-modulated gradient steer growth
        # while the trees actually grow. At gamma=0 grad reduces to p-onehot (CE).
        hess = p * (1.0 - p)
        hess = np.maximum(hess, 1e-6)
        return grad.T.reshape(-1), hess.T.reshape(-1)
    return _obj
