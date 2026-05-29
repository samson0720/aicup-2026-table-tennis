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


def multiclass_focal_objective(num_class: int, gamma: float = 2.0, hess_floor: float = 0.1):
    def _obj(raw, dataset):
        y = dataset.get_label().astype(int)
        n = y.shape[0]
        p = softmax(raw.reshape(num_class, n).T)       # (n, num_class)
        onehot = np.eye(num_class)[y]
        pt = (p * onehot).sum(axis=1, keepdims=True)    # prob of true class, (n,1)
        mod = (1.0 - pt) ** gamma                       # focal modulating factor
        grad = mod * (p - onehot)                       # (n, num_class)
        # Hessian: the STANDARD softmax second derivative p*(1-p), floored at
        # `hess_floor` (NOT scaled by the focal factor). Two failure modes this
        # avoids: (a) scaling hess by `mod` shrinks leaf steps so trees stop
        # splitting -> underfit to near-prior (macro-F1 ~0.04); (b) the bare
        # p*(1-p) -> 0 as probs saturate, so leaf = -grad/hess explodes (raw
        # margins ~5e5, fully-saturated one-hot probs, macro-F1 ~0.14). A floor of
        # ~0.1 caps leaf step sizes, taming the runaway: raw_std ~7, macro-F1
        # ~0.22 (single fold). At gamma=0 grad reduces to p-onehot (CE).
        hess = np.maximum(p * (1.0 - p), hess_floor)
        return grad.T.reshape(-1), hess.T.reshape(-1)
    return _obj
