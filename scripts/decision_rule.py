"""Pluggable macro-F1 decision rules over stacked OOF probabilities.

Every rule maps a raw stacked-probability matrix (rows sum to 1) plus the
empirical class prior to integer labels. The baseline reproduces the current
production rule exactly; the others are P1 candidates.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score

from scripts.postprocess import apply_thresholds, prior_correct, tune_thresholds

# Wider, finer grid than the production default (±0.10, 9 points).
WIDE_GRID = tuple(round(v, 2) for v in np.arange(-0.30, 0.301, 0.02))

# Multiplicative per-class weight grid (log-spaced around 1.0).
WEIGHT_GRID = tuple(round(float(v), 3) for v in np.geomspace(0.5, 2.0, 9))


class AdditiveThreshold:
    """prior_correct -> additive per-class thresholds (argmax of prob - thr).

    Defaults reproduce the production rule (±0.10 grid, 2 passes). P1a uses the
    wide grid run to convergence.
    """

    def __init__(self, grid=None, max_passes: int = 2, beta: float = 1.0):
        self.grid = grid  # None => tune_thresholds default
        self.max_passes = max_passes
        self.beta = beta  # prior-correction temperature (1.0 = legacy full correction)
        self.thr = None

    def fit(self, probs, y, n_cls, prior):
        corrected = prior_correct(probs, prior, self.beta)
        kwargs = {"max_passes": self.max_passes}
        if self.grid is not None:
            kwargs["grid"] = self.grid
        self.thr = tune_thresholds(corrected, y, n_cls, **kwargs)
        return self

    def predict(self, probs, n_cls, prior):
        if self.thr is None:
            raise RuntimeError("AdditiveThreshold.predict called before fit")
        corrected = prior_correct(probs, prior, self.beta)
        return apply_thresholds(corrected, self.thr)


class CalibratedThreshold:
    """Per-class isotonic calibration of each class's probability column, then
    renormalize and tune additive thresholds on the calibrated matrix.

    Replaces the crude prior_correct with a monotone calibrator fit to the
    one-vs-rest class indicator. `prior` is accepted for interface uniformity
    but unused (calibration subsumes it).
    """

    def __init__(self, grid=WIDE_GRID, max_passes: int = 10):
        self.grid = grid
        self.max_passes = max_passes
        self.iso = None
        self.thr = None

    def _calibrate(self, probs, n_cls):
        cal = np.column_stack([
            self.iso[c].predict(probs[:, c]) for c in range(n_cls)
        ])
        denom = cal.sum(axis=1, keepdims=True)
        return cal / np.clip(denom, 1e-12, None)

    def fit(self, probs, y, n_cls, prior):
        self.iso = []
        for c in range(n_cls):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(probs[:, c], (y == c).astype(float))
            self.iso.append(ir)
        cal = self._calibrate(probs, n_cls)
        self.thr = tune_thresholds(cal, y, n_cls, grid=self.grid, max_passes=self.max_passes)
        return self

    def predict(self, probs, n_cls, prior):
        if self.iso is None or self.thr is None:
            raise RuntimeError("CalibratedThreshold.predict called before fit")
        cal = self._calibrate(probs, n_cls)
        return apply_thresholds(cal, self.thr)


def tune_weights(corrected, y, n_cls, grid=WEIGHT_GRID, max_passes=10):
    """Coordinate ascent on per-class multiplicative weights to maximize macro-F1.

    Prediction is argmax_c (w_c * corrected[:, c]). Distinct mechanism from the
    additive rule: rescales class scales rather than shifting them.
    """
    w = np.ones(n_cls)

    def score(weights):
        yhat = (corrected * weights[None, :]).argmax(1)
        return f1_score(y, yhat, labels=list(range(n_cls)),
                        average="macro", zero_division=0)

    best_global = score(w)
    for _ in range(max_passes):
        moved = False
        for c in range(n_cls):
            best_local, best_v = best_global, w[c]
            for v in grid:
                trial = w.copy()
                trial[c] = v
                s = score(trial)
                if s > best_local + 1e-9:
                    best_local, best_v = s, v
            if best_v != w[c]:
                w[c] = best_v
                best_global = best_local
                moved = True
        if not moved:
            break
    return w


class WeightedArgmax:
    """prior_correct -> argmax of (w_c * corrected prob), w tuned for macro-F1."""

    def __init__(self, grid=WEIGHT_GRID, max_passes: int = 10):
        self.grid = grid
        self.max_passes = max_passes
        self.w = None

    def fit(self, probs, y, n_cls, prior):
        corrected = prior_correct(probs, prior)
        self.w = tune_weights(corrected, y, n_cls, grid=self.grid, max_passes=self.max_passes)
        return self

    def predict(self, probs, n_cls, prior):
        if self.w is None:
            raise RuntimeError("WeightedArgmax.predict called before fit")
        corrected = prior_correct(probs, prior)
        return (corrected * self.w[None, :]).argmax(1)


class Argmax:
    """Pure argmax of the stacked (calibrated) probabilities — ACCURACY-optimal.

    No prior-correction, no threshold tuning. If the official metric is accuracy /
    micro-F1 (not macro-F1), this is correct; prior-correction shifts mass to rare
    classes and DESTROYS accuracy (~-0.16 action / -0.12 point measured).
    """

    def fit(self, probs, y, n_cls, prior):
        return self

    def predict(self, probs, n_cls, prior):
        return probs.argmax(1)


RULES = {
    "additive_baseline": lambda: AdditiveThreshold(),
    "additive_wide": lambda: AdditiveThreshold(grid=WIDE_GRID, max_passes=10),
    "calibrated": lambda: CalibratedThreshold(),
    "weighted": lambda: WeightedArgmax(),
    "argmax": lambda: Argmax(),
}
