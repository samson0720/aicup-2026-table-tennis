# P1 — macro-F1 Decision-Rule Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the truncated additive-threshold decision rule (grid ±0.10, 2 passes) with the best of three macro-F1-aware rules, gated on nested-CV **and** a seed-55 holdout, shipping only if it beats the current rule by > the 0.00168 noise floor.

**Architecture:** Make the decision rule a pluggable strategy operating on the per-row stacked OOF probabilities. Three candidate rules — wide-grid additive (P1a), per-class isotonic calibration + additive (P1b), multiplicative class-weight argmax (P1c) — are compared against the current rule as the baseline. A harness reuses the existing `build_final_perrow` stacking helpers, scores every rule with the honest nested-CV ruler and a held-out-seed kill switch, and only the winner (if any) is wired into the production builder.

**Tech Stack:** Python, NumPy, scikit-learn (LogisticRegression, IsotonicRegression, f1_score, GroupKFold), pandas, pytest. Conda env `aicup-tt` (`conda run -n aicup-tt ...`). No new dependencies → no clone env needed. No GPU needed (pure post-processing on cached probabilities).

**Key invariants (from the spec + PROGRESS):**
- Honest per-row scoring only; never seed-average.
- Noise floor overall 0.00168; action 0.00525; point 0.00506. P1 does NOT touch server (no threshold there), so overall lift = 0.4·(Δaction + Δpoint).
- Current production honest overall = **0.32568** (action 0.2963, point 0.1895, server 0.6567).
- A rule SHIPS only if it beats baseline by > 0.00168 overall on the nested-CV ruler **and** does not regress on the seed-55 holdout.

---

### Task 1: Parametrize `tune_thresholds` (grid + passes) without changing defaults

**Files:**
- Modify: `scripts/postprocess.py:25-66`
- Test: `tests/test_postprocess_grid.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_postprocess_grid.py
import numpy as np
from scripts.postprocess import tune_thresholds


def test_default_behaviour_unchanged():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=200)
    y = probs.argmax(1)
    thr = tune_thresholds(probs, y, 3)
    # default grid is bounded to [-0.10, 0.10]
    assert thr.shape == (3,)
    assert thr.min() >= -0.10 - 1e-9 and thr.max() <= 0.10 + 1e-9


def test_wide_grid_and_passes_accepted():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), size=300)
    y = (probs[:, 0] > 0.3).astype(int)  # imbalanced
    wide = tuple(np.round(np.arange(-0.30, 0.31, 0.02), 2))
    thr = tune_thresholds(probs, y, 4, grid=wide, max_passes=10)
    assert thr.shape == (4,)
    # the wider grid must be able to move a threshold beyond the old +/-0.10 cap
    assert thr.min() >= -0.30 - 1e-9 and thr.max() <= 0.30 + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aicup-tt pytest tests/test_postprocess_grid.py -v`
Expected: FAIL — `tune_thresholds() got an unexpected keyword argument 'max_passes'`.

- [ ] **Step 3: Add the `max_passes` parameter (default 2 = current behaviour)**

In `scripts/postprocess.py`, change the `tune_thresholds` signature and the loop:

```python
def tune_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    grid: tuple[float, ...] = (-0.10, -0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06, 0.10),
    max_passes: int = 2,
) -> np.ndarray:
    """Per-class additive threshold grid search to maximize macro-F1.

    Greedy coordinate ascent: for each class c, fix others and pick the best `v`
    from `grid`. Repeat up to `max_passes` times. Stops early when no class moves.
    Defaults reproduce the original (±0.10 grid, 2 passes).
    """
    from sklearn.metrics import f1_score

    thr = np.zeros(n_classes)

    def score(t: np.ndarray) -> float:
        yhat = apply_thresholds(probs, t)
        return f1_score(
            y, yhat, labels=list(range(n_classes)),
            average="macro", zero_division=0,
        )

    best_global = score(thr)
    for _ in range(max_passes):
        moved = False
        for c in range(n_classes):
            best_local = best_global
            best_v = thr[c]
            for v in grid:
                trial = thr.copy()
                trial[c] = v
                s = score(trial)
                if s > best_local + 1e-9:
                    best_local = s
                    best_v = v
            if best_v != thr[c]:
                thr[c] = best_v
                best_global = best_local
                moved = True
        if not moved:
            break
    return thr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aicup-tt pytest tests/test_postprocess_grid.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Confirm nothing else broke (existing suite)**

Run: `conda run -n aicup-tt pytest -q`
Expected: all previously-green tests still pass (the new param is optional).

- [ ] **Step 6: Commit**

```bash
git add scripts/postprocess.py tests/test_postprocess_grid.py
git commit -m "feat(postprocess): parametrize tune_thresholds grid + max_passes (defaults unchanged)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Decision-rule strategies — base interface + additive (baseline + P1a wide)

**Files:**
- Create: `scripts/decision_rule.py`
- Test: `tests/test_decision_rule.py`

The interface every rule implements:
- `fit(self, probs, y, n_cls, prior) -> self` — `probs` are RAW stacked OOF probabilities (rows sum to 1), `prior` is the empirical class prior (1-D, length `n_cls`).
- `predict(self, probs, n_cls, prior) -> np.ndarray[int]` — predicted labels.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aicup-tt pytest tests/test_decision_rule.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.decision_rule'`.

- [ ] **Step 3: Create `scripts/decision_rule.py` with the base + additive rules**

```python
"""Pluggable macro-F1 decision rules over stacked OOF probabilities.

Every rule maps a raw stacked-probability matrix (rows sum to 1) plus the
empirical class prior to integer labels. The baseline reproduces the current
production rule exactly; the others are P1 candidates.
"""
from __future__ import annotations

import numpy as np

from scripts.postprocess import apply_thresholds, prior_correct, tune_thresholds

# Wider, finer grid than the production default (±0.10, 9 points).
WIDE_GRID = tuple(round(v, 2) for v in np.arange(-0.30, 0.301, 0.02))


class AdditiveThreshold:
    """prior_correct -> additive per-class thresholds (argmax of prob - thr).

    Defaults reproduce the production rule (±0.10 grid, 2 passes). P1a uses the
    wide grid run to convergence.
    """

    def __init__(self, grid=None, max_passes: int = 2):
        self.grid = grid  # None => tune_thresholds default
        self.max_passes = max_passes
        self.thr = None

    def fit(self, probs, y, n_cls, prior):
        corrected = prior_correct(probs, prior)
        kwargs = {"max_passes": self.max_passes}
        if self.grid is not None:
            kwargs["grid"] = self.grid
        self.thr = tune_thresholds(corrected, y, n_cls, **kwargs)
        return self

    def predict(self, probs, n_cls, prior):
        corrected = prior_correct(probs, prior)
        return apply_thresholds(corrected, self.thr)


RULES = {
    "additive_baseline": lambda: AdditiveThreshold(),
    "additive_wide": lambda: AdditiveThreshold(grid=WIDE_GRID, max_passes=10),
    # "calibrated" and "weighted" are registered in later tasks.
}
```

- [ ] **Step 4: Run test to verify the first two pass and the registry test still fails**

Run: `conda run -n aicup-tt pytest tests/test_decision_rule.py -v`
Expected: `test_additive_baseline_recovers_known_pipeline` PASS, `test_wide_additive_is_registered_and_runs` PASS, `test_registry_contains_all_four_rules` FAIL (only 2 keys yet). The registry test is completed in Task 4.

- [ ] **Step 5: Commit**

```bash
git add scripts/decision_rule.py tests/test_decision_rule.py
git commit -m "feat(decision_rule): base interface + additive baseline/wide rules

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: P1b — per-class isotonic calibration + additive thresholds

**Files:**
- Modify: `scripts/decision_rule.py`
- Test: `tests/test_decision_rule.py`

- [ ] **Step 1: Write the failing test (append)**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aicup-tt pytest tests/test_decision_rule.py::test_calibrated_rule_outputs_valid_labels_and_is_fit_dependent -v`
Expected: FAIL — `ImportError: cannot import name 'CalibratedThreshold'`.

- [ ] **Step 3: Implement `CalibratedThreshold` and register it**

Append to `scripts/decision_rule.py` (and add the import at the top):

```python
from sklearn.isotonic import IsotonicRegression


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
```

Update the `RULES` dict:

```python
RULES = {
    "additive_baseline": lambda: AdditiveThreshold(),
    "additive_wide": lambda: AdditiveThreshold(grid=WIDE_GRID, max_passes=10),
    "calibrated": lambda: CalibratedThreshold(),
    # "weighted" is registered in Task 4.
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aicup-tt pytest tests/test_decision_rule.py::test_calibrated_rule_outputs_valid_labels_and_is_fit_dependent -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/decision_rule.py tests/test_decision_rule.py
git commit -m "feat(decision_rule): P1b per-class isotonic calibration rule

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: P1c — multiplicative class-weight argmax (macro-F1 plug-in)

**Files:**
- Modify: `scripts/decision_rule.py`
- Test: `tests/test_decision_rule.py`

- [ ] **Step 1: Write the failing test (append)**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aicup-tt pytest tests/test_decision_rule.py -k "weighted or registry" -v`
Expected: FAIL — `ImportError: cannot import name 'WeightedArgmax'` and the registry test still missing `weighted`. (The earlier `test_registry_contains_all_four_rules` from Task 2 also turns green here.)

- [ ] **Step 3: Implement `tune_weights` + `WeightedArgmax`, register it**

Append to `scripts/decision_rule.py`:

```python
from sklearn.metrics import f1_score

# Multiplicative per-class weight grid (log-spaced around 1.0).
WEIGHT_GRID = tuple(round(float(v), 3) for v in np.geomspace(0.5, 2.0, 9))


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
```

Update `RULES` to its final form:

```python
RULES = {
    "additive_baseline": lambda: AdditiveThreshold(),
    "additive_wide": lambda: AdditiveThreshold(grid=WIDE_GRID, max_passes=10),
    "calibrated": lambda: CalibratedThreshold(),
    "weighted": lambda: WeightedArgmax(),
}
```

- [ ] **Step 4: Run the whole decision-rule suite**

Run: `conda run -n aicup-tt pytest tests/test_decision_rule.py -v`
Expected: all PASS (including both registry tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/decision_rule.py tests/test_decision_rule.py
git commit -m "feat(decision_rule): P1c multiplicative class-weight argmax + finalize registry

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Evaluation harness — nested-CV ruler + seed-55 holdout kill switch

**Files:**
- Create: `scripts/eval_decision_rule.py`
- Test: `tests/test_eval_decision_rule.py`

The harness, for `action` and `point`:
1. Builds the per-row stacked OOF once via the existing `build_final_perrow` helpers.
2. For every rule in `RULES`, computes two macro-F1 numbers:
   - **nested-CV** (`GroupKFold` by `match`): fit the rule on each train fold, predict the held-out fold — the honest ruler that produced 0.32568.
   - **seed holdout**: fit on rows with `seed != holdout_seed`, score on `seed == holdout_seed` (default holdout = max seed = 55) — the overfit kill switch.
3. Writes `artifacts/decision_rule_scores.json` and prints the gate verdict.

- [ ] **Step 1: Write the failing test (split logic on synthetic data)**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aicup-tt pytest tests/test_eval_decision_rule.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.eval_decision_rule'`.

- [ ] **Step 3: Implement the harness**

```python
"""Compare macro-F1 decision rules on the per-row stacked OOF (P1 gate).

Honest ruler = nested-CV macro-F1 (the same scheme that produced 0.32568).
Kill switch = seed-55 holdout macro-F1 (rules that only win in-sample regress
here, the hill-climb lesson).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from scripts.build_final_perrow import (
    BASES, KEYS, _perrow_features, _stack_oof, _data_dir,
)
from scripts.decision_rule import RULES
from scripts.score_oof import attach_labels, overall

# Targets P1 can affect (server has no threshold rule).
TARGETS = [("action", 19, "actionId"), ("point", 10, "pointId")]
SERVER_AUC_BASELINE = 0.6567  # production server AUC; P1 leaves it untouched.


def _empirical_prior(y, n_cls):
    p = np.bincount(y, minlength=n_cls).astype(float)
    return p / p.sum()


def nested_cv_f1(rule_factory, probs, y, prior, groups, n_cls, n_folds=5):
    yhat = np.zeros(len(y), dtype=int)
    for tr, va in GroupKFold(n_splits=n_folds).split(probs, y, groups):
        rule = rule_factory().fit(probs[tr], y[tr], n_cls, _empirical_prior(y[tr], n_cls))
        yhat[va] = rule.predict(probs[va], n_cls, prior)
    return float(f1_score(y, yhat, labels=list(range(n_cls)),
                          average="macro", zero_division=0))


def seed_holdout_f1(rule_factory, probs, y, prior, seeds, n_cls, holdout=55):
    seeds = np.asarray(seeds)
    if not (seeds == holdout).any():
        raise ValueError(f"holdout seed {holdout} not present in seeds")
    tr = seeds != holdout
    va = seeds == holdout
    rule = rule_factory().fit(probs[tr], y[tr], n_cls, _empirical_prior(y[tr], n_cls))
    yhat = rule.predict(probs[va], n_cls, prior)
    return float(f1_score(y[va], yhat, labels=list(range(n_cls)),
                          average="macro", zero_division=0))


def _build_stack(target, n_cls):
    """Per-row stacked OOF probabilities + aligned labels/groups/seeds."""
    train = pd.read_csv(_data_dir() / "train.csv")
    match_per_rally = train.drop_duplicates("rally_uid").set_index("rally_uid")["match"]
    frame, feat_cols = _perrow_features(target, n_cls)
    y_col = {"action": "actionId", "point": "pointId"}[target]
    lab = attach_labels(frame[KEYS].copy(), train)[KEYS + [y_col]]
    frame = frame.merge(lab, on=KEYS, how="left")
    frame["match"] = frame["rally_uid"].map(match_per_rally)
    frame = frame.dropna(subset=[y_col, "match"]).reset_index(drop=True)
    X = frame[feat_cols].fillna(0.0).to_numpy()
    y = frame[y_col].astype(int).to_numpy()
    groups = frame["match"].to_numpy()
    seeds = frame["seed"].to_numpy()
    stk = _stack_oof(X, y, groups, "multiclass", n_cls)
    return stk, y, groups, seeds


def main(holdout=55):
    result = {}
    per_target = {}
    for target, n_cls, _ in TARGETS:
        stk, y, groups, seeds = _build_stack(target, n_cls)
        prior = _empirical_prior(y, n_cls)
        per_target[target] = {}
        for name, factory in RULES.items():
            per_target[target][name] = {
                "nested_cv": nested_cv_f1(factory, stk, y, prior, groups, n_cls),
                "seed_holdout": seed_holdout_f1(factory, stk, y, prior, seeds, n_cls, holdout),
            }
    # overall (nested-CV) per rule = 0.4*action + 0.4*point + 0.2*server_baseline
    result["per_target"] = per_target
    result["overall_nested"] = {
        name: overall(per_target["action"][name]["nested_cv"],
                      per_target["point"][name]["nested_cv"],
                      SERVER_AUC_BASELINE)
        for name in RULES
    }
    base = result["overall_nested"]["additive_baseline"]
    result["lift_vs_baseline"] = {n: result["overall_nested"][n] - base for n in RULES}
    result["noise_floor"] = 0.00168
    Path("artifacts/decision_rule_scores.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # verdict
    winner, best_lift = "additive_baseline", 0.0
    for n in RULES:
        if n == "additive_baseline":
            continue
        lift = result["lift_vs_baseline"][n]
        a_hold = per_target["action"][n]["seed_holdout"] >= per_target["action"]["additive_baseline"]["seed_holdout"] - 1e-9
        p_hold = per_target["point"][n]["seed_holdout"] >= per_target["point"]["additive_baseline"]["seed_holdout"] - 1e-9
        if lift > 0.00168 and a_hold and p_hold and lift > best_lift:
            winner, best_lift = n, lift
    print(f"\nGATE VERDICT: winner={winner} lift={best_lift:+.5f} "
          f"({'SHIP' if winner != 'additive_baseline' else 'REJECT — keep production rule'})")


if __name__ == "__main__":
    import sys
    main(holdout=int(sys.argv[1]) if len(sys.argv) > 1 else 55)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aicup-tt pytest tests/test_eval_decision_rule.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_decision_rule.py tests/test_eval_decision_rule.py
git commit -m "feat(eval): P1 decision-rule gate harness (nested-CV + seed-55 holdout)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Run the gate on real OOF and record the verdict

**Files:**
- Produces: `artifacts/decision_rule_scores.json`
- Modify: `PROGRESS.md` (append a P1 results section)

- [ ] **Step 1: Run the harness on real OOF (CPU; a few minutes)**

Run: `conda run -n aicup-tt python -m scripts.eval_decision_rule 2>&1 | tee artifacts/decision_rule_run.log`
Expected: prints per-target nested_cv + seed_holdout for all four rules, the overall-nested table, the lift-vs-baseline, and a `GATE VERDICT` line. `artifacts/decision_rule_scores.json` is written.

- [ ] **Step 2: Sanity-check the baseline reproduces production**

The `additive_baseline` overall_nested should land within noise of the recorded production
0.32568 (it is the same rule on the same stacked OOF; small drift is acceptable because the
prior here is per-fold). Confirm `per_target["action"]["additive_baseline"]["nested_cv"]` ≈ 0.296
and `["point"]...` ≈ 0.189. If they are wildly off, STOP — the harness is not reproducing the
production pipeline and the comparison is invalid.

- [ ] **Step 3: Record the verdict in `PROGRESS.md`**

Append a section titled `## Private-push v3 — P1 (macro-F1 decision rule) <SHIP/REJECT> (2026-05-29)` containing:
- the four-rule table (action/point nested_cv + seed_holdout, overall_nested, lift),
- the gate decision (winner + lift vs 0.00168 floor; both seed-holdout targets non-regressing),
- whether Task 7 (wiring) runs or the production rule stays.

- [ ] **Step 4: Commit**

```bash
git add -f artifacts/decision_rule_scores.json artifacts/decision_rule_run.log
git add PROGRESS.md
git commit -m "feat(p1): decision-rule gate result on real OOF — <SHIP/REJECT>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7 (CONDITIONAL — only if Task 6 verdict is SHIP): wire the winning rule into production

Skip this task entirely if the verdict was REJECT; the production rule already stays in place and P1 is complete (record REJECT and move to P2). Only run this if a rule cleared > 0.00168 on nested-CV with non-regressing seed holdout on both targets.

**Files:**
- Modify: `scripts/build_final_perrow.py:90-96` (`_nested_f1`) and `:120-148` (deploy path)
- Test: `tests/test_build_final_rule.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_final_rule.py
import numpy as np
from scripts.build_final_perrow import _nested_f1
from scripts.decision_rule import RULES


def test_nested_f1_accepts_rule_factory_and_matches_baseline():
    # With the additive_baseline factory, _nested_f1 must equal its own prior behaviour.
    rng = np.random.default_rng(0)
    n, n_cls = 500, 3
    corrected = rng.dirichlet(np.ones(n_cls), size=n)
    y = corrected.argmax(1)
    groups = rng.integers(0, 15, size=n)
    f = _nested_f1(corrected, y, groups, n_cls, rule_factory=RULES["additive_baseline"])
    assert 0.0 <= f <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aicup-tt pytest tests/test_build_final_rule.py -v`
Expected: FAIL — `_nested_f1() got an unexpected keyword argument 'rule_factory'`.

- [ ] **Step 3: Make `_nested_f1` and the deploy path rule-pluggable**

Replace `_nested_f1` in `scripts/build_final_perrow.py` with a version that takes a rule factory
(defaulting to the winner identified in Task 6 — substitute `RULES["<winner>"]` for `<winner>`):

```python
from scripts.decision_rule import RULES

PROD_RULE = "<winner>"  # set to the Task-6 winner, e.g. "additive_wide"


def _nested_f1(corrected, y, groups, n_cls, n_folds=5, rule_factory=None):
    """Honest macro-F1 via the pluggable decision rule, nested over folds.

    `corrected` is already prior-corrected upstream; the additive/weighted rules
    re-apply prior_correct on raw probs, so here we pass a uniform prior to make
    prior_correct a no-op and keep the existing call sites valid.
    """
    rule_factory = rule_factory or RULES[PROD_RULE]
    uniform = np.full(n_cls, 1.0 / n_cls)
    yhat = np.zeros(len(y), dtype=int)
    for tr, va in GroupKFold(n_splits=n_folds).split(corrected, y, groups):
        rule = rule_factory().fit(corrected[tr], y[tr], n_cls, uniform)
        yhat[va] = rule.predict(corrected[va], n_cls, uniform)
    return float(f1_score(y, yhat, labels=list(range(n_cls)),
                          average="macro", zero_division=0))
```

Then update the deploy path (lines ~128-129 and ~146-148) so the submission uses the same rule
instead of the raw `tune_thresholds` / `apply_thresholds`:

```python
        # deployment rule: fit on ALL OOF corrected probs (no test labels available)
        uniform = np.full(n_cls, 1.0 / n_cls)
        deploy_rule[target] = RULES[PROD_RULE]().fit(corrected, y, n_cls, uniform)
```

```python
            aligned_corrected = prior_correct(aligned, prior)
            pred = deploy_rule[target].predict(aligned_corrected, n_cls, uniform)
            submission[y_col] = pred.astype(int)
```

(Replace the `deploy_thr` dict with `deploy_rule: dict[str, object] = {}` near line 106.)

- [ ] **Step 4: Run the new test + full suite**

Run: `conda run -n aicup-tt pytest tests/test_build_final_rule.py -q && conda run -n aicup-tt pytest -q`
Expected: all PASS.

- [ ] **Step 5: Rebuild production submissions and confirm the lift**

Run: `conda run -n aicup-tt python -m scripts.build_final_perrow`
Expected: prints honest per-row scores; overall ≥ 0.32568 + 0.00168. Guardrails (1845 rallies,
valid ranges) pass. `submission_FINAL_safe_perrow.csv` + `submission_FINAL_smooth_perrow.csv` rewritten.

- [ ] **Step 6: Record + commit**

Append the shipped overall to the `PROGRESS.md` P1 section, then:

```bash
git add scripts/build_final_perrow.py tests/test_build_final_rule.py PROGRESS.md
git add -f artifacts/final_perrow_scores.json
git commit -m "feat(p1): SHIP <winner> decision rule into production builder (+<lift>)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes

- **Spec coverage:** P1a (Task 1+2 wide grid/convergence), P1b (Task 3 calibration), P1c (Task 4 multiplicative weights — the spec's "frequency-quota" intent realized as the cleaner, non-degenerate multiplicative plug-in; both exploit the set-level nature of macro-F1, and multiplicative weights avoid collapsing into the additive rule). Gate = nested-CV + seed-55 holdout (Task 5/6), matching the spec's overfit guard. Production wiring is gated (Task 7). Server untouched, as the spec states.
- **No new deps:** isotonic/LR/f1 are all in the existing `aicup-tt` env; no clone env needed.
- **Type consistency:** every rule implements `fit(probs, y, n_cls, prior) -> self` and `predict(probs, n_cls, prior) -> int[]`; `RULES` maps name -> zero-arg factory; the harness and the builder both consume `RULES[name]` factories.
- **Honesty:** baseline rule reproduces the current pipeline exactly (Task 2 test asserts equality), so the lift is measured against the real production rule, not a strawman.
