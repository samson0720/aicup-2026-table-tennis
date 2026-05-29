# Rare-Class Macro-F1 Lever (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether training-time rare-class rebalancing (stronger class weights, then rare-class oversampling) on the CatBoost base lifts the honest per-row ensemble above the current 0.32379, and ship any variant that clears the 0.00168 noise floor.

**Architecture:** Reuse the CatBoost OOF/test pipeline (`produce_catboost_oof.py`, `predict_test_catboost.py`) and the per-row ensemble (`build_final_perrow.py`). Add a `--weight-mode` passthrough (the fit helpers already support `"balanced"`) and a small numpy rare-class oversampler, producing rebalanced base variants `cat_bal` / `cat_os`. Gate each by ensemble lift; the metric is macro-F1 so rare action/point recall is the lever. Threshold tuning + prior-correction already run downstream, so these training-time variants must add value *on top of* that.

**Tech Stack:** Python 3, CatBoost (GPU, 3090), numpy/pandas, scikit-learn, pytest. conda env **`aicup-tt`** only. Honest per-row scoring; no seed-averaging.

## Critical context (read before starting)

- **Honest ruler.** Current production = 6-base per-row ensemble incl. `cat`, honest overall **0.32379** (action 0.2939, point 0.1873, server 0.6567). Noise floor (across-seed std) = **0.00168**. Gate baseline for this plan = **0.32379**.
- **Why rare classes:** metric `overall = 0.4·action_macroF1 + 0.4·point_macroF1 + 0.2·server_AUC`. Macro-F1 weights all classes equally; action (19 cls) and point (10 cls) have heavy class imbalance (e.g. action class 1 ≈ 15k vs rare classes ≈ hundreds). Up-weighting/oversampling rare classes can raise macro-F1 if it improves rare-class probability estimates beyond what downstream threshold tuning already extracts.
- **Existing knobs:**
  - `scripts/train_lgbm_baseline.class_weights(y, classes, mode)` supports `"none" | "sqrt" | "balanced"` (`balanced` = full `n/(k·cnt)`, `sqrt` = its sqrt). The CatBoost fit helpers (`scripts/train_catboost_baseline.py`) already take `weight_mode` and build CatBoost `class_weights` from it. The current base `cat` uses `"sqrt"`.
  - `scripts/produce_catboost_oof.py` (GPU, per-row OOF, `--gpu --iterations 400 --depth 6 --model-name`) and `scripts/predict_test_catboost.py` (full-train test, GPU) — both currently hardcode `weight_mode="sqrt"`.
  - `scripts/build_final_perrow.py` BASES (currently includes `"cat"`); read_oof/test paths compose `<model>_<target>[_test].parquet`.
- **GPU:** `env CUDA_VISIBLE_DEVICES=0` (3090). catboost 1.2.10 has no GPU-multiclass class-dropping bug (verified).
- **conda:** `conda run -n aicup-tt ...` only.

---

## Task 1: `--weight-mode` passthrough + `cat_bal` (balanced weights)

**Files:**
- Modify: `scripts/produce_catboost_oof.py`
- Modify: `scripts/predict_test_catboost.py`

- [ ] **Step 1: Add `--weight-mode` to the OOF producer**

In `scripts/produce_catboost_oof.py`, the three fit calls currently pass the literal `"sqrt"`. Replace each `"sqrt"` argument in the `fit_multiclass(...)` calls with `args.weight_mode`, and add the arg in `main()`:
```python
    p.add_argument("--weight-mode", default="sqrt", choices=["none", "sqrt", "balanced"])
```
(`fit_binary` ignores weight_mode — it uses pos/neg ratio — so leave it.)

- [ ] **Step 2: Add `--weight-mode` to the test-inference script**

In `scripts/predict_test_catboost.py`, change `run(...)` to accept `weight_mode: str = "sqrt"`, pass it into both `fit_full_multiclass(...)` calls (replace the literal `"sqrt"`), and add `--weight-mode` to `main()` forwarding into `run(...)`.

- [ ] **Step 3: Generate `cat_bal` OOF (balanced weights) on GPU**

Run (background; ~10–15 min):
```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt --no-capture-output python -u -m scripts.produce_catboost_oof --gpu --iterations 400 --depth 6 --weight-mode balanced --model-name cat_bal 2>&1 | grep -v "unique classes"
```
Expected: 25 `cat_bal seed=.. fold=..` lines, then three `wrote artifacts/oof/cat_bal_* : rows=74975`.

- [ ] **Step 4: Generate `cat_bal` test predictions**

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt --no-capture-output python -u -m scripts.predict_test_catboost --iterations 600 --weight-mode balanced --model-name cat_bal 2>&1 | grep -viE "unique classes|metric period"
```
Expected: `cat_bal test: action (1845, 19), point (1845, 10), server (1845, 1)` then the wrote line.

- [ ] **Step 5: Score `cat_bal` standalone**

```bash
conda run -n aicup-tt --no-capture-output python -m scripts.score_oof cat cat_bal 2>&1 | grep -A6 '"cat_bal"'
```
Record action/point F1 vs `cat` (0.2716 / 0.1739). Balanced weights should raise rare-class recall (possibly lifting macro-F1, possibly hurting via majority-class loss).

- [ ] **Step 6: Commit**

```bash
git add scripts/produce_catboost_oof.py scripts/predict_test_catboost.py
git add -f artifacts/oof/cat_bal_action.parquet artifacts/oof/cat_bal_point.parquet artifacts/oof/cat_bal_server.parquet artifacts/oof/cat_bal_action_test.parquet artifacts/oof/cat_bal_point_test.parquet artifacts/oof/cat_bal_server_test.parquet
git add artifacts/base_oof_scores.json
git commit -m "feat(cat_bal): balanced-weight CatBoost variant (rare-class macro-F1)"
```

---

## Task 2: Rare-class oversampler + `cat_os`

**Files:**
- Create: `scripts/resample.py`
- Test: `tests/test_resample.py`
- Modify: `scripts/produce_catboost_oof.py`
- Modify: `scripts/predict_test_catboost.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_resample.py`:
```python
import numpy as np
import pandas as pd

from scripts.resample import oversample_rare


def test_oversample_brings_rare_classes_up_to_floor():
    # class 0: 100 rows, class 1: 10 rows, class 2: 5 rows
    y = pd.Series([0] * 100 + [1] * 10 + [2] * 5)
    x = pd.DataFrame({"f": np.arange(len(y))})
    xo, yo = oversample_rare(x, y, seed=0, min_frac=0.5)
    counts = yo.value_counts().to_dict()
    # rare classes raised to >= 0.5 * max (=50); majority untouched
    assert counts[0] == 100
    assert counts[1] >= 50
    assert counts[2] >= 50
    # only duplicates real rows (feature values stay within the original set)
    assert set(xo["f"]).issubset(set(x["f"]))
    # x and y stay aligned (same length, index reset)
    assert len(xo) == len(yo)


def test_oversample_noop_when_balanced():
    y = pd.Series([0] * 50 + [1] * 50)
    x = pd.DataFrame({"f": np.arange(100)})
    xo, yo = oversample_rare(x, y, seed=0, min_frac=0.5)
    assert len(xo) == 100  # already balanced -> no change
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n aicup-tt python -m pytest tests/test_resample.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.resample'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/resample.py`:
```python
"""Random rare-class oversampling for macro-F1 (duplicates real rows only)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def oversample_rare(x: pd.DataFrame, y: pd.Series, seed: int, min_frac: float = 0.3):
    """Duplicate rows of minority classes until each class has at least
    ``min_frac * max_class_count`` rows. Majority classes are untouched.
    Returns (x_resampled, y_resampled) with reset indices."""
    y = y.reset_index(drop=True)
    x = x.reset_index(drop=True)
    counts = y.value_counts()
    floor = int(np.ceil(min_frac * int(counts.max())))
    rng = np.random.default_rng(seed)
    extra_idx: list[np.ndarray] = []
    for cls, cnt in counts.items():
        if cnt >= floor:
            continue
        pool = np.flatnonzero((y == cls).to_numpy())
        extra_idx.append(rng.choice(pool, size=floor - cnt, replace=True))
    if not extra_idx:
        return x, y
    add = np.concatenate(extra_idx)
    keep = np.arange(len(y))
    order = np.concatenate([keep, add])
    return x.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n aicup-tt python -m pytest tests/test_resample.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Wire `--oversample` into the OOF producer**

In `scripts/produce_catboost_oof.py`: import `from scripts.resample import oversample_rare`. Add arg `p.add_argument("--oversample", type=float, default=0.0, help="min_frac for rare-class oversampling; 0 disables")`. For action and point only (not server), when `args.oversample > 0`, build a per-target oversampled training matrix before fitting:
```python
        if args.oversample > 0:
            xa, ya = oversample_rare(x_train, df_train["y_actionId"], 9000 + fold, args.oversample)
            xp, yp = oversample_rare(x_train, df_train["y_pointId"], 9100 + fold, args.oversample)
        else:
            xa, ya = x_train, df_train["y_actionId"]
            xp, yp = x_train, df_train["y_pointId"]
        pa = fit_multiclass(xa, ya, x_valid, TARGET_ACTION_CLASSES, cat_idx, args.weight_mode, 9000 + fold, args.iterations, depth=args.depth, task_type=task_type, devices=devices)
        pp = fit_multiclass(xp, yp, x_valid, TARGET_POINT_CLASSES, cat_idx, args.weight_mode, 9100 + fold, args.iterations, depth=args.depth, task_type=task_type, devices=devices)
```
(Replace the existing `pa`/`pp` fit lines with the block above; leave `ps`/server unchanged.)

- [ ] **Step 6: Wire `--oversample` into the test-inference script**

In `scripts/predict_test_catboost.py`: import `oversample_rare`; add `oversample: float = 0.0` to `run(...)` and `--oversample` to `main()`. Before the action/point `fit_full_multiclass` calls, oversample the full-train matrix per target when `oversample > 0` (mirror Task 2 Step 5, using seeds 9000/9100); leave server unchanged.

- [ ] **Step 7: Generate `cat_os` OOF + test (oversample, keep sqrt weights)**

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt --no-capture-output python -u -m scripts.produce_catboost_oof --gpu --iterations 400 --depth 6 --weight-mode sqrt --oversample 0.3 --model-name cat_os 2>&1 | grep -v "unique classes"
env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt --no-capture-output python -u -m scripts.predict_test_catboost --iterations 600 --weight-mode sqrt --oversample 0.3 --model-name cat_os 2>&1 | grep -viE "unique classes|metric period"
```
Expected: `cat_os_* : rows=74975` (OOF) and `cat_os test: action (1845, 19) ...`.

- [ ] **Step 8: Score `cat_os` standalone + commit**

```bash
conda run -n aicup-tt --no-capture-output python -m scripts.score_oof cat cat_bal cat_os 2>&1 | grep -A6 '"cat_os"'
git add scripts/resample.py tests/test_resample.py scripts/produce_catboost_oof.py scripts/predict_test_catboost.py
git add -f artifacts/oof/cat_os_action.parquet artifacts/oof/cat_os_point.parquet artifacts/oof/cat_os_server.parquet artifacts/oof/cat_os_action_test.parquet artifacts/oof/cat_os_point_test.parquet artifacts/oof/cat_os_server_test.parquet
git add artifacts/base_oof_scores.json
git commit -m "feat(cat_os): rare-class oversampled CatBoost variant + oversampler"
```

---

## Task 3: Gate the rebalanced variants into the ensemble

**Files:**
- Modify: `scripts/build_final_perrow.py:29-33` (BASES)
- Modify: `PROGRESS.md`

Test each variant two ways — as an *extra* base alongside `cat`, and as a *replacement* for `cat` — and keep only what clears the floor.

- [ ] **Step 1: Capture the gate baseline**

```bash
conda run -n aicup-tt --no-capture-output python -c "import json; print('baseline', json.load(open('artifacts/final_perrow_scores.json'))['overall'])"
```
Expected: `baseline 0.32379414606382295`.

- [ ] **Step 2: Try `cat_bal` as an extra base**

Edit `BASES` in `scripts/build_final_perrow.py` to append `"cat_bal"` to all three lists (alongside `"cat"`). Run:
```bash
conda run -n aicup-tt --no-capture-output python -m scripts.build_final_perrow 2>&1 | grep -E '"overall"'
```
Record overall; lift = overall − 0.32379.

- [ ] **Step 3: Try `cat_os` as an extra base**

Revert the Step-2 edit; instead append `"cat_os"` to all three lists. Rerun the build; record overall/lift.

- [ ] **Step 4: Try the best variant as a replacement for `cat`**

For whichever of `cat_bal`/`cat_os` scored higher in Steps 2–3, also test *replacing* `"cat"` with it (swap the name in all three BASES lists, no duplicate). Rerun; record overall/lift.

- [ ] **Step 5: Apply the gate**

- If the best configuration's lift **> 0.00168**: keep that `BASES` edit (it ships). Proceed to Step 6 and commit the integration.
- If no configuration clears **0.00168**: revert all edits and restore artifacts:
  ```bash
  git checkout -- scripts/build_final_perrow.py artifacts/final_perrow_scores.json artifacts/submission_FINAL_safe_perrow.csv artifacts/submission_FINAL_smooth_perrow.csv
  ```
  Then record the negative result (Step 6) and stop Phase 1 (rare-class rebalancing did not beat the existing prior+threshold pipeline).

- [ ] **Step 6: Record in PROGRESS.md + commit**

Add a `## Rare-class macro-F1 (Phase 1) result` section: `cat_bal`/`cat_os` standalone scores, the best ensemble overall + lift vs 0.32379, and SHIP/REJECT vs the 0.00168 floor. If shipped, note the new production overall; if rejected, note variants/scripts kept for reproducibility (parquets discardable). Commit:
```bash
git add PROGRESS.md scripts/build_final_perrow.py
# if shipped, also: git add artifacts/final_perrow_scores.json artifacts/submission_FINAL_safe_perrow.csv artifacts/submission_FINAL_smooth_perrow.csv
git commit -m "docs(progress): record rare-class macro-F1 (Phase 1) result"
```

---

## Self-review checklist (run before handing off)

- [ ] `conda run -n aicup-tt python -m pytest -q` — all green (existing 38 + `test_resample` 2 = 40).
- [ ] No seed-averaging; honest per-row scoring; gate baseline = 0.32379.
- [ ] `cat_bal`/`cat_os` OOF key sets match the other bases (74,975 rows; inner-join keeps full population).
- [ ] OOF/test parquet `git add` uses `-f` (artifacts/oof/*.parquet gitignored).
- [ ] Gate is explicit: ship only if ensemble lift > 0.00168; otherwise revert cleanly.
- [ ] Phases 2 (XGBoost/FT-Transformer) and 3 (hill-climbing stacker) are planned just-in-time after this gate.
