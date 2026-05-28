# CatBoost Base (Prong A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CatBoost base to the honest per-row ensemble and decide (gate on the 0.00168 noise floor) whether it lifts the ensemble overall above the current 0.3206.

**Architecture:** Reuse the *exact* prefix-feature pipeline that feeds the LGBM bases (`build_one_sample_per_rally` + `feature_columns`), swapping the learner to CatBoost with native categorical handling. Mirror the existing OOF producer (`produce_base_oof.run_lgbm`) and test-inference helper (`predict_test_base.predict_test_lgbm`) so CatBoost is apples-to-apples with `lgbm15`. CatBoost runs on **CPU** (≈12k rows/fold is below GPU break-even and sidesteps CatBoost's historical GPU-multiclass class-dropping bug — same rationale as the repo's LGBM-stays-CPU policy).

**Tech Stack:** Python 3, CatBoost (conda-forge), pandas/numpy, scikit-learn, pytest. conda env **`aicup-tt`** only. No seed-averaging; honest per-row scoring throughout.

## Critical context (read before starting)

- **Honest ruler only.** Score on the per-row `(rally_uid, seed, fold, cut_strikeNumber)` population. Current honest ensemble overall = **0.3206**; best base lgbm15 = 0.3027; noise floor (across-seed std on overall) = **0.00168**.
- **Existing CatBoost helpers** live in `scripts/train_catboost_baseline.py` (CPU, working logic): `cat_feature_indices(cols)->list[int]`, `prepare_x(x, cat_cols)->DataFrame` (casts cat cols to str), `align_multiclass(model, x, classes)->np.ndarray` (n×len(classes), fills uniform for absent classes), `fit_multiclass(x_train,y_train,x_valid,classes,cat_features,weight_mode,seed,iterations)->proba`, `fit_binary(x_train,y_train,x_valid,cat_features,seed,iterations)->p[:,1]`, `fit_full_multiclass(x,y,classes,cat_idx,weight_mode,seed,iterations)->model`, `fit_full_binary(x,y,cat_idx,seed,iterations)->model`. **But its top-level imports are bare** (`from make_lgbm_submission import ...`), so it is not importable as `scripts.train_catboost_baseline` until Task 2 fixes them. **And CatBoost is not yet installed** (import currently raises `ModuleNotFoundError`).
- **Feature pipeline (reused, do not reimplement):**
  - `scripts.diagnose_cv_gap.build_one_sample_per_rally(view, splits_sub) -> df` — one row per rally with `rally_uid, match, target_strikeNumber, phase, y_actionId, y_pointId, y_serverGetPoint` + feature columns.
  - `scripts.train_lgbm_baseline.feature_columns(df) -> list[str]` (excludes `y_*`, `rally_uid`), `TARGET_ACTION_CLASSES=list(range(19))`, `TARGET_POINT_CLASSES=list(range(10))`, `build_prefix_dataset(train)->df`.
  - `scripts.cv_splits.iter_cv_folds(train, splits)` yields `(seed, fold, train_view, valid_view)`.
  - `scripts.make_lgbm_submission.build_test_dataset(test) -> df` (test prefix features, one row per test rally).
  - `scripts.oof_loader.write_oof(model, target, rally_uid, seed, fold, cut, probs)`; `scripts.predict_test_base._write_test_parquet(model, target, rally_uid, probs) -> Path`.
- **OOF parquet schema:** `rally_uid, seed, fold, cut_strikeNumber` + `p_0..p_{k-1}` (action k=19, point k=10) or `p_1` (server). Full run = 74,975 rows. Test parquet = `rally_uid` + `p_*`, 1,845 rows.
- **GPU note:** CatBoost here is CPU. Do NOT set `task_type='GPU'`. (The RTX 3090 is reserved for the PyTorch prongs.)
- **conda:** run everything via `conda run -n aicup-tt ...`; never base/system Python.

---

## Task 1: Install CatBoost (conda-forge) + env-safety check

**Files:** none (environment change + `environment.yml` note).

- [ ] **Step 1: Install CatBoost from conda-forge**

Run:
```bash
conda install -n aicup-tt -c conda-forge catboost -y
```
Expected: resolves and installs `catboost` (4.x). Does NOT downgrade `pytorch`/`numpy`/`pandas`.

- [ ] **Step 2: Verify CatBoost imports AND PyTorch/CUDA still works**

Run:
```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt --no-capture-output python -c "import catboost, torch; print('catboost', catboost.__version__); print('torch.cuda', torch.cuda.is_available())"
```
Expected: prints a catboost version and `torch.cuda True`. If `torch.cuda` is now False or torch import breaks, STOP — the install disturbed the CUDA stack; roll back (`conda remove -n aicup-tt catboost`) and instead clone the env (`conda create --clone aicup-tt -n aicup-tt-cat`) and install there, running all CatBoost commands in `aicup-tt-cat`.

- [ ] **Step 3: Verify a tiny CPU CatBoost multiclass fit predicts all classes**

Run:
```bash
conda run -n aicup-tt --no-capture-output python -c "
import numpy as np
from catboost import CatBoostClassifier
rng=np.random.default_rng(0); X=rng.random((300,4)); y=rng.integers(0,10,300)
m=CatBoostClassifier(iterations=20, loss_function='MultiClass', verbose=False, allow_writing_files=False, thread_count=-1, classes_count=10)
m.fit(X,y); p=m.predict_proba(X)
print('proba shape', p.shape, 'classes', sorted(int(c) for c in m.classes_))
assert p.shape==(300,10)
print('OK')
"
```
Expected: `proba shape (300, 10)`, classes `[0..9]`, `OK`.

- [ ] **Step 4: Record the dependency in environment.yml**

Add `catboost` under the conda-forge dependencies in `environment.yml` (match the existing formatting; pin only the major version if other deps are pinned). Then commit:
```bash
git add environment.yml
git commit -m "build(env): add catboost (conda-forge) for the CatBoost base"
```

---

## Task 2: Make `train_catboost_baseline.py` importable as a module

**Files:**
- Modify: `scripts/train_catboost_baseline.py` (imports only)
- Test: `tests/test_catboost_helpers.py`

The helpers are reused by Tasks 3 and 5, so the module must import cleanly as `scripts.train_catboost_baseline`. Only the two bare imports change; all function bodies stay as-is.

- [ ] **Step 1: Write the failing test**

Create `tests/test_catboost_helpers.py`:
```python
import numpy as np
import pandas as pd

from scripts.train_catboost_baseline import (
    cat_feature_indices,
    prepare_x,
    fit_multiclass,
)


def test_cat_feature_indices_splits_continuous_from_categorical():
    cols = ["prefix_len", "scoreSelf", "strikeId", "handId", "act_cnt_3", "spin_entropy"]
    idx = cat_feature_indices(cols)
    # continuous + *_cnt_* + *_entropy excluded; only strikeId, handId remain
    assert [cols[i] for i in idx] == ["strikeId", "handId"]


def test_fit_multiclass_returns_aligned_proba():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.integers(0, 5, 200), "b": rng.integers(0, 3, 200)})
    y = pd.Series(rng.integers(0, 10, 200))
    cat_idx = [0, 1]
    xs = prepare_x(x, ["a", "b"])
    p = fit_multiclass(xs, y, xs.iloc[:20], list(range(10)), cat_idx, "sqrt", 0, 20)
    assert p.shape == (20, 10)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-3)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n aicup-tt python -m pytest tests/test_catboost_helpers.py -q`
Expected: FAIL on import — `ModuleNotFoundError: No module named 'make_lgbm_submission'` (the bare import).

- [ ] **Step 3: Fix the two bare imports**

In `scripts/train_catboost_baseline.py`, change:
```python
from make_lgbm_submission import build_test_dataset
from train_lgbm_baseline import (
```
to:
```python
from scripts.make_lgbm_submission import build_test_dataset
from scripts.train_lgbm_baseline import (
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n aicup-tt python -m pytest tests/test_catboost_helpers.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_catboost_baseline.py tests/test_catboost_helpers.py
git commit -m "refactor(catboost): make helpers importable as a module + tests"
```

---

## Task 3: CatBoost OOF producer `produce_catboost_oof.py`

**Files:**
- Create: `scripts/produce_catboost_oof.py`

Mirrors `produce_base_oof.run_lgbm` exactly, swapping the LGBM fits for the CatBoost helpers. Adds `--seeds`/`--folds` filters so the full run can be smoke-tested on one cell first.

- [ ] **Step 1: Write the implementation**

Create `scripts/produce_catboost_oof.py`:
```python
"""Produce CatBoost OOF parquets on cv_splits.parquet (per-row, honest).

Mirrors produce_base_oof.run_lgbm but uses the CatBoost helpers from
train_catboost_baseline. CPU only. Writes artifacts/oof/cat_{target}.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.oof_loader import write_oof
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    feature_columns,
)
from scripts.train_catboost_baseline import (
    cat_feature_indices,
    fit_binary,
    fit_multiclass,
    prepare_x,
)


def _stack(rs, ss, fs, cs, ps):
    return (np.concatenate(rs), np.concatenate(ss), np.concatenate(fs),
            np.concatenate(cs), np.concatenate(ps, axis=0))


def run(args) -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))

    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        if args.seeds and seed not in args.seeds:
            continue
        if args.folds and fold not in args.folds:
            continue
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty:
            continue
        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        cat_idx = cat_feature_indices(feats)
        cat_cols = [feats[i] for i in cat_idx]
        x_train = prepare_x(df_train[feats], cat_cols)
        x_valid = prepare_x(df_valid[feats], cat_cols)

        pa = fit_multiclass(x_train, df_train["y_actionId"], x_valid,
                            TARGET_ACTION_CLASSES, cat_idx, "sqrt", 9000 + fold, args.iterations)
        pp = fit_multiclass(x_train, df_train["y_pointId"], x_valid,
                            TARGET_POINT_CLASSES, cat_idx, "sqrt", 9100 + fold, args.iterations)
        ps = fit_binary(x_train, df_train["y_serverGetPoint"], x_valid,
                        cat_idx, 9200 + fold, args.iterations).reshape(-1, 1)

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed); fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()
        for tgt, p in (("action", pa), ("point", pp), ("server", ps)):
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"cat seed={seed} fold={fold} valid_n={len(rally)}", flush=True)

    for tgt in ("action", "point", "server"):
        r, s, f, c, p = _stack(bag[tgt]["r"], bag[tgt]["s"], bag[tgt]["f"], bag[tgt]["c"], bag[tgt]["p"])
        out = write_oof("cat", tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--iterations", type=int, default=400)
    run(p.parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test one cell (seed 11, fold 0)**

Run:
```bash
conda run -n aicup-tt --no-capture-output python -u -m scripts.produce_catboost_oof --seeds 11 --folds 0 --iterations 80 2>&1 | tail -6
```
Expected: one `cat seed=11 fold=0 valid_n=...` line, then three `wrote artifacts/oof/cat_*` lines with matching `rows=` (the single fold's valid count, ~600–700). Completes in well under 2 minutes.

- [ ] **Step 3: Verify the smoke OOF schema**

Run:
```bash
conda run -n aicup-tt --no-capture-output python -c "import pandas as pd; d=pd.read_parquet('artifacts/oof/cat_action.parquet'); print(d.shape, list(d.columns)[:6], d.seed.unique(), sorted(d.fold.unique()))"
```
Expected: columns start `['rally_uid','seed','fold','cut_strikeNumber','p_0','p_1']`, `seed.unique()=[11]`, `fold.unique()=[0]`, 19 `p_` columns total.

- [ ] **Step 4: Commit the producer**

```bash
git add scripts/produce_catboost_oof.py
git commit -m "feat(catboost): per-row OOF producer (mirrors lgbm, CPU)"
```

---

## Task 4: Full 25-fold CatBoost OOF + standalone score

**Files:** none (produces `artifacts/oof/cat_{action,point,server}.parquet`).

- [ ] **Step 1: Run the full 25-fold OOF**

Run (in the background; ~10–25 min CPU):
```bash
conda run -n aicup-tt --no-capture-output python -u -m scripts.produce_catboost_oof --iterations 400 2>&1 | tail -8
```
Expected: 25 `cat seed=.. fold=..` lines, then three `wrote artifacts/oof/cat_* : rows=74975`.

- [ ] **Step 2: Verify row count and key alignment with the other bases**

Run:
```bash
conda run -n aicup-tt --no-capture-output python -c "
import pandas as pd
KEYS=['rally_uid','seed','fold','cut_strikeNumber']
a=pd.read_parquet('artifacts/oof/lgbm15_action.parquet')[KEYS]
b=pd.read_parquet('artifacts/oof/cat_action.parquet')[KEYS]
print('lgbm15', len(a), 'cat', len(b))
print('identical key set?', set(map(tuple,a.values.tolist()))==set(map(tuple,b.values.tolist())))
"
```
Expected: both `74975`, `identical key set? True` (required so the ensemble inner-join keeps the full population).

- [ ] **Step 3: Score CatBoost standalone (honest argmax)**

Run:
```bash
conda run -n aicup-tt --no-capture-output python -m scripts.score_oof lgbm15 lgbm31 markov phase_lgbm cat 2>&1 | grep -A6 '"cat"'
```
Expected: prints `cat` action/point/server/overall. Record these numbers (sanity vs lgbm15 0.3027 — standalone may be lower; the value is ensemble diversity).

- [ ] **Step 4: Commit the OOF**

```bash
git add -f artifacts/oof/cat_action.parquet artifacts/oof/cat_point.parquet artifacts/oof/cat_server.parquet
git add artifacts/base_oof_scores.json
git commit -m "feat(catboost): full 25-fold OOF + standalone scores"
```

---

## Task 5: CatBoost full-train test inference `predict_test_catboost.py`

**Files:**
- Create: `scripts/predict_test_catboost.py`

Mirrors `predict_test_base.predict_test_lgbm`, using the CatBoost full-train helpers. Writes `artifacts/oof/cat_{target}_test.parquet` (1,845 rows, base schema).

- [ ] **Step 1: Write the implementation**

Create `scripts/predict_test_catboost.py`:
```python
"""Full-train CatBoost test inference -> artifacts/oof/cat_{target}_test.parquet.

Mirrors predict_test_base.predict_test_lgbm with the CatBoost full-train fits.
Single-cut per test rally; distribution-matched to the OOF models.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_lgbm_submission import build_test_dataset
from scripts.predict_test_base import _write_test_parquet
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    build_prefix_dataset,
    feature_columns,
)
from scripts.train_catboost_baseline import (
    align_multiclass,
    cat_feature_indices,
    fit_full_binary,
    fit_full_multiclass,
    prepare_x,
)


def _full_train_features(train: pd.DataFrame) -> pd.DataFrame:
    cache = Path("artifacts/prefix_train_baseline.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    df = build_prefix_dataset(train)
    df.to_parquet(cache, index=False)
    return df


def run(iterations: int = 600) -> None:
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")

    df_train = _full_train_features(train)
    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(df_train) if c in test_features.columns]
    cat_idx = cat_feature_indices(feats)
    cat_cols = [feats[i] for i in cat_idx]
    x_train = prepare_x(df_train[feats], cat_cols)
    x_test = prepare_x(test_features[feats], cat_cols)

    action_model = fit_full_multiclass(x_train, df_train["y_actionId"], TARGET_ACTION_CLASSES, cat_idx, "sqrt", 9000, iterations)
    point_model = fit_full_multiclass(x_train, df_train["y_pointId"], TARGET_POINT_CLASSES, cat_idx, "sqrt", 9100, iterations)
    server_model = fit_full_binary(x_train, df_train["y_serverGetPoint"], cat_idx, 9200, iterations)

    rally = test_features["rally_uid"].to_numpy()
    p_action = align_multiclass(action_model, x_test, TARGET_ACTION_CLASSES)
    p_point = align_multiclass(point_model, x_test, TARGET_POINT_CLASSES)
    pos_idx = list(server_model.classes_).index(1)
    p_server = server_model.predict_proba(x_test)[:, pos_idx].reshape(-1, 1)

    print(f"cat test: action {p_action.shape}, point {p_point.shape}, server {p_server.shape}", flush=True)
    _write_test_parquet("cat", "action", rally, p_action)
    _write_test_parquet("cat", "point", rally, p_point)
    _write_test_parquet("cat", "server", rally, p_server)
    print("wrote cat_{action,point,server}_test.parquet", flush=True)


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run test inference**

Run:
```bash
conda run -n aicup-tt --no-capture-output python -u -m scripts.predict_test_catboost 2>&1 | tail -4
```
Expected: `cat test: action (1845, 19), point (1845, 10), server (1845, 1)`, then the wrote line.

- [ ] **Step 3: Verify schema + rally_uid alignment with test**

Run:
```bash
conda run -n aicup-tt --no-capture-output python -c "
import pandas as pd, numpy as np
from pathlib import Path
tu=np.sort(pd.read_csv(next(Path.cwd().glob('AI CUP*/test_new.csv')))['rally_uid'].unique())
for t in ['action','point','server']:
    d=pd.read_parquet(f'artifacts/oof/cat_{t}_test.parquet'); b=pd.read_parquet(f'artifacts/oof/lgbm15_{t}_test.parquet')
    print(t, d.shape, 'cols_ok', list(d.columns)==list(b.columns), 'uids_ok', np.array_equal(np.sort(d.rally_uid.unique()),tu))
"
```
Expected: each target `cols_ok True`, `uids_ok True`; action (1845,20), point (1845,11), server (1845,2).

- [ ] **Step 4: Commit**

```bash
git add scripts/predict_test_catboost.py
git add -f artifacts/oof/cat_action_test.parquet artifacts/oof/cat_point_test.parquet artifacts/oof/cat_server_test.parquet
git commit -m "feat(catboost): full-train test inference"
```

---

## Task 6: Integrate into the per-row ensemble + GATE

**Files:**
- Modify: `scripts/build_final_perrow.py:29-33` (the `BASES` dict)

- [ ] **Step 1: Capture the current 5-base baseline**

Run:
```bash
conda run -n aicup-tt --no-capture-output python -c "import json; print('baseline overall', json.load(open('artifacts/final_perrow_scores.json'))['overall'])"
```
Expected: `baseline overall 0.3205552079188966`. (This is the gate reference.)

- [ ] **Step 2: Add `cat` to each base list**

In `scripts/build_final_perrow.py`, change `BASES` to append `"cat"`:
```python
BASES = {
    "action": ["lgbm15", "lgbm31", "markov", "phase_lgbm", "chain_action", "cat"],
    "point": ["lgbm15", "lgbm31", "markov", "phase_lgbm", "chain_point", "cat"],
    "server": ["lgbm15", "lgbm31", "markov", "phase_lgbm", "chain_server", "cat"],
}
```

- [ ] **Step 3: Rebuild and read the honest lift**

Run:
```bash
conda run -n aicup-tt --no-capture-output python -m scripts.build_final_perrow 2>&1 | grep -E "action_macro|point_macro|server_auc|overall"
```
Expected: prints the new honest scores. Compute lift = new `overall` − 0.32056.

- [ ] **Step 4: Apply the gate (decision step)**

- If lift **> 0.00168**: CatBoost ships. Keep the `BASES` edit; proceed to Task 7 Step 1 (commit the integration).
- If lift **≤ 0.00168**: CatBoost is sub-noise. Revert the `BASES` edit and restore production artifacts:
  ```bash
  git checkout -- scripts/build_final_perrow.py artifacts/final_perrow_scores.json artifacts/submission_FINAL_safe_perrow.csv artifacts/submission_FINAL_smooth_perrow.csv
  ```
  Then skip to Task 7 Step 2 (record the negative result). Do NOT ship.

---

## Task 7: Record result + decide upload

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1 (only if shipped): Commit the integration**

```bash
git add scripts/build_final_perrow.py artifacts/final_perrow_scores.json artifacts/submission_FINAL_safe_perrow.csv artifacts/submission_FINAL_smooth_perrow.csv
git commit -m "feat(catboost): integrate into per-row ensemble (lift > noise floor)"
```

- [ ] **Step 2: Record the result in PROGRESS.md**

Add a `## CatBoost base result (Prong A)` section: CatBoost standalone honest scores (Task 4 Step 3), the ensemble overall with `cat` and the lift vs 0.32056, and the SHIP/REJECT verdict against the 0.00168 floor. If shipped, note the new production overall; if rejected, note artifacts/code kept for reproducibility (as done for seq2). Commit:
```bash
git add PROGRESS.md
git commit -m "docs(progress): record CatBoost base result (Prong A)"
```

- [ ] **Step 3: Decide on at most one public upload**

Only suggest uploading `submission_FINAL_safe_perrow.csv` if CatBoost shipped (honest lift > noise floor). Public is a confirmation/tie-breaker, never a model-selection oracle. Defer the actual upload to the user (daily-limited, teammate-shared).

---

## Self-review checklist (run before handing off)

- [ ] `conda run -n aicup-tt python -m pytest -q` — all tests green (existing 36 + `test_catboost_helpers` 2 = 38).
- [ ] No seed-averaging anywhere; honest per-row scoring only.
- [ ] CatBoost OOF key set is identical to the other bases (Task 4 Step 2) so the ensemble inner-join keeps 74,975 rows.
- [ ] OOF/test parquet `git add` uses `-f` (artifacts/oof/*.parquet is gitignored).
- [ ] Gate is explicit: ship only if ensemble lift > 0.00168; otherwise revert cleanly (Task 6 Step 4).
- [ ] CatBoost runs CPU (no `task_type='GPU'`).
```
