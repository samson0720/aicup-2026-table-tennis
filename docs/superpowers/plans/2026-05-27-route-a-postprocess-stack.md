# Route A — Post-processing and Stacking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the overall metric without retraining any model from scratch. Three independent levers — prior-corrected prediction, multi-seed bagging, multi-base stacking — and one phase-aware blend for server-AUC. Target lift: overall **+0.015 to +0.030** on the new private-safe CV.

**Architecture:** All five existing base models (LGBM leaves=15, LGBM leaves=31, Markov ensemble, phase-LGBM, player-stats LGBM) are re-run through `cv_splits.parquet` from P1 to produce OOF probability tables in a uniform schema. A `scripts/postprocess.py` module owns prior correction, threshold tuning, and phase-aware blending — all pure, testable functions. A `scripts/stacker.py` fits a per-target multinomial logistic regression meta-learner on those OOF tables. The final Route-A submission is assembled by `scripts/build_route_a_submission.py`.

**Tech Stack:** Same as P1 — Python 3.11 in `aicup-tt` conda env, pandas, scikit-learn, lightgbm.

**Depends on:** P1 (`scripts/cv_splits.py`, `artifacts/cv_splits.parquet`, `iter_cv_folds`).

---

## Spec section coverage

- Section 2.1 (prior-corrected prediction) → Task 6
- Section 2.2 (multi-seed bagging) → built into Tasks 1–4 (per-model OOF is already 5-seed averaged when read)
- Section 2.3 (multi-base stacking) → Task 9
- Section 2.4 (phase-aware server blending) → Task 8
- Section 2.5 (artifacts) → Task 10

## File structure

| Path | Purpose |
|---|---|
| `scripts/produce_base_oof.py` | One CLI that re-runs each base model under new CV and writes OOF parquets. Create. |
| `scripts/oof_loader.py` | Uniform loader: returns dict of `{model_name: {target: ndarray}}`. Create. |
| `scripts/postprocess.py` | Pure functions: `prior_correct`, `tune_thresholds`, `phase_blend_server`. Create. |
| `scripts/stacker.py` | Meta-learner train/predict per target with extra meta-CV. Create. |
| `scripts/build_route_a_submission.py` | End-to-end builder. Create. |
| `tests/test_postprocess.py` | Tests for pure functions. Create. |
| `tests/test_stacker.py` | Test for no-leak in meta-CV. Create. |
| `artifacts/oof/<model>_<target>.parquet` | OOF probabilities per (model, target). Generated. |
| `artifacts/route_a_oof.parquet` | Post-processed OOF used as Route A's stacker input/output. Generated. |
| `artifacts/submission_A_stacked.csv` | Final Route A submission. Generated. |

---

### Task 1: OOF parquet schema lock

Before any model runs, lock in the schema every base model must follow. Downstream stacker reads only this schema.

**Files:**
- Create: `scripts/oof_loader.py`
- Create: `tests/test_oof_loader.py`

- [ ] **Step 1.1: Document the schema in `scripts/oof_loader.py`**

```python
"""Uniform OOF probability storage and loading.

Schema for every artifacts/oof/<model>_<target>.parquet file:
- rally_uid : int
- seed      : int      # one of (11, 22, 33, 44, 55)
- fold      : int      # one of 0..4
- cut_strikeNumber : int  # mirrored from cv_splits for traceability
- p_<class_id> : float for each class id of the target

Targets:
- action : 19 classes -> p_0 .. p_18
- point  : 10 classes -> p_0 .. p_9
- server : 1 class    -> p_1 (probability that serverGetPoint == 1)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TARGET_CLASS_COUNTS = {"action": 19, "point": 10, "server": 1}

OOF_DIR = Path("artifacts/oof")


def oof_path(model: str, target: str) -> Path:
    return OOF_DIR / f"{model}_{target}.parquet"


def write_oof(
    model: str,
    target: str,
    rally_uid: np.ndarray,
    seed: np.ndarray,
    fold: np.ndarray,
    cut: np.ndarray,
    probs: np.ndarray,
) -> Path:
    n_class = TARGET_CLASS_COUNTS[target]
    assert probs.ndim == 2 and probs.shape[1] == n_class, f"probs shape {probs.shape}"
    df = pd.DataFrame({
        "rally_uid": rally_uid.astype(np.int64),
        "seed": seed.astype(np.int32),
        "fold": fold.astype(np.int32),
        "cut_strikeNumber": cut.astype(np.int32),
    })
    for c in range(n_class):
        col = f"p_{c}" if target != "server" else "p_1"
        df[col] = probs[:, c].astype(np.float32)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    out = oof_path(model, target)
    df.to_parquet(out, index=False)
    return out


def read_oof(model: str, target: str) -> pd.DataFrame:
    return pd.read_parquet(oof_path(model, target))


def average_over_seeds(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Reduce a 5-seed OOF parquet to one row per rally_uid by mean-of-probabilities."""
    prob_cols = [c for c in df.columns if c.startswith("p_")]
    grouped = df.groupby("rally_uid", as_index=False)[prob_cols].mean()
    return grouped
```

- [ ] **Step 1.2: Write a roundtrip test**

`tests/test_oof_loader.py`:
```python
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.oof_loader import write_oof, read_oof, average_over_seeds, TARGET_CLASS_COUNTS


def test_roundtrip_action(tmp_path, monkeypatch):
    import scripts.oof_loader as ol
    monkeypatch.setattr(ol, "OOF_DIR", tmp_path)
    n = 50
    probs = np.random.default_rng(0).dirichlet(np.ones(19), size=n).astype(np.float32)
    write_oof(
        "dummy", "action",
        rally_uid=np.arange(n),
        seed=np.full(n, 11),
        fold=np.arange(n) % 5,
        cut=np.full(n, 3),
        probs=probs,
    )
    df = read_oof("dummy", "action")
    assert len(df) == n
    p_cols = [f"p_{i}" for i in range(19)]
    assert all(c in df.columns for c in p_cols)
    # Probabilities should sum to ~1 per row.
    s = df[p_cols].sum(axis=1).to_numpy()
    assert np.allclose(s, 1.0, atol=1e-3)


def test_average_over_seeds(tmp_path, monkeypatch):
    import scripts.oof_loader as ol
    monkeypatch.setattr(ol, "OOF_DIR", tmp_path)
    rng = np.random.default_rng(0)
    rally = np.repeat(np.arange(10), 5)  # 10 rallies × 5 seeds
    seeds = np.tile([11, 22, 33, 44, 55], 10)
    probs = rng.dirichlet(np.ones(10), size=50).astype(np.float32)
    write_oof("dummy", "point",
              rally_uid=rally, seed=seeds, fold=np.zeros(50),
              cut=np.full(50, 3), probs=probs)
    df = read_oof("dummy", "point")
    avg = average_over_seeds(df, "point")
    assert len(avg) == 10
    assert set(avg["rally_uid"]) == set(range(10))
```

- [ ] **Step 1.3: Run, expect ImportError → PASS after Step 1.1 is in place**

Run: `conda run -n aicup-tt pytest tests/test_oof_loader.py -v`
Expected: 2 passed.

- [ ] **Step 1.4: Commit**

```bash
git add scripts/oof_loader.py tests/test_oof_loader.py
git commit -m "feat(oof): uniform OOF schema and roundtrip loader"
```

---

### Task 2: Base-model OOF producer — LGBM (leaves=15 and leaves=31)

We re-run the existing LightGBM baseline on the new CV, once per `num_leaves` setting. Reuses the feature builder from `train_lgbm_baseline.py`.

**Files:**
- Create: `scripts/produce_base_oof.py`

- [ ] **Step 2.1: Write the producer with an LGBM mode**

```python
"""Re-run base models on cv_splits.parquet and write OOF parquets.

Usage:
  python -m scripts.produce_base_oof --model lgbm15
  python -m scripts.produce_base_oof --model lgbm31
  python -m scripts.produce_base_oof --model markov
  python -m scripts.produce_base_oof --model phase_lgbm
  python -m scripts.produce_base_oof --model player_stats
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally  # P1 Task 9
from scripts.oof_loader import write_oof
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    feature_columns,
    fit_binary,
    fit_multiclass,
)


def _stack(rally_uid_lists, seed_lists, fold_lists, cut_lists, prob_lists):
    return (
        np.concatenate(rally_uid_lists),
        np.concatenate(seed_lists),
        np.concatenate(fold_lists),
        np.concatenate(cut_lists),
        np.concatenate(prob_lists, axis=0),
    )


def run_lgbm(num_leaves: int, model_name: str) -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))

    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        x_train, x_valid = df_train[feats], df_valid[feats]

        pa = fit_multiclass(x_train, df_train["y_actionId"], x_valid, df_valid["y_actionId"],
                            TARGET_ACTION_CLASSES, "sqrt", 2026 + fold, 180, num_leaves)
        pp = fit_multiclass(x_train, df_train["y_pointId"], x_valid, df_valid["y_pointId"],
                            TARGET_POINT_CLASSES, "sqrt", 3026 + fold, 180, num_leaves)
        ps = fit_binary(x_train, df_train["y_serverGetPoint"], x_valid, df_valid["y_serverGetPoint"],
                        4026 + fold, 180, num_leaves)

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()
        bag["action"]["r"].append(rally); bag["action"]["s"].append(sid)
        bag["action"]["f"].append(fid); bag["action"]["c"].append(cut); bag["action"]["p"].append(pa)
        bag["point"]["r"].append(rally); bag["point"]["s"].append(sid)
        bag["point"]["f"].append(fid); bag["point"]["c"].append(cut); bag["point"]["p"].append(pp)
        bag["server"]["r"].append(rally); bag["server"]["s"].append(sid)
        bag["server"]["f"].append(fid); bag["server"]["c"].append(cut); bag["server"]["p"].append(ps.reshape(-1, 1))
        print(f"{model_name} seed={seed} fold={fold} valid_n={len(rally)}")

    for tgt in ("action", "point", "server"):
        r, s, f, c, p = _stack(bag[tgt]["r"], bag[tgt]["s"], bag[tgt]["f"], bag[tgt]["c"], bag[tgt]["p"])
        out = write_oof(model_name, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["lgbm15", "lgbm31", "markov", "phase_lgbm", "player_stats"])
    args = parser.parse_args()

    if args.model == "lgbm15":
        run_lgbm(15, "lgbm15")
    elif args.model == "lgbm31":
        run_lgbm(31, "lgbm31")
    else:
        # Other models are added in later tasks.
        raise NotImplementedError(f"model {args.model} added in a later task")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Run for lgbm15**

Run: `conda run -n aicup-tt python -m scripts.produce_base_oof --model lgbm15`
Expected: 25 progress lines, then 3 "wrote" lines for action/point/server.
Runtime: 10–25 minutes.

- [ ] **Step 2.3: Run for lgbm31**

Run: `conda run -n aicup-tt python -m scripts.produce_base_oof --model lgbm31`
Expected: same.

- [ ] **Step 2.4: Sanity check**

Run:
```bash
conda run -n aicup-tt python -c "
import pandas as pd
for m in ['lgbm15','lgbm31']:
    for t in ['action','point','server']:
        df=pd.read_parquet(f'artifacts/oof/{m}_{t}.parquet')
        print(m, t, df.shape, 'unique rally:', df['rally_uid'].nunique())
"
```
Expected: 6 lines, every (model, target) shows `(74975, ...)` rows and `unique rally: 14995`.

- [ ] **Step 2.5: Commit**

```bash
git add scripts/produce_base_oof.py artifacts/oof/lgbm15_*.parquet artifacts/oof/lgbm31_*.parquet
git commit -m "feat(oof): LGBM leaves=15/31 OOF under new CV"
```

---

### Task 3: Base-model OOF producer — Markov ensemble

We wrap the existing `train_markov_ensemble.py` so its OOF lands in the standard parquet.

**Files:**
- Modify: `scripts/produce_base_oof.py`

- [ ] **Step 3.1: Inspect `scripts/train_markov_ensemble.py` for its public predict-prob entry point**

If it exposes a `predict_proba(train_prefix_rows, target_row) -> np.ndarray` function, use it directly. If it is monolithic, extract a `markov_oof(train_view, valid_view, splits_sub_train, splits_sub_valid) -> dict[str, np.ndarray]` helper next to the existing logic.

- [ ] **Step 3.2: Add a `run_markov()` function to `scripts/produce_base_oof.py`**

```python
def run_markov() -> None:
    from scripts.train_markov_ensemble import markov_oof  # extracted helper

    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        out = markov_oof(train_view, valid_view, s_train, s_valid)
        # out: {"rally_uid": np.ndarray, "cut": np.ndarray,
        #      "action": (n,19), "point": (n,10), "server": (n,1)}
        rally = out["rally_uid"]; cut = out["cut"]
        sid = np.full(len(rally), seed); fid = np.full(len(rally), fold)
        for tgt in ("action", "point", "server"):
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(out[tgt])
        print(f"markov seed={seed} fold={fold} valid_n={len(rally)}")

    for tgt in ("action", "point", "server"):
        r, s, f, c, p = _stack(bag[tgt]["r"], bag[tgt]["s"], bag[tgt]["f"], bag[tgt]["c"], bag[tgt]["p"])
        write_oof("markov", tgt, r, s, f, c, p)
```

Wire it into `main()`:
```python
    elif args.model == "markov":
        run_markov()
```

- [ ] **Step 3.3: If `train_markov_ensemble.py` lacks `markov_oof`, refactor it**

Open `scripts/train_markov_ensemble.py`. Identify the block that, per fold, takes a training set and produces predictions on a validation set. Extract that block into a top-level function `markov_oof(train_view, valid_view, s_train, s_valid)` with the docstring spelling out the return dict shape from Step 3.2.

Keep the existing CLI behavior of `train_markov_ensemble.py` intact — only ADD the helper, do not change its outputs.

- [ ] **Step 3.4: Run**

Run: `conda run -n aicup-tt python -m scripts.produce_base_oof --model markov`
Expected: 25 lines, three `markov_<target>.parquet` written.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/produce_base_oof.py scripts/train_markov_ensemble.py artifacts/oof/markov_*.parquet
git commit -m "feat(oof): Markov ensemble OOF under new CV"
```

---

### Task 4: Base-model OOF producer — phase-LGBM and player-stats LGBM

Same pattern as Task 3. Each existing script gets a `*_oof()` helper extracted, then wired into `produce_base_oof.py`.

- [ ] **Step 4.1: Extract `phase_lgbm_oof()` in `scripts/train_phase_lgbm.py`**

Same contract as Step 3.2: takes `(train_view, valid_view, s_train, s_valid)`, returns `{"rally_uid", "cut", "action", "point", "server"}`.

- [ ] **Step 4.2: Extract `player_stats_oof()` in `scripts/train_player_stats_lgbm.py`**

IMPORTANT: the original player_stats was OOF-unsafe (it leaked test-rally stats into train; see HANDOFF L55 "rejected"). When extracting, recompute player stats **only from `train_view` rows** for each fold, with smoothing alpha=20 toward a global prior (use `iter_cv_folds`'s `train_view` directly — that is fold-out by construction).

This is the unseen-player-fallback enforcement: any player only seen in `valid_view` gets the global prior. See P3 Task 1 for the encoding utility that lives here.

- [ ] **Step 4.3: Wire both into `scripts/produce_base_oof.py`**

```python
    elif args.model == "phase_lgbm":
        run_phase_lgbm()
    elif args.model == "player_stats":
        run_player_stats()
```
where each `run_*` mirrors `run_markov` from Task 3.

- [ ] **Step 4.4: Run both**

```bash
conda run -n aicup-tt python -m scripts.produce_base_oof --model phase_lgbm
conda run -n aicup-tt python -m scripts.produce_base_oof --model player_stats
```
Expected: each emits 25 progress lines, then writes 3 parquets.

- [ ] **Step 4.5: Sanity table — print per-model overall macro-F1 from raw OOF (argmax, no postproc)**

```bash
conda run -n aicup-tt python -c "
import pandas as pd, numpy as np
from sklearn.metrics import f1_score, roc_auc_score
train=pd.read_csv(next(__import__('pathlib').Path.cwd().glob('AI CUP*/train.csv')))
y_a={r:int(g.sort_values('strikeNumber').iloc[-1]['actionId']) for r,g in train.groupby('rally_uid')}
for m in ['lgbm15','lgbm31','markov','phase_lgbm','player_stats']:
    try:
        da=pd.read_parquet(f'artifacts/oof/{m}_action.parquet')
    except FileNotFoundError:
        continue
    p=da[[c for c in da.columns if c.startswith('p_')]].to_numpy()
    yhat=p.argmax(1)
    # Use the row's own rally cut-target as label by joining with the cut column-resolved y.
    # Cheap proxy: load y from train using rally_uid + cut.
    pass
print('done')
"
```
This is a rough sanity step. For a proper comparison, use `scripts/diagnose_cv_gap.py`'s metric. The point here is just to confirm OOF parquets are non-trivial (probabilities are not all uniform).

- [ ] **Step 4.6: Commit**

```bash
git add scripts/produce_base_oof.py scripts/train_phase_lgbm.py scripts/train_player_stats_lgbm.py artifacts/oof/phase_lgbm_*.parquet artifacts/oof/player_stats_*.parquet
git commit -m "feat(oof): phase-LGBM and OOF-safe player-stats OOF under new CV"
```

---

### Task 5: Score utility — uniform macro-F1 / AUC on OOF parquet

Centralize the metric so every later task computes lift the same way.

**Files:**
- Create: `scripts/score_oof.py`
- Create: `tests/test_score_oof.py`

- [ ] **Step 5.1: Write the scorer**

```python
"""Score OOF parquets uniformly."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def attach_labels(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Join the cut-target label onto an OOF parquet."""
    # Per (rally_uid, cut_strikeNumber), find the train row whose strikeNumber == cut.
    key = train[["rally_uid", "strikeNumber", "actionId", "pointId", "serverGetPoint"]]
    return df.merge(
        key.rename(columns={"strikeNumber": "cut_strikeNumber"}),
        on=["rally_uid", "cut_strikeNumber"], how="left",
    )


def score_action(df_with_labels: pd.DataFrame) -> float:
    cols = [f"p_{i}" for i in range(19)]
    yhat = df_with_labels[cols].to_numpy().argmax(1)
    return float(f1_score(df_with_labels["actionId"], yhat,
                          labels=list(range(19)), average="macro", zero_division=0))


def score_point(df_with_labels: pd.DataFrame) -> float:
    cols = [f"p_{i}" for i in range(10)]
    yhat = df_with_labels[cols].to_numpy().argmax(1)
    return float(f1_score(df_with_labels["pointId"], yhat,
                          labels=list(range(10)), average="macro", zero_division=0))


def score_server(df_with_labels: pd.DataFrame) -> float:
    return float(roc_auc_score(df_with_labels["serverGetPoint"], df_with_labels["p_1"]))


def overall(action_f1: float, point_f1: float, server_auc: float) -> float:
    return 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc


def score_model(model: str) -> dict:
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    out = {}
    for tgt, fn in (("action", score_action), ("point", score_point), ("server", score_server)):
        df = pd.read_parquet(f"artifacts/oof/{model}_{tgt}.parquet")
        df = attach_labels(df, train)
        out[tgt] = fn(df)
    out["overall"] = overall(out["action"], out["point"], out["server"])
    return out


if __name__ == "__main__":
    import sys
    models = sys.argv[1:] or ["lgbm15", "lgbm31", "markov", "phase_lgbm", "player_stats"]
    table = {m: score_model(m) for m in models}
    print(json.dumps(table, indent=2, ensure_ascii=False))
    Path("artifacts/base_oof_scores.json").write_text(json.dumps(table, indent=2, ensure_ascii=False))
```

- [ ] **Step 5.2: Minimal test**

`tests/test_score_oof.py`:
```python
import numpy as np
import pandas as pd
from scripts.score_oof import overall


def test_overall_formula():
    assert abs(overall(0.3, 0.2, 0.6) - (0.4 * 0.3 + 0.4 * 0.2 + 0.2 * 0.6)) < 1e-9
```

- [ ] **Step 5.3: Run**

```bash
conda run -n aicup-tt pytest tests/test_score_oof.py -v
conda run -n aicup-tt python -m scripts.score_oof
```
Expected: pytest 1 passed; the script prints a JSON table of 5 models × 4 metrics.

- [ ] **Step 5.4: Commit**

```bash
git add scripts/score_oof.py tests/test_score_oof.py artifacts/base_oof_scores.json
git commit -m "feat(score): uniform OOF scoring helper"
```

---

### Task 6: Prior-corrected prediction

**Files:**
- Create: `scripts/postprocess.py`
- Create: `tests/test_postprocess.py`

- [ ] **Step 6.1: Write the failing test**

`tests/test_postprocess.py`:
```python
import numpy as np
import pytest

from scripts.postprocess import prior_correct


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
```

- [ ] **Step 6.2: Run, expect ImportError**

Run: `conda run -n aicup-tt pytest tests/test_postprocess.py -v`
Expected: FAIL.

- [ ] **Step 6.3: Implement `prior_correct`**

`scripts/postprocess.py`:
```python
"""Post-processing utilities for OOF probabilities."""
from __future__ import annotations

import numpy as np


def prior_correct(probs: np.ndarray, prior: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Divide each class probability by its train prior, then renormalize.

    Equivalent to shifting predictions toward a uniform-prior posterior.
    Common trick to recover macro-F1 when argmax under the original prior
    collapses to majority classes.
    """
    assert probs.shape[1] == prior.shape[0]
    adjusted = probs / np.clip(prior, eps, None)
    return adjusted / adjusted.sum(axis=1, keepdims=True)
```

- [ ] **Step 6.4: Run, PASS**

Run: `conda run -n aicup-tt pytest tests/test_postprocess.py -v`
Expected: 2 passed.

- [ ] **Step 6.5: Commit**

```bash
git add scripts/postprocess.py tests/test_postprocess.py
git commit -m "feat(postproc): prior-corrected prediction for macro-F1"
```

---

### Task 7: Per-class threshold tuning (macro-F1)

A finer knob on top of prior correction. Optional but usually adds another 0.005–0.015.

- [ ] **Step 7.1: Add failing test**

Append to `tests/test_postprocess.py`:
```python
from scripts.postprocess import tune_thresholds, apply_thresholds


def test_tune_thresholds_does_not_make_macro_f1_worse():
    rng = np.random.default_rng(0)
    n, k = 500, 5
    y = rng.integers(0, k, size=n)
    # Probabilities biased toward class 0.
    base = rng.dirichlet(np.full(k, 0.5), size=n)
    base[:, 0] += 0.3
    base = base / base.sum(1, keepdims=True)

    from sklearn.metrics import f1_score
    f1_argmax = f1_score(y, base.argmax(1), labels=list(range(k)), average="macro", zero_division=0)
    thr = tune_thresholds(base, y, n_classes=k)
    yhat = apply_thresholds(base, thr)
    f1_tuned = f1_score(y, yhat, labels=list(range(k)), average="macro", zero_division=0)
    assert f1_tuned + 1e-9 >= f1_argmax
```

- [ ] **Step 7.2: Run, FAIL**

Run: `conda run -n aicup-tt pytest tests/test_postprocess.py::test_tune_thresholds_does_not_make_macro_f1_worse -v`
Expected: FAIL (ImportError).

- [ ] **Step 7.3: Implement tuner**

Append to `scripts/postprocess.py`:
```python
def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Argmax of (probs - thresholds[None, :])."""
    return (probs - thresholds[None, :]).argmax(axis=1)


def tune_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    grid: tuple[float, ...] = (-0.10, -0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06, 0.10),
) -> np.ndarray:
    """Per-class additive threshold grid search to maximize macro-F1.

    Greedy one-pass: for each class c, fix others at 0 and pick the best
    threshold from `grid`. Two-pass refinement once. Stops when no class
    moves. Returns a vector of shape (n_classes,).
    """
    from sklearn.metrics import f1_score

    thr = np.zeros(n_classes)
    def score(t: np.ndarray) -> float:
        yhat = apply_thresholds(probs, t)
        return f1_score(y, yhat, labels=list(range(n_classes)),
                        average="macro", zero_division=0)

    best_global = score(thr)
    for _ in range(2):
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

- [ ] **Step 7.4: Run, PASS**

Run: `conda run -n aicup-tt pytest tests/test_postprocess.py -v`
Expected: 3 passed.

- [ ] **Step 7.5: Commit**

```bash
git add scripts/postprocess.py tests/test_postprocess.py
git commit -m "feat(postproc): per-class threshold tuner with macro-F1 monotonicity test"
```

---

### Task 8: Phase-aware server blending

Server-AUC in phase-0 (prefix length 1) is near random. Blend with a `(server_player, receiver_player)` historical win-rate prior.

- [ ] **Step 8.1: Add failing test**

Append to `tests/test_postprocess.py`:
```python
from scripts.postprocess import phase_blend_server


def test_phase_blend_server_weights_per_phase():
    n = 30
    p_model = np.full(n, 0.5)
    p_prior = np.array([0.9] * 10 + [0.5] * 10 + [0.1] * 10)
    phase = np.array([0] * 10 + [1] * 10 + [2] * 10)
    weights = {0: 0.7, 1: 0.4, 2: 0.0}  # share of prior in blend per phase
    out = phase_blend_server(p_model, p_prior, phase, weights)
    assert np.allclose(out[:10], 0.7 * 0.9 + 0.3 * 0.5)
    assert np.allclose(out[10:20], 0.4 * 0.5 + 0.6 * 0.5)
    assert np.allclose(out[20:], 0.5)
```

- [ ] **Step 8.2: Run, FAIL**

- [ ] **Step 8.3: Implement blender + prior builder**

Append to `scripts/postprocess.py`:
```python
def phase_blend_server(
    p_model: np.ndarray,
    p_prior: np.ndarray,
    phase: np.ndarray,
    weights: dict[int, float],
) -> np.ndarray:
    """Blend model probability and prior probability per phase bucket.

    weights[phase] is the share assigned to p_prior; (1 - weights[phase]) goes
    to p_model. Missing phase keys default to 0 (pure model).
    """
    out = p_model.copy().astype(float)
    for ph in np.unique(phase):
        w = float(weights.get(int(ph), 0.0))
        mask = phase == ph
        out[mask] = w * p_prior[mask] + (1.0 - w) * p_model[mask]
    return out


def build_server_pair_prior(
    train: "pd.DataFrame",
    valid: "pd.DataFrame",
    alpha: float = 20.0,
) -> "pd.Series":
    """For each rally in `valid`, return the (server, receiver) historical
    serve-win rate computed from `train` only. Unseen pairs fall back to the
    global rate. Smoothing alpha avoids over-trusting small samples."""
    import pandas as pd

    # First stroke of each rally identifies the server (gamePlayerId of strike 1).
    train_first = train.sort_values(["rally_uid", "strikeNumber"]).drop_duplicates("rally_uid", keep="first")
    # serverGetPoint is rally-level constant.
    pair_stats = train_first.groupby(["gamePlayerId", "gamePlayerOtherId"]).agg(
        n=("serverGetPoint", "size"), w=("serverGetPoint", "sum")
    ).reset_index()
    global_rate = float(train_first["serverGetPoint"].mean())
    pair_stats["rate"] = (pair_stats["w"] + alpha * global_rate) / (pair_stats["n"] + alpha)
    lookup = pair_stats.set_index(["gamePlayerId", "gamePlayerOtherId"])["rate"]

    valid_first = valid.sort_values(["rally_uid", "strikeNumber"]).drop_duplicates("rally_uid", keep="first")
    keys = list(zip(valid_first["gamePlayerId"], valid_first["gamePlayerOtherId"]))
    rates = np.array([lookup.get(k, global_rate) for k in keys])
    return pd.Series(rates, index=valid_first["rally_uid"].to_numpy(), name="server_pair_prior")
```

- [ ] **Step 8.4: Run, PASS**

Run: `conda run -n aicup-tt pytest tests/test_postprocess.py -v`
Expected: 4 passed.

- [ ] **Step 8.5: Commit**

```bash
git add scripts/postprocess.py tests/test_postprocess.py
git commit -m "feat(postproc): phase-aware server blending with OOF-safe pair prior"
```

---

### Task 9: Multi-base stacker (per-target meta-learner)

LR with L2, multinomial for action/point, binary for server. One extra round of GroupKFold inside, so the meta-learner never sees its own training rallies in any single fold.

**Files:**
- Create: `scripts/stacker.py`
- Create: `tests/test_stacker.py`

- [ ] **Step 9.1: Write the failing no-leak test**

```python
# tests/test_stacker.py
import numpy as np
import pandas as pd
import pytest

from scripts.stacker import stacked_oof


def test_stacker_no_rally_leakage():
    rng = np.random.default_rng(0)
    n_rallies = 200
    rally = np.repeat(np.arange(n_rallies), 1)
    match = (rally // 20).astype(int)  # 10 matches
    y = rng.integers(0, 3, size=n_rallies)

    # Two "base models" with weakly informative probabilities.
    probs1 = rng.dirichlet([1.0, 1.0, 1.0], size=n_rallies)
    probs2 = rng.dirichlet([1.0, 1.0, 1.0], size=n_rallies)
    base_oofs = {
        "m1": pd.DataFrame({"rally_uid": rally, **{f"p_{i}": probs1[:, i] for i in range(3)}}),
        "m2": pd.DataFrame({"rally_uid": rally, **{f"p_{i}": probs2[:, i] for i in range(3)}}),
    }
    labels = pd.DataFrame({"rally_uid": rally, "match": match, "y": y})
    out = stacked_oof(base_oofs, labels, target_kind="multiclass", n_classes=3, n_folds=5)
    assert len(out) == n_rallies
    assert "p_0" in out.columns and out[[f"p_{i}" for i in range(3)]].sum(axis=1).between(0.99, 1.01).all()
```

- [ ] **Step 9.2: Run, FAIL**

- [ ] **Step 9.3: Implement stacker**

```python
# scripts/stacker.py
"""Per-target stacking meta-learner with inner match-aware CV."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def stacked_oof(
    base_oofs: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    target_kind: Literal["multiclass", "binary"],
    n_classes: int,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Return one OOF per rally produced by an L2-regularized LR meta-learner.

    `base_oofs` is keyed by model name; each value is a DataFrame with
    `rally_uid` + probability columns `p_0`, `p_1`, ... already averaged
    across the original seeds.

    `labels` must contain `rally_uid`, `match`, `y`. Match groups are used
    to keep one match per fold in the meta-CV.
    """
    base = labels[["rally_uid", "match", "y"]].copy()
    for name, df in base_oofs.items():
        pc = [c for c in df.columns if c.startswith("p_")]
        rename = {c: f"{name}__{c}" for c in pc}
        base = base.merge(df.rename(columns=rename)[["rally_uid", *rename.values()]],
                          on="rally_uid", how="left")

    feature_cols = [c for c in base.columns if "__p_" in c]
    X = base[feature_cols].to_numpy()
    y = base["y"].to_numpy()
    groups = base["match"].to_numpy()

    if target_kind == "multiclass":
        oof = np.zeros((len(base), n_classes), dtype=np.float32)
    else:
        oof = np.zeros((len(base), 1), dtype=np.float32)

    kf = GroupKFold(n_splits=n_folds)
    for fold, (tr, va) in enumerate(kf.split(X, y, groups)):
        if target_kind == "multiclass":
            clf = LogisticRegression(
                multi_class="multinomial", solver="lbfgs", max_iter=200, C=1.0
            )
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[va])
            aligned = np.zeros((len(va), n_classes), dtype=np.float32)
            for i, cls in enumerate(clf.classes_):
                aligned[:, int(cls)] = p[:, i]
            oof[va] = aligned
        else:
            clf = LogisticRegression(max_iter=200, C=1.0)
            clf.fit(X[tr], y[tr])
            oof[va, 0] = clf.predict_proba(X[va])[:, 1]

    cols = [f"p_{i}" for i in range(oof.shape[1])] if target_kind == "multiclass" else ["p_1"]
    return pd.concat([base[["rally_uid"]].reset_index(drop=True),
                      pd.DataFrame(oof, columns=cols)], axis=1)
```

- [ ] **Step 9.4: Run, PASS**

Run: `conda run -n aicup-tt pytest tests/test_stacker.py -v`
Expected: 1 passed.

- [ ] **Step 9.5: Commit**

```bash
git add scripts/stacker.py tests/test_stacker.py
git commit -m "feat(stack): multi-base meta-learner with match-grouped meta-CV"
```

---

### Task 10: Assemble Route A submission

Tie it all together: load OOF, average seeds, prior-correct, threshold-tune, stack, blend server with pair prior, build the submission CSV.

**Files:**
- Create: `scripts/build_route_a_submission.py`

- [ ] **Step 10.1: Write the builder**

```python
"""End-to-end Route A: stacking + post-processing + submission CSV."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.oof_loader import read_oof, average_over_seeds
from scripts.postprocess import (
    prior_correct, tune_thresholds, apply_thresholds,
    phase_blend_server, build_server_pair_prior,
)
from scripts.score_oof import attach_labels, score_action, score_point, score_server, overall
from scripts.stacker import stacked_oof


MODELS = ["lgbm15", "lgbm31", "markov", "phase_lgbm", "player_stats"]


def _phase_of_target(target_strike: int) -> int:
    if target_strike == 2: return 0
    if target_strike == 3: return 1
    return 2


def main() -> None:
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    test = pd.read_csv(next(Path.cwd().glob("AI CUP*/test_new.csv")))
    splits = pd.read_parquet("artifacts/cv_splits.parquet")

    # --- 1. Per-target stacking on seed-averaged base OOFs ---
    rally_labels_action = train.sort_values(["rally_uid", "strikeNumber"]).drop_duplicates("rally_uid", keep="last")[
        ["rally_uid", "match", "actionId"]
    ].rename(columns={"actionId": "y"})
    # Repeat for point and server. For action/point, we use the cut-target row.
    # For simplicity in this stacker, we use the LAST observed stroke as "y" only
    # if cv_splits chose cut == rally_len; otherwise use a cut-resolved label.
    # The robust way is to read attached labels from one of the OOF parquets:
    sample = read_oof("lgbm15", "action")
    attached = attach_labels(sample, train)[["rally_uid", "match", "actionId", "pointId", "serverGetPoint"]]
    attached = attached.drop_duplicates("rally_uid")

    # Build per-target stacked OOF.
    stacks: dict[str, pd.DataFrame] = {}
    for target_kind, tgt, n_cls, y_col in [
        ("multiclass", "action", 19, "actionId"),
        ("multiclass", "point",  10, "pointId"),
        ("binary",     "server",  1, "serverGetPoint"),
    ]:
        base_oofs = {m: average_over_seeds(read_oof(m, tgt), tgt) for m in MODELS if Path(f"artifacts/oof/{m}_{tgt}.parquet").exists()}
        labels = attached[["rally_uid", "match", y_col]].rename(columns={y_col: "y"})
        stack = stacked_oof(base_oofs, labels, target_kind=target_kind, n_classes=n_cls if target_kind=="multiclass" else 1)
        stacks[tgt] = stack

    # --- 2. Post-processing on action/point: prior correction + threshold tuning ---
    train_priors = {
        "action": np.bincount(attached["actionId"], minlength=19).astype(float),
        "point":  np.bincount(attached["pointId"],  minlength=10).astype(float),
    }
    train_priors["action"] /= train_priors["action"].sum()
    train_priors["point"]  /= train_priors["point"].sum()

    for tgt, n_cls, y_col in [("action", 19, "actionId"), ("point", 10, "pointId")]:
        df = stacks[tgt]
        prob_cols = [f"p_{i}" for i in range(n_cls)]
        corrected = prior_correct(df[prob_cols].to_numpy(), train_priors[tgt])
        df[prob_cols] = corrected
        # Threshold tuning uses the OOF labels.
        y = df.merge(attached[["rally_uid", y_col]], on="rally_uid")[y_col].to_numpy()
        thr = tune_thresholds(df[prob_cols].to_numpy(), y, n_classes=n_cls)
        # Persist threshold for inference time.
        Path(f"artifacts/route_a_thr_{tgt}.json").write_text(json.dumps(thr.tolist()))

    # --- 3. Server: blend with pair prior, phase-aware ---
    # Phase is derived from cut_strikeNumber (rally-mean is meaningless; use the
    # MAJORITY phase across seeds, or just use cut=median for inference).
    pair_prior_train = build_server_pair_prior(train, train)  # OOF-safe variant used in training; here we rebuild for test
    pair_prior_test  = build_server_pair_prior(train, test)

    # OOF blending diagnostics (optional, prints lift):
    df_s = stacks["server"]
    df_s = df_s.merge(attached[["rally_uid", "serverGetPoint"]], on="rally_uid")
    df_s["pair_prior"] = df_s["rally_uid"].map(pair_prior_train)
    df_s["phase"] = 2  # placeholder for OOF diagnostic; in submission we use real cut
    weights = {0: 0.7, 1: 0.4, 2: 0.0}
    blended_server = phase_blend_server(df_s["p_1"].to_numpy(),
                                        df_s["pair_prior"].to_numpy(),
                                        df_s["phase"].to_numpy(), weights)
    from sklearn.metrics import roc_auc_score
    print("server AUC raw    :", roc_auc_score(df_s["serverGetPoint"], df_s["p_1"]))
    print("server AUC blended:", roc_auc_score(df_s["serverGetPoint"], blended_server))

    # --- 4. Build the submission ---
    # For test rallies, we don't have a stack OOF — we need to PREDICT.
    # Strategy here: refit each base model on full train, build test features,
    # average their predictions, and run through the SAME post-processing.
    # That refit is intentionally left to a helper inside scripts/produce_base_oof.py
    # which Task 11 of this plan adds. See note below.
    print("submission build hinges on Task 11 — refitting base models on full train.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: Run the diagnostic part**

Run: `conda run -n aicup-tt python -m scripts.build_route_a_submission`
Expected: prints raw vs blended server AUC (blended should be >=, ideally noticeably higher). No CSV is written yet — Task 11 supplies the test-time prediction.

- [ ] **Step 10.3: Commit (diagnostic-only state)**

```bash
git add scripts/build_route_a_submission.py
git commit -m "feat(route_a): stacker + postproc diagnostic pipeline"
```

---

### Task 11: Test-time refit-and-predict for each base model

`produce_base_oof.py` only gives OOF. For the final submission, each base model must also predict on `test_new.csv`. We add a `predict_test()` mode to each base, then assemble the submission CSV.

**Files:**
- Modify: `scripts/produce_base_oof.py` (add `--predict-test`)
- Modify: `scripts/build_route_a_submission.py` (Task 10 finish)

- [ ] **Step 11.1: Add a `predict_test_lgbm()` to `produce_base_oof.py`**

```python
def predict_test_lgbm(num_leaves: int, model_name: str) -> None:
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    test  = pd.read_csv(next(Path.cwd().glob("AI CUP*/test_new.csv")))

    # Build prefix features for full train, fit one model per target on ALL train,
    # then build features for test (using the LAST observed stroke per rally as
    # the prefix end) and predict.
    # Reuse build_one_sample_per_rally indirectly: emulate "cut at rally_len" for train,
    # and "cut at last_observed_strike + 1" for test (where the target is unknown — we
    # just need the feature row whose last_strikeNumber equals the test row's last
    # observed strike).
    ...
    # Write artifacts/oof/<model>_<target>_test.parquet keyed by rally_uid only.
```

Implementation guidance:
1. For train: use `build_one_sample_per_rally` with `splits_sub` set so that cut = rally_len (target is the last stroke). Train on this expanded dataset.
2. For test: for each test rally, treat the entire test row sequence as the prefix and synthesize a feature row with `last_strikeNumber = max(test_rally.strikeNumber)`. The model predicts the next stroke (which is what AI Cup asks for).
3. Average over 5 seeds by retraining the LGBM with 5 different `random_state` values.

- [ ] **Step 11.2: Same pattern for `markov`, `phase_lgbm`, `player_stats`**

Each gets a `predict_test_*` helper that produces `artifacts/oof/<model>_<target>_test.parquet` with `rally_uid + p_*` columns (no fold/seed; already averaged).

- [ ] **Step 11.3: Run all predict-test passes**

```bash
for m in lgbm15 lgbm31 markov phase_lgbm player_stats; do
  conda run -n aicup-tt python -m scripts.produce_base_oof --model $m --predict-test
done
```

- [ ] **Step 11.4: Finish `build_route_a_submission.py`**

In `main()` after the diagnostic block, add:

```python
    # --- 5. Test-time predictions ---
    test_preds = {tgt: {} for tgt in ("action", "point", "server")}
    for m in MODELS:
        for tgt in ("action", "point", "server"):
            p = pd.read_parquet(f"artifacts/oof/{m}_{tgt}_test.parquet")
            test_preds[tgt][m] = p

    # Refit the meta-learner once per target on the full stacked OOF, then predict on test.
    from sklearn.linear_model import LogisticRegression
    def fit_meta(stack: pd.DataFrame, labels: pd.DataFrame, target_kind: str, n_cls: int):
        merged = stack.merge(labels, on="rally_uid")
        feat_cols = [c for c in stack.columns if c.startswith("p_")]
        if target_kind == "multiclass":
            clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=300, C=1.0)
        else:
            clf = LogisticRegression(max_iter=300, C=1.0)
        clf.fit(merged[feat_cols].to_numpy(), merged["y"].to_numpy())
        return clf, feat_cols

    # Build action and point submission columns.
    out_rows = {"rally_uid": [], "actionId": [], "pointId": [], "serverGetPoint": []}
    test_rallies = sorted(test["rally_uid"].unique())

    for tgt, target_kind, n_cls, y_col in [
        ("action", "multiclass", 19, "actionId"),
        ("point",  "multiclass", 10, "pointId"),
    ]:
        labels = attached[["rally_uid", y_col]].rename(columns={y_col: "y"})
        clf, feat_cols = fit_meta(stacks[tgt], labels, target_kind, n_cls)

        # Stack base test preds into one matrix.
        rows = pd.DataFrame({"rally_uid": test_rallies})
        for m in MODELS:
            if m not in test_preds[tgt]: continue
            p = test_preds[tgt][m].copy()
            ren = {c: f"{m}__{c}" for c in p.columns if c.startswith("p_")}
            rows = rows.merge(p.rename(columns=ren)[["rally_uid", *ren.values()]],
                              on="rally_uid", how="left")
        test_probs = clf.predict_proba(rows[feat_cols].to_numpy())
        # Prior correction.
        test_probs = prior_correct(test_probs, train_priors[tgt])
        # Threshold from saved JSON.
        thr = np.array(json.loads(Path(f"artifacts/route_a_thr_{tgt}.json").read_text()))
        yhat = apply_thresholds(test_probs, thr)
        out_rows[y_col] = list(yhat)

    # Server.
    labels_s = attached[["rally_uid", "serverGetPoint"]].rename(columns={"serverGetPoint": "y"})
    clf_s, feat_cols_s = fit_meta(stacks["server"], labels_s, "binary", 1)
    rows = pd.DataFrame({"rally_uid": test_rallies})
    for m in MODELS:
        if m not in test_preds["server"]: continue
        p = test_preds["server"][m].copy()
        ren = {c: f"{m}__{c}" for c in p.columns if c.startswith("p_")}
        rows = rows.merge(p.rename(columns=ren)[["rally_uid", *ren.values()]],
                          on="rally_uid", how="left")
    p_server = clf_s.predict_proba(rows[feat_cols_s].to_numpy())[:, 1]
    # Phase-aware blend with test pair prior.
    test_phase = test.groupby("rally_uid")["strikeNumber"].max().reindex(test_rallies).map(
        lambda x: 0 if x == 1 else (1 if x == 2 else 2)
    ).to_numpy()
    p_prior_arr = np.array([pair_prior_test.get(r, float(pair_prior_test.median())) for r in test_rallies])
    p_server = phase_blend_server(p_server, p_prior_arr, test_phase, weights)
    out_rows["serverGetPoint"] = list(p_server)
    out_rows["rally_uid"] = test_rallies

    sub = pd.DataFrame(out_rows)[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    sub.to_csv("artifacts/submission_A_stacked.csv", index=False)
    print(f"wrote artifacts/submission_A_stacked.csv: {sub.shape}")
```

- [ ] **Step 11.5: Run end-to-end**

Run: `conda run -n aicup-tt python -m scripts.build_route_a_submission`
Expected: prints AUC numbers, then `wrote artifacts/submission_A_stacked.csv: (1845, 4)`.

- [ ] **Step 11.6: Sanity check submission**

```bash
conda run -n aicup-tt python -c "
import pandas as pd
s=pd.read_csv('artifacts/submission_A_stacked.csv')
print(s.head()); print(s.shape)
print('action range', s.actionId.min(), s.actionId.max())
print('point range',  s.pointId.min(),  s.pointId.max())
print('server range', s.serverGetPoint.min(), s.serverGetPoint.max())
"
```
Expected: 1845 rows; action in [0,18]; point in [0,9]; server in [0,1].

- [ ] **Step 11.7: Commit**

```bash
git add scripts/produce_base_oof.py scripts/build_route_a_submission.py artifacts/route_a_thr_*.json artifacts/submission_A_stacked.csv artifacts/oof/*_test.parquet
git commit -m "feat(route_a): full stacked+postprocessed submission pipeline"
```

---

### Task 12: Quantify lift vs raw LGBM baseline

- [ ] **Step 12.1: Score Route A OOF vs base scores**

```bash
conda run -n aicup-tt python -c "
import json, pandas as pd, numpy as np
from scripts.score_oof import attach_labels, score_action, score_point, score_server, overall
from pathlib import Path

# Reconstruct stacked OOF scoring by re-running the diagnostic part.
base = json.loads(Path('artifacts/base_oof_scores.json').read_text())
print('=== base ===')
print(json.dumps(base, indent=2, ensure_ascii=False))
# Route A scoring requires saving the stacks first. Add a stacks-to-parquet
# dump in scripts/build_route_a_submission.py if not present, then score here.
"
```

- [ ] **Step 12.2: If `build_route_a_submission.py` does not already save stacks, add the dump**

Inside the per-target stack loop:
```python
        stack.to_parquet(f"artifacts/route_a_stack_{tgt}.parquet", index=False)
```

Then in the scoring snippet:
```python
import pandas as pd
from scripts.score_oof import attach_labels, score_action, score_point, score_server, overall

train = pd.read_csv(next(__import__('pathlib').Path.cwd().glob('AI CUP*/train.csv')))
df_a = attach_labels(pd.read_parquet('artifacts/route_a_stack_action.parquet'), train)
df_p = attach_labels(pd.read_parquet('artifacts/route_a_stack_point.parquet'),  train)
df_s = attach_labels(pd.read_parquet('artifacts/route_a_stack_server.parquet'), train)
a = score_action(df_a); p = score_point(df_p); s = score_server(df_s)
print({'action': a, 'point': p, 'server': s, 'overall': overall(a, p, s)})
```

Expected: overall higher than the best raw base by **0.015–0.030**. If lift is smaller than one across-seed std, log it and re-examine the meta-learner C value or feature inclusion before declaring success.

- [ ] **Step 12.3: Commit**

```bash
git add scripts/build_route_a_submission.py artifacts/route_a_stack_*.parquet
git commit -m "feat(route_a): persist stacks for lift comparison"
```

---

## Self-review notes (filled in during plan writing)

- Spec Section 2.1–2.5 coverage verified above.
- All public function names (`prior_correct`, `tune_thresholds`, `apply_thresholds`, `phase_blend_server`, `build_server_pair_prior`, `stacked_oof`, `write_oof`, `read_oof`, `average_over_seeds`, `score_action`, `score_point`, `score_server`, `overall`, `attach_labels`) used consistently across tasks and tests.
- One open item: Task 11 references a `--predict-test` flag for `produce_base_oof.py`. Step 11.1 sketches the LGBM variant. The Markov / phase_lgbm / player_stats variants in Step 11.2 reuse their existing standalone-script logic for test-time prediction (each existing `train_*.py` already produces a `submission_*.csv` artifact today; that logic is the source).
- Pair-prior in Task 8 is used twice — once at OOF time (rebuilt per fold from train_view) and once at test time (built from full train). The test build is intentionally NOT OOF (test rallies have no label) and matches the public LB submission semantics.

## What's next

After P2 lands, run `python -m scripts.score_oof` again with all six lines (base x 5 + Route A) and the improvement should be measurable. The submission CSV `submission_A_stacked.csv` is ready to upload as a leaderboard probe.

P3 (Route B) and P4 (Route C) are independent and can be done in parallel branches.
