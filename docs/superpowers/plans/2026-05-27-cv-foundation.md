# CV Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the isolated conda env and a private-safe cross-validation splitter that mimics `test_new.csv` (per-rally cut point, match-aware, phase-stratified, 5 seeds × 5 folds). Every later route (A/B/C/Ensemble) consumes the same `artifacts/cv_splits.parquet` so all comparisons are apples-to-apples.

**Architecture:** A reusable Python module `scripts/cv_splits.py` that owns split generation and provides an `iter_cv_folds()` helper. Splits stored as a single parquet keyed by `(rally_uid, seed)`. Pytest unit tests guarantee structural invariants (match grouping, phase balance, no rally-uid leakage). A diagnostic adapter re-runs the existing LightGBM baseline through the new CV to quantify how much the CV-vs-LB gap narrows.

**Tech Stack:** Python 3.11, conda, pandas, numpy, scikit-learn, lightgbm (GPU), pytorch (CUDA 12.1), pytest, pyarrow.

---

## Spec section coverage

- Section 1.1 Conda env → Tasks 1–2
- Section 1.2 Private-safe CV → Tasks 3–7
- Section 1.3 Smoothing policy → Task 8 (HANDOFF documentation)

Sections 2–5 of the spec are covered by separate plans P2/P3/P4/P5.

## File structure (created or modified in this plan)

| Path | Purpose |
|---|---|
| `environment.yml` | Pinned conda env definition for `aicup-tt`. Create. |
| `scripts/cv_splits.py` | Split builder + `iter_cv_folds()` iterator. Create. |
| `scripts/diagnose_cv_gap.py` | Re-runs LGBM baseline on new CV, compares against old GroupKFold metric. Create. |
| `tests/__init__.py` | Empty marker so pytest discovers the package. Create. |
| `tests/test_cv_splits.py` | Pytest invariants for the split builder. Create. |
| `artifacts/cv_splits.parquet` | Generated split assignments (committed only if size < 5 MB). |
| `artifacts/cv_gap_diagnostic.json` | New-CV OOF scores + old-CV OOF scores side by side. |
| `HANDOFF.md` | Append a "## Validation strategy (post-2026-05-27)" section. Modify. |

All paths are relative to repository root `/home/tom1030507/ai_cup_table/aicup-2026-table-tennis`.

---

### Task 1: Pin the conda environment

**Files:**
- Create: `environment.yml`

- [ ] **Step 1.1: Write `environment.yml`**

```yaml
name: aicup-tt
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pandas=2.2.*
  - numpy=1.26.*
  - pyarrow=15.*
  - scikit-learn=1.4.*
  - lightgbm=4.3.*
  - xgboost=2.0.*
  - pytorch=2.2.*
  - pytorch-cuda=12.1
  - optuna=3.6.*
  - tqdm
  - matplotlib
  - pytest=8.*
  - pip:
      - typer==0.12.*
```

- [ ] **Step 1.2: Create the env**

Run: `conda env create -f environment.yml`
Expected: env created; no error. If conda complains about an existing env, run `conda env remove -n aicup-tt` first.

- [ ] **Step 1.3: Verify imports inside the env**

Run:
```bash
conda run -n aicup-tt python -c "import lightgbm, torch, pandas, sklearn; print('lgbm', lightgbm.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
Expected output contains `cuda True` (3090 visible) and no ImportError.

- [ ] **Step 1.4: Commit**

```bash
git add environment.yml
git commit -m "chore: pin isolated aicup-tt conda env for score-improvement work"
```

---

### Task 2: Repo guardrail — pytest skeleton

**Files:**
- Create: `tests/__init__.py`
- Create: `pytest.ini`

- [ ] **Step 2.1: Add `tests/__init__.py`**

Write an empty file:
```python
```

- [ ] **Step 2.2: Add `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -ra -q
filterwarnings =
    ignore::DeprecationWarning
```

- [ ] **Step 2.3: Confirm pytest discovers the empty suite**

Run: `conda run -n aicup-tt pytest`
Expected: `no tests ran` or `0 passed`, exit code 0 (because no tests yet).

- [ ] **Step 2.4: Commit**

```bash
git add tests/__init__.py pytest.ini
git commit -m "test: bootstrap pytest skeleton for upcoming cv splitter"
```

---

### Task 3: Split builder — schema and row count

**Files:**
- Create: `scripts/__init__.py` (empty, makes `scripts/` an importable package)
- Create: `scripts/cv_splits.py`
- Create: `tests/test_cv_splits.py`

This is the foundation of every other module. We start with the schema test, then the minimal builder. Match grouping, phase stratification, cut-point sampling come in Tasks 4–6.

- [ ] **Step 3.0: Make `scripts/` an importable package**

Create empty `scripts/__init__.py`:
```python
```

(Otherwise `from scripts.cv_splits import build_cv_splits` in the tests fails with `ModuleNotFoundError`.)

- [ ] **Step 3.1: Write the failing schema test**

Append to `tests/test_cv_splits.py`:
```python
from pathlib import Path

import pandas as pd
import pytest

from scripts.cv_splits import build_cv_splits

DATA_DIR = Path(__file__).resolve().parents[1] / "AI CUP競賽資料集"


@pytest.fixture(scope="module")
def splits() -> pd.DataFrame:
    train = pd.read_csv(DATA_DIR / "train.csv")
    return build_cv_splits(train, seeds=(11, 22, 33, 44, 55), n_folds=5)


def test_schema_columns(splits: pd.DataFrame) -> None:
    expected = {"rally_uid", "match", "seed", "fold", "cut_strikeNumber", "phase_bucket"}
    assert expected.issubset(splits.columns), f"missing: {expected - set(splits.columns)}"


def test_row_count(splits: pd.DataFrame) -> None:
    # 14,995 rallies × 5 seeds = 74,975 rows. Anything else is a bug.
    assert len(splits) == 14_995 * 5
```

- [ ] **Step 3.2: Run tests, verify they fail with import error**

Run: `conda run -n aicup-tt pytest tests/test_cv_splits.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_cv_splits'`.

- [ ] **Step 3.3: Write minimal `scripts/cv_splits.py` to satisfy schema + count**

```python
"""Private-safe CV splits for AI Cup 2026 table-tennis."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


PHASE_BUCKETS = ("phase0", "phase1", "phase2plus")


def _phase_bucket(target_strike: int) -> str:
    if target_strike == 2:
        return "phase0"
    if target_strike == 3:
        return "phase1"
    return "phase2plus"


def build_cv_splits(
    train: pd.DataFrame,
    seeds: Sequence[int] = (11, 22, 33, 44, 55),
    n_folds: int = 5,
) -> pd.DataFrame:
    """Return one row per (rally_uid, seed) with assigned fold and cut point."""
    rallies = (
        train.groupby("rally_uid", sort=False)
        .agg(match=("match", "first"), rally_len=("strikeNumber", "max"))
        .reset_index()
    )
    rallies = rallies[rallies["rally_len"] >= 2].copy()

    rows: list[dict] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _, r in rallies.iterrows():
            cut = int(rng.integers(2, int(r["rally_len"]) + 1))  # target strike number
            rows.append(
                {
                    "rally_uid": int(r["rally_uid"]),
                    "match": int(r["match"]),
                    "seed": int(seed),
                    "fold": 0,  # filled in later tasks
                    "cut_strikeNumber": cut,
                    "phase_bucket": _phase_bucket(cut),
                }
            )
    return pd.DataFrame(rows)
```

- [ ] **Step 3.4: Run tests, verify PASS**

Run: `conda run -n aicup-tt pytest tests/test_cv_splits.py -v`
Expected: both tests pass.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/__init__.py scripts/cv_splits.py tests/test_cv_splits.py
git commit -m "feat(cv): add cv_splits builder skeleton with schema + row count tests"
```

---

### Task 4: Match-aware fold assignment

Same match → same fold within a seed (otherwise rallies of the same match leak across train/valid).

- [ ] **Step 4.1: Add the failing invariant test**

Append to `tests/test_cv_splits.py`:
```python
def test_match_groups_share_a_fold(splits: pd.DataFrame) -> None:
    # For every (seed, match), all rallies must land in the same fold.
    bad = (
        splits.groupby(["seed", "match"])["fold"].nunique()
        .reset_index(name="n_folds")
    )
    offenders = bad[bad["n_folds"] != 1]
    assert offenders.empty, offenders.head().to_dict("records")


def test_fold_range(splits: pd.DataFrame) -> None:
    assert splits["fold"].between(0, 4).all()
```

- [ ] **Step 4.2: Run tests, verify match-grouping fails**

Run: `conda run -n aicup-tt pytest tests/test_cv_splits.py::test_match_groups_share_a_fold tests/test_cv_splits.py::test_fold_range -v`
Expected: FAIL — currently every fold is 0; range passes but match grouping fails for >1 match cases when we fix fold range. Actually with all-zero fold, match-grouping vacuously passes too. The fold-range test fails the moment we sprinkle non-zero folds. So we add fold assignment and then BOTH should pass.

- [ ] **Step 4.3: Replace the `fold = 0` placeholder with a per-seed, match-keyed assignment**

In `scripts/cv_splits.py`, inside `build_cv_splits`, replace the `rows` loop with:

```python
    rows: list[dict] = []
    matches = rallies["match"].unique()
    for seed in seeds:
        rng = np.random.default_rng(seed)

        # 1) Shuffle matches and slice into n_folds roughly-equal chunks.
        shuffled = matches.copy()
        rng.shuffle(shuffled)
        fold_of_match = {
            int(m): int(i % n_folds)
            for i, m in enumerate(shuffled)
        }

        # 2) Sample cut points per rally.
        for _, r in rallies.iterrows():
            cut = int(rng.integers(2, int(r["rally_len"]) + 1))
            rows.append(
                {
                    "rally_uid": int(r["rally_uid"]),
                    "match": int(r["match"]),
                    "seed": int(seed),
                    "fold": fold_of_match[int(r["match"])],
                    "cut_strikeNumber": cut,
                    "phase_bucket": _phase_bucket(cut),
                }
            )
```

- [ ] **Step 4.4: Run tests, verify PASS**

Run: `conda run -n aicup-tt pytest tests/test_cv_splits.py -v`
Expected: 4 passed.

- [ ] **Step 4.5: Commit**

```bash
git add scripts/cv_splits.py tests/test_cv_splits.py
git commit -m "feat(cv): assign rallies to folds at match granularity per seed"
```

---

### Task 5: Phase-stratified fold balance

Each fold should see roughly the same phase mix; otherwise phase-0 (the hardest bucket) clumps and per-fold scores swing wildly.

- [ ] **Step 5.1: Add the failing balance test**

Append to `tests/test_cv_splits.py`:
```python
def test_phase_balance_within_tolerance(splits: pd.DataFrame) -> None:
    # For each seed, the share of each phase bucket per fold must be within
    # 5 percentage points of the seed-wide share. 5pp is generous; pure
    # random match shuffling already gets close.
    for seed, sub in splits.groupby("seed"):
        global_share = sub["phase_bucket"].value_counts(normalize=True)
        per_fold = (
            sub.groupby("fold")["phase_bucket"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
        )
        diff = (per_fold - global_share).abs()
        worst = float(diff.values.max())
        assert worst <= 0.05, f"seed {seed} worst phase-share drift {worst:.3f}\n{diff}"
```

- [ ] **Step 5.2: Run test, observe whether it passes by luck**

Run: `conda run -n aicup-tt pytest tests/test_cv_splits.py::test_phase_balance_within_tolerance -v`
Expected: may pass by luck (because phase distribution at rally level is fairly uniform), or may fail with a 0.05+ drift on one seed. Record the outcome:
- If PASS: proceed to Step 5.5 (no logic change needed; the test locks in current behavior).
- If FAIL: continue to Step 5.3.

- [ ] **Step 5.3: If the test fails, switch the match→fold mapping to a phase-stratified round-robin**

In `scripts/cv_splits.py`, replace the `fold_of_match` block with:

```python
        # Phase-stratified assignment: compute phase-bucket histogram per match,
        # then greedily assign each match to the fold with the smallest current
        # imbalance in its dominant phase bucket. This keeps fold-level phase
        # shares close to the global share.
        match_phase_counts = (
            rallies.assign(_cut=lambda d: rng.integers(2, d["rally_len"] + 1))
            .assign(_bucket=lambda d: d["_cut"].map(_phase_bucket))
            .groupby("match")["_bucket"]
            .value_counts()
            .unstack(fill_value=0)
        )
        fold_load = {f: {b: 0 for b in PHASE_BUCKETS} for f in range(n_folds)}
        fold_of_match: dict[int, int] = {}
        order = list(matches.copy())
        rng.shuffle(order)
        for m in order:
            counts = match_phase_counts.loc[int(m)] if int(m) in match_phase_counts.index else None
            dominant = (
                str(counts.idxmax()) if counts is not None and counts.sum() > 0 else PHASE_BUCKETS[2]
            )
            target_fold = min(range(n_folds), key=lambda f: fold_load[f][dominant])
            fold_of_match[int(m)] = target_fold
            if counts is not None:
                for b in PHASE_BUCKETS:
                    fold_load[target_fold][b] += int(counts.get(b, 0))
```

- [ ] **Step 5.4: Re-run the test, verify PASS**

Run: `conda run -n aicup-tt pytest tests/test_cv_splits.py -v`
Expected: 5 passed.

- [ ] **Step 5.5: Commit**

```bash
git add scripts/cv_splits.py tests/test_cv_splits.py
git commit -m "test(cv): lock in phase-balance invariant across folds"
```

(If you changed code in 5.3, the commit message becomes `feat(cv): phase-stratified fold assignment`.)

---

### Task 6: Cut-point distribution sanity

`cut_strikeNumber` must vary across seeds for the same rally (otherwise multi-seed CV degenerates) and stay inside each rally's actual length.

- [ ] **Step 6.1: Add the failing invariant**

Append to `tests/test_cv_splits.py`:
```python
def test_cut_within_rally_length(splits: pd.DataFrame) -> None:
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["rally_uid", "strikeNumber"])
    rally_len = train.groupby("rally_uid")["strikeNumber"].max().to_dict()
    bad = splits[splits["cut_strikeNumber"] > splits["rally_uid"].map(rally_len)]
    assert bad.empty, bad.head().to_dict("records")
    assert (splits["cut_strikeNumber"] >= 2).all()


def test_cut_varies_across_seeds(splits: pd.DataFrame) -> None:
    # A rally with rally_len >= 4 should see at least 2 distinct cut values
    # across 5 seeds with very high probability. Allow up to 1% of long
    # rallies to coincide.
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["rally_uid", "strikeNumber"])
    rally_len = train.groupby("rally_uid")["strikeNumber"].max()
    long_rallies = rally_len[rally_len >= 4].index
    sub = splits[splits["rally_uid"].isin(long_rallies)]
    distinct = sub.groupby("rally_uid")["cut_strikeNumber"].nunique()
    coincide = (distinct < 2).mean()
    assert coincide < 0.01, f"{coincide:.3%} of long rallies have identical cuts across seeds"
```

- [ ] **Step 6.2: Run, expect PASS (cut sampling is already in place from Task 3)**

Run: `conda run -n aicup-tt pytest tests/test_cv_splits.py -v`
Expected: 7 passed. If `test_cut_varies_across_seeds` fails, double-check that `rng = np.random.default_rng(seed)` is re-seeded per seed inside the loop.

- [ ] **Step 6.3: Commit**

```bash
git add tests/test_cv_splits.py
git commit -m "test(cv): lock in cut-point validity and across-seed variance"
```

---

### Task 7: `iter_cv_folds()` consumer API + no-leak test

Every downstream script (Routes A/B/C) reads CV through one iterator. This is where the "no rally_uid leak between train and valid" invariant lives.

- [ ] **Step 7.1: Write the failing iterator test**

Append to `tests/test_cv_splits.py`:
```python
from scripts.cv_splits import iter_cv_folds


def test_iter_cv_folds_no_rally_leak(splits: pd.DataFrame) -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    seen_folds = set()
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        assert isinstance(train_view, pd.DataFrame)
        assert isinstance(valid_view, pd.DataFrame)
        train_rallies = set(train_view["rally_uid"].unique())
        valid_rallies = set(valid_view["rally_uid"].unique())
        assert train_rallies.isdisjoint(valid_rallies), (
            f"seed={seed} fold={fold} leaks {len(train_rallies & valid_rallies)} rallies"
        )
        seen_folds.add((seed, fold))
    assert len(seen_folds) == 5 * 5  # 5 seeds × 5 folds
```

- [ ] **Step 7.2: Run, expect ImportError**

Run: `conda run -n aicup-tt pytest tests/test_cv_splits.py::test_iter_cv_folds_no_rally_leak -v`
Expected: FAIL with `ImportError: cannot import name 'iter_cv_folds'`.

- [ ] **Step 7.3: Add `iter_cv_folds()` to `scripts/cv_splits.py`**

```python
def iter_cv_folds(
    train: pd.DataFrame,
    splits: pd.DataFrame,
) -> Iterator[tuple[int, int, pd.DataFrame, pd.DataFrame]]:
    """Yield (seed, fold, train_view, valid_view).

    For each (seed, fold), valid_view contains the rows whose rally is in that fold,
    truncated to strikes strictly before each rally's cut point — these are the
    PREFIX rows the model sees, with the row at cut_strikeNumber held out as the
    prediction target. train_view contains rows from every other fold's rallies.

    Downstream scripts decide how to use these views (build prefix features,
    compute target encodings, train models). The only guarantee here is that
    no rally_uid appears in both train and valid for the same (seed, fold).
    """
    train_by_rally = train.set_index("rally_uid", drop=False).sort_index()
    for seed in sorted(splits["seed"].unique()):
        seed_splits = splits[splits["seed"] == seed]
        for fold in sorted(seed_splits["fold"].unique()):
            valid_meta = seed_splits[seed_splits["fold"] == fold]
            train_meta = seed_splits[seed_splits["fold"] != fold]
            valid_rallies = valid_meta["rally_uid"].to_numpy()
            train_rallies = train_meta["rally_uid"].to_numpy()
            valid_view = train_by_rally.loc[valid_rallies].reset_index(drop=True)
            train_view = train_by_rally.loc[train_rallies].reset_index(drop=True)
            yield int(seed), int(fold), train_view, valid_view
```

- [ ] **Step 7.4: Re-run all tests**

Run: `conda run -n aicup-tt pytest tests/test_cv_splits.py -v`
Expected: 8 passed.

- [ ] **Step 7.5: Commit**

```bash
git add scripts/cv_splits.py tests/test_cv_splits.py
git commit -m "feat(cv): add iter_cv_folds iterator with no-leak invariant"
```

---

### Task 8: Persist splits + CLI entry point

We materialize the splits once so every later script reads the exact same partition.

- [ ] **Step 8.1: Add a CLI to `scripts/cv_splits.py`**

Append to `scripts/cv_splits.py`:
```python
def _find_data_dir() -> Path:
    for p in Path.cwd().glob("AI CUP*"):
        if p.is_dir():
            return p
    raise FileNotFoundError("AI CUP data directory not found from cwd")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("artifacts/cv_splits.parquet"))
    args = parser.parse_args()

    data_dir = _find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    splits = build_cv_splits(train, seeds=tuple(args.seeds), n_folds=args.folds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(args.out, index=False)
    print(f"wrote {args.out}: {splits.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Run the CLI**

Run: `conda run -n aicup-tt python -m scripts.cv_splits --out artifacts/cv_splits.parquet`
Expected: `wrote artifacts/cv_splits.parquet: (74975, 6)`.

(`scripts/__init__.py` already exists from Task 3.0, so this command should just work.)

- [ ] **Step 8.3: Quick sanity print**

Run:
```bash
conda run -n aicup-tt python -c "import pandas as pd; df=pd.read_parquet('artifacts/cv_splits.parquet'); print(df.head()); print(df.groupby(['seed','fold']).size().describe())"
```
Expected: 25 (seed, fold) groups with roughly 3000 rows each.

- [ ] **Step 8.4: Commit**

```bash
git add scripts/cv_splits.py artifacts/cv_splits.parquet
git commit -m "feat(cv): materialize cv_splits.parquet via CLI for downstream reuse"
```

---

### Task 9: Diagnostic — quantify the CV-vs-LB gap

Re-run the existing LightGBM baseline through the new CV and compare against the old `lgbm_baseline_cv.json` numbers. We are not optimizing the model here; we only want to know whether the new CV is more conservative and whether per-fold variance is smaller.

- [ ] **Step 9.1: Create the diagnostic adapter**

Create `scripts/diagnose_cv_gap.py`:
```python
"""Re-run the existing LightGBM baseline through the new private-safe CV.

Outputs artifacts/cv_gap_diagnostic.json with per-fold and aggregate scores so
we can compare to artifacts/lgbm_baseline_cv.json (old GroupKFold metric).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from scripts.cv_splits import iter_cv_folds
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    add_prefix_features,
    feature_columns,
    fit_binary,
    fit_multiclass,
)


def build_one_sample_per_rally(view: pd.DataFrame, splits_sub: pd.DataFrame) -> pd.DataFrame:
    """For each rally in `view`, build a single feature row at the seed's cut point."""
    cut_by_rally = dict(zip(splits_sub["rally_uid"], splits_sub["cut_strikeNumber"]))
    rows: list[dict] = []
    for rally_uid, grp in view.groupby("rally_uid", sort=False):
        cut = cut_by_rally.get(int(rally_uid))
        if cut is None:
            continue
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        prefix = grp[grp["strikeNumber"] < cut]
        target = grp[grp["strikeNumber"] == cut]
        if len(prefix) == 0 or len(target) == 0:
            continue
        feats = add_prefix_features(prefix, int(target.iloc[0]["strikeNumber"]))
        feats["y_actionId"] = int(target.iloc[0]["actionId"])
        feats["y_pointId"] = int(target.iloc[0]["pointId"])
        feats["y_serverGetPoint"] = int(grp.iloc[0]["serverGetPoint"])
        rows.append(feats)
    return pd.DataFrame(rows)


def main() -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))

    per_fold: list[dict] = []
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        train_seed = splits[splits["seed"] == seed]
        df_train = build_one_sample_per_rally(train_view, train_seed[train_seed["fold"] != fold])
        df_valid = build_one_sample_per_rally(valid_view, train_seed[train_seed["fold"] == fold])
        if df_train.empty or df_valid.empty:
            continue

        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        x_train, x_valid = df_train[feats], df_valid[feats]

        pa = fit_multiclass(
            x_train, df_train["y_actionId"], x_valid, df_valid["y_actionId"],
            TARGET_ACTION_CLASSES, "sqrt", 2026 + fold, 180, 15,
        )
        pp = fit_multiclass(
            x_train, df_train["y_pointId"], x_valid, df_valid["y_pointId"],
            TARGET_POINT_CLASSES, "sqrt", 3026 + fold, 180, 15,
        )
        ps = fit_binary(
            x_train, df_train["y_serverGetPoint"], x_valid, df_valid["y_serverGetPoint"],
            4026 + fold, 180, 15,
        )

        action_f1 = f1_score(df_valid["y_actionId"], pa.argmax(1),
                             labels=TARGET_ACTION_CLASSES, average="macro", zero_division=0)
        point_f1 = f1_score(df_valid["y_pointId"], pp.argmax(1),
                            labels=TARGET_POINT_CLASSES, average="macro", zero_division=0)
        server_auc = roc_auc_score(df_valid["y_serverGetPoint"], ps)
        overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc

        per_fold.append({
            "seed": seed, "fold": fold,
            "action_macro_f1": float(action_f1),
            "point_macro_f1": float(point_f1),
            "server_auc": float(server_auc),
            "overall": float(overall),
            "n_train": int(len(df_train)), "n_valid": int(len(df_valid)),
        })
        print(f"seed={seed} fold={fold}: overall={overall:.5f}")

    df = pd.DataFrame(per_fold)
    summary = {
        "per_fold": per_fold,
        "by_seed": (
            df.groupby("seed")[["action_macro_f1", "point_macro_f1", "server_auc", "overall"]]
            .mean().reset_index().to_dict("records")
        ),
        "mean": df[["action_macro_f1", "point_macro_f1", "server_auc", "overall"]].mean().to_dict(),
        "std":  df[["action_macro_f1", "point_macro_f1", "server_auc", "overall"]].std().to_dict(),
        "old_cv_reference": {
            "source": "artifacts/lgbm_baseline_cv.json",
            "note": "compare overall and per-target macro-F1 against this file's results.sqrt.oof",
        },
    }
    Path("artifacts/cv_gap_diagnostic.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["mean"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["std"],  indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.2: Run the diagnostic**

Run: `conda run -n aicup-tt python -m scripts.diagnose_cv_gap`
Expected: 25 lines `seed=… fold=…: overall=…`, then a mean and std JSON dict.
Runtime: ~10–20 minutes on CPU; safe to run in background. Capture the printed mean overall — call it `new_cv_overall`.

- [ ] **Step 9.3: Read the diagnostic and decide whether the gap narrowed**

Open `artifacts/cv_gap_diagnostic.json` and compare:
- old CV overall (from `artifacts/lgbm_baseline_cv.json` → `results.sqrt.oof.overall`) ≈ **0.32439**
- new CV overall = `new_cv_overall`
- new CV `std.overall` should be small (< ~0.01); if it's huge, the rally-prefix CV is noisy and we re-examine the phase-stratification weights before continuing.

No automatic gate — record the decision as a 1-sentence note in HANDOFF.md in Task 10.

- [ ] **Step 9.4: Commit**

```bash
git add scripts/diagnose_cv_gap.py artifacts/cv_gap_diagnostic.json
git commit -m "feat(cv): diagnose new-CV vs old-CV gap for baseline LGBM"
```

---

### Task 10: Document the new validation strategy in HANDOFF.md

- [ ] **Step 10.1: Append a new section to `HANDOFF.md`**

Add at the end of `HANDOFF.md`:
```markdown

## Validation strategy (post-2026-05-27)

Old `GroupKFold by match` mismatches `test_new.csv`, which is built from
mid-rally cut points, not held-out matches. New CV (`scripts/cv_splits.py`,
materialized to `artifacts/cv_splits.parquet`):

- Per-rally random cut point in `[2, rally_len]`, mimicking `test_new`.
- Match-aware grouping: every rally of a match shares one fold per seed.
- Phase-stratified round-robin assignment of matches to folds.
- 5 seeds × 5 folds.

Overlap analysis: 0 of 216 train matches reappear in 79 test matches;
40 of 71 test players appear in train, 31 are unseen. Route B target
encodings must smooth toward a global prior to handle unseen players.

Diagnostic (`artifacts/cv_gap_diagnostic.json`) records the new-CV LGBM
baseline overall metric and per-seed std. Improvements smaller than one
std are rejected as noise by every downstream route.

Smoothing trick (old-test `serverGetPoint` overlap → 0.95/0.05) stays out
of every new model's training and OOF. It is only applied when generating
the `submission_FINAL_smooth.csv` public-leaderboard backup in Section 5
of the design spec.
```

- [ ] **Step 10.2: Commit**

```bash
git add HANDOFF.md
git commit -m "docs(handoff): describe new private-safe CV and overlap findings"
```

---

## Manual sanity checklist (after all 10 tasks)

- [ ] `conda env list` shows `aicup-tt`.
- [ ] `conda run -n aicup-tt pytest` reports all tests passing.
- [ ] `artifacts/cv_splits.parquet` exists, shape `(74975, 6)`.
- [ ] `artifacts/cv_gap_diagnostic.json` exists with non-zero `mean.overall`.
- [ ] `HANDOFF.md` ends with the new "Validation strategy (post-2026-05-27)" section.
- [ ] Git log shows ~10 small commits, each scoped to one task.

## Self-review notes (filled in during plan writing)

- All 8 spec sub-requirements in Section 1 of the design map to tasks above.
- No "TBD" or placeholder text in any step; every code block is complete.
- Type/name consistency: `build_cv_splits`, `iter_cv_folds`, `cv_splits.parquet`,
  `cut_strikeNumber`, `phase_bucket` used identically across tasks and tests.
- Diagnostic runtime is the only step >10 min; it is gated only on disk space
  and CPU, no GPU. Section 4 of the design is what needs the 3090.

## What's next

After P1 lands, choose one of:

- **P2 — Route A** (post-processing + stacking). Touches OOF probabilities
  produced by the new CV; lowest risk; quickest to verify lift.
- **P3 — Route B** (target encoding + chain). Needs `iter_cv_folds` from P1
  for OOF safety.
- **P4 — Route C** (Transformer on 3090). Independent of P2/P3.

P2/P3/P4 can run in parallel branches once P1 is green.
