# Private-Push v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run four new private-score bases in parallel (ShuttleNet-style neural, focal-loss GBDT, higher-order Markov, action→point joint), gate each against the 0.00168 noise floor, then rebuild the public leak-max submission on whatever survives.

**Architecture:** Each lever is an independent OOF producer that writes `artifacts/oof/<base>_<target>{,_test}.parquet` in the standard schema (`scripts/oof_loader.py`), mirroring the proven `produce_markovp_oof.py` pattern. Integration is a single A/B in `scripts/build_final_perrow.py` (add the base to `BASES`, rerun, compare honest per-row overall vs 0.32568). Survivors ship; sub-floor levers are reverted but their scripts/parquets are kept.

**Tech Stack:** Python, conda env `aicup-tt` (`conda run -n aicup-tt`), pandas/numpy, LightGBM (CPU custom objective), PyTorch 2.2 on the RTX 3090 (`CUDA_VISIBLE_DEVICES=0`), pytest.

---

## Ground rules (read before any task)

- **Conda only**: every command is `conda run -n aicup-tt <cmd>` (or `env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt ...` for GPU). Never touch `base`, never `pip install --user`. No new pip installs — all levers use already-installed packages.
- **Honest ruler**: never seed-average before scoring. The integration A/B uses `scripts/build_final_perrow.py` unchanged except for the `BASES` dict.
- **Gate**: ship a base only if the real `build_final_perrow` honest overall rises by **> 0.00168** vs the current production **0.32568**. Sub-floor → revert `BASES`, keep the script + parquets, record REJECT in `PROGRESS.md`.
- **Neural double-gate**: L1 runs a pilot (seed 11 × folds 0–2) first; only a pilot competitive with the GBDT bases (overall ≳ 0.30 standalone, or a positive ensemble delta on the pilot slice) proceeds to the full 25-fold OOF.
- **OOF schema** (from `scripts/oof_loader.py`): columns `rally_uid, seed, fold, cut_strikeNumber, p_0..p_{n-1}`. Use `write_oof(model, target, rally, seed, fold, cut, probs)` and `_write_test_parquet(model, target, rally, probs)` from `scripts/predict_test_base.py`.
- **Parallelism**: L2/L3/L4 are CPU and independent; L1 is GPU. They can run as separate background processes. The integration A/Bs are serial (each rewrites `artifacts/submission_FINAL_safe_perrow.csv`), so run integration one base at a time.

---

## Lever L3 — higher-order player×context Markov (do first: cheapest, highest confidence)

**Files:**
- Create: `scripts/produce_markov2_oof.py`
- Test: `tests/test_markov2.py`
- Modify (integration, later): `scripts/build_final_perrow.py:30-35` (`BASES` dict)

### Task L3.1: Backoff table builder + test

- [ ] **Step 1: Write the failing test** — `tests/test_markov2.py`

```python
import numpy as np
from scripts.produce_markov2_oof import fit_tables2, predict2

def _toy():
    import pandas as pd
    # 2 players, last1/last2 context, action target with 3 of 19 classes present
    return pd.DataFrame({
        "y_actionId": [0, 0, 1, 1, 2, 0, 1, 2],
        "last1_actionId": [5, 5, 6, 6, 5, 5, 6, 5],
        "last2_actionId": [3, 3, 4, 4, 3, 3, 4, 3],
        "next_gamePlayerId_inferred": [10, 10, 11, 11, 10, 10, 11, 10],
    })

def test_fit_predict_shape_and_simplex():
    df = _toy()
    tables = fit_tables2(df, "action")
    p = predict2(df, "action", tables)
    assert p.shape == (len(df), 19)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert (p >= 0).all()

def test_backoff_unseen_context_falls_back_to_global():
    df = _toy()
    tables = fit_tables2(df, "action")
    import pandas as pd
    unseen = pd.DataFrame({
        "y_actionId": [0], "last1_actionId": [99],
        "last2_actionId": [99], "next_gamePlayerId_inferred": [999],
    })
    p = predict2(unseen, "action", tables)
    glob = tables[0]
    assert np.allclose(p[0], glob, atol=1e-6)
```

- [ ] **Step 2: Run it, verify it fails**

Run: `conda run -n aicup-tt pytest tests/test_markov2.py -v`
Expected: FAIL (`ModuleNotFoundError`/`cannot import name fit_tables2`).

- [ ] **Step 3: Implement `fit_tables2` / `predict2`** — `scripts/produce_markov2_oof.py`

This extends `produce_markovp_oof.py` with a **2-gram** backoff level. Backoff chain:
global → last1 → last-2-gram(last1,last2) → (player,last1) → (player,last1,last2).

```python
"""Higher-order player x context Markov (v4 L3). Extends markovp with a
last-2-gram level and a (player, 2-gram) level via Dirichlet backoff. OOF-safe:
fit on each fold's train, apply to its valid; fit on full train for test."""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.train_lgbm_baseline import build_prefix_dataset

N = {"action": 19, "point": 10}
LAST1 = {"action": "last1_actionId", "point": "last1_pointId"}
LAST2 = {"action": "last2_actionId", "point": "last2_pointId"}
YCOL = {"action": "y_actionId", "point": "y_pointId"}
PLAYER = "next_gamePlayerId_inferred"
ALPHA = 8.0


def _smooth(counts, parent, alpha=ALPHA):
    return (counts + alpha * parent) / (counts.sum() + alpha)


def fit_tables2(df, target):
    n = N[target]
    y = df[YCOL[target]].to_numpy().astype(int)
    l1 = df[LAST1[target]].to_numpy()
    l2 = df[LAST2[target]].to_numpy()
    pl = df[PLAYER].to_numpy()
    glob = np.bincount(y, minlength=n).astype(float)
    glob = glob / glob.sum() if glob.sum() else np.full(n, 1.0 / n)
    # level 1: last1
    d1 = collections.defaultdict(lambda: np.zeros(n))
    for yy, a in zip(y, l1):
        d1[a][yy] += 1
    by_l1 = {a: _smooth(c, glob) for a, c in d1.items()}
    # level 2: (last1, last2)
    d2 = collections.defaultdict(lambda: np.zeros(n))
    for yy, a, b in zip(y, l1, l2):
        d2[(a, b)][yy] += 1
    by_l2 = {(a, b): _smooth(c, by_l1.get(a, glob)) for (a, b), c in d2.items()}
    # level 3: (player, last1)
    d3 = collections.defaultdict(lambda: np.zeros(n))
    for yy, p, a in zip(y, pl, l1):
        d3[(p, a)][yy] += 1
    by_pl1 = {(p, a): _smooth(c, by_l1.get(a, glob)) for (p, a), c in d3.items()}
    # level 4: (player, last1, last2)
    d4 = collections.defaultdict(lambda: np.zeros(n))
    for yy, p, a, b in zip(y, pl, l1, l2):
        d4[(p, a, b)][yy] += 1
    by_pl2 = {}
    for (p, a, b), c in d4.items():
        parent = by_pl1.get((p, a), by_l2.get((a, b), by_l1.get(a, glob)))
        by_pl2[(p, a, b)] = _smooth(c, parent)
    return glob, by_l1, by_l2, by_pl1, by_pl2


def predict2(df, target, tables):
    glob, by_l1, by_l2, by_pl1, by_pl2 = tables
    n = N[target]
    l1 = df[LAST1[target]].to_numpy()
    l2 = df[LAST2[target]].to_numpy()
    pl = df[PLAYER].to_numpy()
    out = np.zeros((len(df), n))
    for i, (a, b, p) in enumerate(zip(l1, l2, pl)):
        if (p, a, b) in by_pl2:
            out[i] = by_pl2[(p, a, b)]
        elif (p, a) in by_pl1:
            out[i] = by_pl1[(p, a)]
        elif (a, b) in by_l2:
            out[i] = by_l2[(a, b)]
        elif a in by_l1:
            out[i] = by_l1[a]
        else:
            out[i] = glob
    return out
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `conda run -n aicup-tt pytest tests/test_markov2.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/produce_markov2_oof.py tests/test_markov2.py
git commit -m "feat(markov2): higher-order player x 2-gram backoff tables + tests"
```

### Task L3.2: Verify `last2_*` features exist; add OOF/test runners

- [ ] **Step 1: Confirm the 2-gram feature columns exist**

Run: `conda run -n aicup-tt python -c "from scripts.diagnose_cv_gap import build_one_sample_per_rally; import pandas as pd; from pathlib import Path; from scripts.cv_splits import iter_cv_folds; tr=pd.read_csv(next(Path.cwd().glob('AI CUP*/train.csv'))); sp=pd.read_parquet('artifacts/cv_splits.parquet'); s,f,tv,vv=next(iter_cv_folds(tr,sp)); d=build_one_sample_per_rally(tv, sp[(sp.seed==s)&(sp.fold!=f)]); print([c for c in d.columns if 'last2' in c or 'last1' in c])"`
Expected: prints a list containing `last1_actionId`, `last1_pointId`. **If `last2_actionId`/`last2_pointId` are absent**, add them in `build_one_sample_per_rally` (in `scripts/diagnose_cv_gap.py`) right next to where `last1_*` is computed: `df["last2_actionId"] = <2nd-to-last observed actionId, fill -1>` (mirror the existing `last1_*` construction, shifting by 2). Re-run this check until both appear.

- [ ] **Step 2: Append `run_oof` / `run_test` / `main`** — mirror `produce_markovp_oof.py:68-118` exactly, but call `fit_tables2`/`predict2` and pass `model="markov2"`. The bagging loop, `write_oof`, and `_write_test_parquet` calls are identical to markovp (both targets `action`, `point`).

```python
def run_oof(args) -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point")}
    for seed, fold, tv, vv in iter_cv_folds(train, splits):
        st = splits[(splits.seed == seed) & (splits.fold != fold)]
        sv = splits[(splits.seed == seed) & (splits.fold == fold)]
        dtr = build_one_sample_per_rally(tv, st); dva = build_one_sample_per_rally(vv, sv)
        if dtr.empty or dva.empty:
            continue
        rally = dva["rally_uid"].to_numpy(); sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold); cut = dva["target_strikeNumber"].to_numpy()
        for t in ("action", "point"):
            p = predict2(dva, t, fit_tables2(dtr, t))
            bag[t]["r"].append(rally); bag[t]["s"].append(sid)
            bag[t]["f"].append(fid); bag[t]["c"].append(cut); bag[t]["p"].append(p)
        print(f"markov2 seed={seed} fold={fold} n={len(rally)}", flush=True)
    for t in ("action", "point"):
        r = np.concatenate(bag[t]["r"]); s = np.concatenate(bag[t]["s"])
        f = np.concatenate(bag[t]["f"]); c = np.concatenate(bag[t]["c"])
        p = np.concatenate(bag[t]["p"], axis=0)
        out = write_oof("markov2", t, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)


def run_test() -> None:
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    cache = Path("artifacts/prefix_train_baseline.parquet")
    df_train = pd.read_parquet(cache) if cache.exists() else build_prefix_dataset(train)
    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)
    rally = test_features["rally_uid"].to_numpy()
    for t in ("action", "point"):
        p = predict2(test_features, t, fit_tables2(df_train, t))
        _write_test_parquet("markov2", t, rally, p)
        print(f"wrote markov2_{t}_test: {p.shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-test", action="store_true")
    args = ap.parse_args()
    run_test() if args.predict_test else run_oof(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate OOF + test parquets**

Run: `conda run -n aicup-tt python -m scripts.produce_markov2_oof` then `conda run -n aicup-tt python -m scripts.produce_markov2_oof --predict-test`
Expected: prints `wrote artifacts/oof/markov2_action.parquet: rows=74975` (and point), plus `markov2_{action,point}_test`.

- [ ] **Step 4: Commit**

```bash
git add scripts/produce_markov2_oof.py scripts/diagnose_cv_gap.py
git commit -m "feat(markov2): 25-fold OOF + test inference for higher-order markov base"
```

### Task L3.3: Integration A/B gate

- [ ] **Step 1: Add `markov2` to BASES** — `scripts/build_final_perrow.py:30-35`. The `BASES` dict maps base→targets; add `"markov2": ("action", "point")` (match the existing markovp entry's format — open the file and copy its shape exactly).

- [ ] **Step 2: Run the production builder, record overall**

Run: `conda run -n aicup-tt python -m scripts.build_final_perrow`
Expected: prints the honest per-row scores + overall. **Record the overall.**

- [ ] **Step 3: Decide ship/reject**

If overall − 0.32568 > 0.00168 → **SHIP** (keep the BASES edit, keep regenerated `submission_FINAL_safe_perrow.csv`). Else → **REJECT**: `git checkout scripts/build_final_perrow.py` (revert the BASES edit) and rebuild to restore the 0.32568 submission.

- [ ] **Step 4: Record result in PROGRESS.md** — add a dated `## v4 L3 — higher-order markov` section with the v2/v3 result-table format (config | action | point | server | overall | lift) and the verdict.

- [ ] **Step 5: Commit**

```bash
git add PROGRESS.md scripts/build_final_perrow.py artifacts/oof/markov2_*.parquet
git commit -m "feat(markov2): integration A/B — <SHIP/REJECT> (lift <x> vs floor 0.00168)"
```

---

## Lever L4 — structured (action, point) joint base

**Files:**
- Create: `scripts/produce_joint_oof.py`
- Test: `tests/test_joint.py`

### Task L4.1: Joint conditional + marginalization + test

The new signal: `P(point) = Σ_a P(point | a) · P̂(a)`, where `P(point|a)` is an OOF-safe smoothed conditional from train counts and `P̂(a)` is the **ensemble's** action probability for that row (read from the action OOF stack — use `cat_action`/`markovp_action` as a cheap proxy for `P̂(a)`; the design only needs a reasonable action distribution, not the final stack).

- [ ] **Step 1: Write the failing test** — `tests/test_joint.py`

```python
import numpy as np
from scripts.produce_joint_oof import fit_point_given_action, marginalize_point

def test_marginalize_is_simplex_and_matches_hand_calc():
    # 3 action classes, 2 point classes, deterministic conditional
    # P(point|a=0)=[1,0], a=1->[0,1], a=2->[0.5,0.5]
    cond = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])  # (n_action=3, n_point=2)
    phat_a = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 2 rows
    out = marginalize_point(phat_a, cond)
    assert out.shape == (2, 2)
    assert np.allclose(out[0], [1.0, 0.0])
    assert np.allclose(out[1], [0.5, 0.5])
    assert np.allclose(out.sum(axis=1), 1.0)

def test_fit_conditional_simplex():
    import pandas as pd
    df = pd.DataFrame({"y_actionId": [0, 0, 1, 2, 2], "y_pointId": [3, 3, 4, 5, 6]})
    cond = fit_point_given_action(df, n_action=19, n_point=10, alpha=2.0)
    assert cond.shape == (19, 10)
    assert np.allclose(cond.sum(axis=1), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run it, verify it fails**

Run: `conda run -n aicup-tt pytest tests/test_joint.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement the core** — `scripts/produce_joint_oof.py` (core functions first)

```python
"""Structured (action, point) joint base (v4 L4). New signal vs the chain:
the marginalized joint P(point) = sum_a P(point|a) P_hat(a). OOF-safe: the
conditional P(point|a) is fit on each fold's train; P_hat(a) is read from an
existing action OOF base (cat). Evaluated only through the full downstream
pipeline (build_final_perrow), never argmax-only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import read_oof, write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.train_lgbm_baseline import build_prefix_dataset

N_ACTION, N_POINT = 19, 10
ALPHA = 4.0


def fit_point_given_action(df, n_action=N_ACTION, n_point=N_POINT, alpha=ALPHA):
    a = df["y_actionId"].to_numpy().astype(int)
    p = df["y_pointId"].to_numpy().astype(int)
    glob = np.bincount(p, minlength=n_point).astype(float)
    glob = glob / glob.sum() if glob.sum() else np.full(n_point, 1.0 / n_point)
    cond = np.zeros((n_action, n_point))
    for cls in range(n_action):
        c = np.bincount(p[a == cls], minlength=n_point).astype(float)
        cond[cls] = (c + alpha * glob) / (c.sum() + alpha)
    return cond


def marginalize_point(phat_action, cond):
    # phat_action: (n_rows, n_action); cond: (n_action, n_point) -> (n_rows, n_point)
    out = phat_action @ cond
    return out / out.sum(axis=1, keepdims=True)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `conda run -n aicup-tt pytest tests/test_joint.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/produce_joint_oof.py tests/test_joint.py
git commit -m "feat(joint): P(point|action) conditional + marginalization core + tests"
```

### Task L4.2: OOF/test runners reading the action OOF for P̂(a)

- [ ] **Step 1: Append `run_oof` / `run_test` / `main`** to `scripts/produce_joint_oof.py`.

For OOF: for each `(seed, fold)`, fit `cond` on the fold-train one-sample-per-rally frame; read `cat_action` OOF rows for this `(seed, fold)` as `P̂(a)` (align by `rally_uid`); marginalize; write as base `joint` target `point`.

```python
def _phat_action(model, seed, fold, rally_order):
    df = read_oof(model, "action")
    df = df[(df.seed == seed) & (df.fold == fold)].set_index("rally_uid")
    cols = [f"p_{c}" for c in range(N_ACTION)]
    return df.loc[rally_order, cols].to_numpy()


def run_oof(args) -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    bag = {"r": [], "s": [], "f": [], "c": [], "p": []}
    for seed, fold, tv, vv in iter_cv_folds(train, splits):
        st = splits[(splits.seed == seed) & (splits.fold != fold)]
        sv = splits[(splits.seed == seed) & (splits.fold == fold)]
        dtr = build_one_sample_per_rally(tv, st); dva = build_one_sample_per_rally(vv, sv)
        if dtr.empty or dva.empty:
            continue
        rally = dva["rally_uid"].to_numpy()
        cond = fit_point_given_action(dtr)
        phat = _phat_action(args.action_base, seed, fold, rally)
        p = marginalize_point(phat, cond)
        bag["r"].append(rally); bag["s"].append(np.full(len(rally), seed))
        bag["f"].append(np.full(len(rally), fold)); bag["c"].append(dva["target_strikeNumber"].to_numpy())
        bag["p"].append(p)
        print(f"joint seed={seed} fold={fold} n={len(rally)}", flush=True)
    r = np.concatenate(bag["r"]); s = np.concatenate(bag["s"]); f = np.concatenate(bag["f"])
    c = np.concatenate(bag["c"]); p = np.concatenate(bag["p"], axis=0)
    print("wrote", write_oof("joint", "point", r, s, f, c, p), "rows=", len(r), flush=True)


def run_test() -> None:
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    cache = Path("artifacts/prefix_train_baseline.parquet")
    df_train = pd.read_parquet(cache) if cache.exists() else build_prefix_dataset(train)
    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)
    rally = test_features["rally_uid"].to_numpy()
    cond = fit_point_given_action(df_train)
    phat = pd.read_parquet("artifacts/oof/cat_action_test.parquet").set_index("rally_uid")
    phat = phat.loc[rally, [f"p_{c}" for c in range(N_ACTION)]].to_numpy()
    p = marginalize_point(phat, cond)
    _write_test_parquet("joint", "point", rally, p)
    print(f"wrote joint_point_test: {p.shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-base", default="cat")
    ap.add_argument("--predict-test", action="store_true")
    args = ap.parse_args()
    run_test() if args.predict_test else run_oof(args)


if __name__ == "__main__":
    main()
```

Note: confirm the action test parquet filename — run `ls artifacts/oof/cat_action*` first. If it is `cat_action_action_test.parquet` (target-in-name), use that exact name in `run_test`.

- [ ] **Step 2: Generate OOF + test**

Run: `conda run -n aicup-tt python -m scripts.produce_joint_oof` then `--predict-test`.
Expected: `wrote artifacts/oof/joint_point.parquet rows= 74975` and `joint_point_test`.

- [ ] **Step 3: Commit**

```bash
git add scripts/produce_joint_oof.py
git commit -m "feat(joint): marginalized-joint point base OOF + test inference"
```

### Task L4.3: Integration A/B gate

- [ ] **Step 1: Add `"joint": ("point",)` to `BASES`** in `scripts/build_final_perrow.py`.
- [ ] **Step 2: Run `conda run -n aicup-tt python -m scripts.build_final_perrow`; record overall.**
- [ ] **Step 3: Ship/reject** by the > 0.00168 rule (revert BASES + rebuild if sub-floor).
- [ ] **Step 4: Record `## v4 L4 — (action,point) joint` in PROGRESS.md** (result table + verdict). Note: this base touches **point only**; confirm action/server unchanged in the printout.
- [ ] **Step 5: Commit**

```bash
git add PROGRESS.md scripts/build_final_perrow.py artifacts/oof/joint_*.parquet
git commit -m "feat(joint): integration A/B — <SHIP/REJECT> (lift <x>)"
```

---

## Lever L2 — focal-loss GBDT base (`lgbm_focal`)

**Files:**
- Create: `scripts/focal_loss.py`, `scripts/produce_lgbm_focal_oof.py`
- Test: `tests/test_focal_loss.py`

### Task L2.1: Multiclass focal objective + test

- [ ] **Step 1: Write the failing test** — `tests/test_focal_loss.py`

```python
import numpy as np
from scripts.focal_loss import softmax, multiclass_focal_objective

def test_softmax_rows_sum_to_one():
    z = np.random.randn(5, 4)
    p = softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)

def test_focal_gamma_zero_matches_ce_gradient_sign():
    # gamma=0 reduces to softmax cross-entropy: grad = p - onehot
    n, k = 6, 3
    rng = np.random.default_rng(0)
    raw = rng.normal(size=n * k)
    y = rng.integers(0, k, size=n)
    obj = multiclass_focal_objective(num_class=k, gamma=0.0)
    grad, hess = obj(raw, _FakeDataset(y))
    p = softmax(raw.reshape(n, k))
    onehot = np.eye(k)[y]
    assert np.allclose(grad.reshape(n, k), p - onehot, atol=1e-6)
    assert (hess > 0).all()

class _FakeDataset:
    def __init__(self, y): self._y = y
    def get_label(self): return self._y
```

- [ ] **Step 2: Run it, verify it fails**

Run: `conda run -n aicup-tt pytest tests/test_focal_loss.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement** — `scripts/focal_loss.py`

LightGBM multiclass custom objective passes `raw` as a flat array of length `n*num_class` (column-major per LightGBM: index = class*n + row). Reshape accordingly. Focal: `L = -(1-p_t)^γ log p_t`. We use the standard practical approximation where the per-class gradient keeps the softmax-CE structure scaled by the focal modulating factor; for γ=0 it must equal `p-onehot` exactly (asserted by the test).

```python
"""Multiclass focal loss as a LightGBM custom objective (v4 L2).
Reference: maxhalford.github.io/blog/lightgbm-focal-loss/. CPU; data is small."""
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
        # LightGBM lays raw out as (num_class, n) flattened -> reshape then transpose
        p = softmax(raw.reshape(num_class, n).T)  # (n, num_class)
        onehot = np.eye(num_class)[y]
        pt = (p * onehot).sum(axis=1, keepdims=True)  # prob of true class
        mod = (1.0 - pt) ** gamma  # focal modulating factor (n,1)
        grad = mod * (p - onehot)  # (n, num_class)
        # diagonal hessian approximation, kept strictly positive for stability
        hess = mod * p * (1.0 - p)
        hess = np.maximum(hess, 1e-6)
        return grad.T.reshape(-1), hess.T.reshape(-1)
    return _obj
```

Note: the test calls `obj(raw, ds)` with `raw` shaped so that `raw.reshape(n,k)` is row-major in the test. Make the test and impl consistent: in the test, build `raw` as `softmax-CE` expects. **Adjust the test's `raw` construction to `raw = rng.normal(size=k*n)` and compare against `softmax(raw.reshape(k,n).T)`** so the layout matches LightGBM's. Fix whichever side is wrong until `grad == p-onehot` at γ=0 passes.

- [ ] **Step 4: Run the test, verify it passes** (fix layout until green)

Run: `conda run -n aicup-tt pytest tests/test_focal_loss.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/focal_loss.py tests/test_focal_loss.py
git commit -m "feat(focal): multiclass focal LightGBM objective + tests"
```

### Task L2.2: Focal-LGBM OOF producer

- [ ] **Step 1: Inspect the existing LGBM OOF producer** — open `scripts/produce_base_oof.py` (150 lines) and identify how it builds the per-fold LGBM, calls `lgb.train`, and writes OOF. The focal producer reuses that exact feature/fold/IO machinery.

- [ ] **Step 2: Create `scripts/produce_lgbm_focal_oof.py`** that mirrors `produce_base_oof.py`'s lgbm path but passes `fobj=multiclass_focal_objective(num_class, gamma)` and `params` with `"objective": None` (custom obj) + `"num_class": n`. Predictions from a custom-objective booster are **raw scores** → apply `softmax(raw.reshape(n,k).T)` before `write_oof`. Targets: `action`, `point`. CLI: `--gamma {float}`, `--model-name lgbm_focal`, `--predict-test`. Write OOF as model `lgbm_focal` (and a `_test` variant via `_write_test_parquet`).

```python
# key delta vs produce_base_oof.py's lgbm call:
from scripts.focal_loss import multiclass_focal_objective, softmax
params = {"num_class": n, "learning_rate": 0.05, "num_leaves": 15,
          "min_data_in_leaf": 50, "verbose": -1}  # NO "objective" key
booster = lgb.train(params, dtrain, num_boost_round=300,
                    fobj=multiclass_focal_objective(n, gamma))
raw = booster.predict(X_valid)              # raw scores, shape (n_valid, k) or flat
prob = softmax(raw if raw.ndim == 2 else raw.reshape(-1, n))
```

(Match the feature columns, sample order, and `write_oof` call to `produce_base_oof.py`. If that file already factors a reusable `run_one_fold`, import and wrap it instead of copy-pasting.)

- [ ] **Step 3: Generate OOF for γ=2.0**

Run: `conda run -n aicup-tt python -m scripts.produce_lgbm_focal_oof --gamma 2.0` then `--gamma 2.0 --predict-test`.
Expected: `lgbm_focal_{action,point}{,_test}.parquet` written, 74975 OOF rows.

- [ ] **Step 4: Standalone honest score (sanity)** — score the new OOF the same way other bases are scored (use `scripts/score_oof.py`); print action/point macro-F1. Expected: in the same ballpark as `cat` (action ~0.27, point ~0.17); a wildly lower score means the layout/softmax is wrong — fix before integrating.

- [ ] **Step 5: Commit**

```bash
git add scripts/produce_lgbm_focal_oof.py artifacts/oof/lgbm_focal_*.parquet
git commit -m "feat(focal): lgbm_focal OOF + test (gamma=2.0)"
```

### Task L2.3: Integration A/B (γ=2.0, then γ=1.0)

- [ ] **Step 1: Add `"lgbm_focal": ("action", "point")` to `BASES`; run builder; record overall.**
- [ ] **Step 2: If sub-floor at γ=2.0, regenerate at γ=1.0** (`--gamma 1.0`, overwrites the parquets) and rerun the A/B. Two honest comparisons total — no wider grid.
- [ ] **Step 3: Ship/reject** by > 0.00168 (revert + rebuild if sub-floor). Redundancy note: if it lifts but is ~0.98 correlated with `cat`/`lgbm15`, treat as no real diversity (depth-8 lesson) and reject.
- [ ] **Step 4: Record `## v4 L2 — focal-loss GBDT` in PROGRESS.md** (both γ rows + verdict).
- [ ] **Step 5: Commit**

```bash
git add PROGRESS.md scripts/build_final_perrow.py
git commit -m "feat(focal): integration A/B — <SHIP/REJECT> (best gamma <g>, lift <x>)"
```

---

## Lever L1 — ShuttleNet-style neural base (GPU; pilot-gated)

**Files:**
- Create: `scripts/shuttle_model.py`, `scripts/train_shuttle.py`
- Test: `tests/test_shuttle_model.py`
- Reuse: `scripts/seq_dataset.py` (`RallyPrefixDataset`, `collate_batch`) unchanged.

### Task L1.1: Dual-encoder position-aware fusion module + test

The architectural differentiators vs the rejected `seq_model.RallyTransformer`: (1) **two** encoders — a rally-progress encoder over all token features and a player-style encoder over the player/sex embeddings only; (2) a **position-aware gated fusion** combining the two pooled contexts; (3) action + point heads only (no server head).

- [ ] **Step 1: Write the failing test** — `tests/test_shuttle_model.py`

```python
import torch
from scripts.shuttle_model import ShuttleForecaster
from scripts.seq_dataset import CATEGORICAL_COLS, SCORE_FLOAT_COLS, MAX_LEN

def test_forward_shapes():
    b = 4
    model = ShuttleForecaster(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    tokens = torch.randint(0, 7, (b, MAX_LEN, len(CATEGORICAL_COLS)))
    floats = torch.randn(b, MAX_LEN, len(SCORE_FLOAT_COLS))
    mask = torch.ones(b, MAX_LEN, dtype=torch.bool)
    out = model(tokens, floats, mask)
    assert out["action_logits"].shape == (b, 19)
    assert out["point_logits"].shape == (b, 10)
    assert "server_logit" not in out  # server intentionally excluded

def test_gate_is_bounded():
    model = ShuttleForecaster(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    b = 2
    tokens = torch.zeros(b, MAX_LEN, len(CATEGORICAL_COLS), dtype=torch.long)
    floats = torch.zeros(b, MAX_LEN, len(SCORE_FLOAT_COLS))
    mask = torch.ones(b, MAX_LEN, dtype=torch.bool)
    g = model.last_gate(tokens, floats, mask)
    assert g.min() >= 0.0 and g.max() <= 1.0
```

- [ ] **Step 2: Run it, verify it fails**

Run: `conda run -n aicup-tt pytest tests/test_shuttle_model.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement** — `scripts/shuttle_model.py`

```python
"""ShuttleNet-style dual-encoder forecaster (v4 L1). Rally-progress encoder +
player-style encoder + position-aware gated fusion. action + point heads only.
Reimplemented in pure PyTorch (no new deps), inspired by ShuttleNet (AAAI'22)."""
from __future__ import annotations

import torch
from torch import nn

from scripts.seq_dataset import CATEGORICAL_COLS, MAX_LEN, SCORE_FLOAT_COLS
from scripts.seq_model import DEFAULT_CARDINALITIES

PLAYER_COLS = ("gamePlayerId", "gamePlayerOtherId", "sex")


def _encoder(d_model, nhead, num_layers, dim_feedforward, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
        dropout=dropout, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=num_layers)


class ShuttleForecaster(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=512,
                 dropout=0.3, cardinalities=None):
        super().__init__()
        cards = cardinalities or DEFAULT_CARDINALITIES
        emb_dim = d_model // len(CATEGORICAL_COLS)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cards[c], emb_dim, padding_idx=0) for c in CATEGORICAL_COLS])
        self.cat_cols = CATEGORICAL_COLS
        self.player_idx = [CATEGORICAL_COLS.index(c) for c in PLAYER_COLS]
        cat_dim = emb_dim * len(CATEGORICAL_COLS)
        self.rally_proj = nn.Linear(cat_dim + len(SCORE_FLOAT_COLS), d_model)
        self.style_proj = nn.Linear(emb_dim * len(PLAYER_COLS), d_model)
        self.pos = nn.Embedding(MAX_LEN, d_model)
        self.rally_enc = _encoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.style_enc = _encoder(d_model, nhead, max(1, num_layers // 2), dim_feedforward, dropout)
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, 19)
        self.point_head = nn.Linear(d_model, 10)

    def _embed(self, tokens):
        return [tbl(tokens[:, :, i].clamp(0, tbl.num_embeddings - 1))
                for i, tbl in enumerate(self.embeddings)]

    def _contexts(self, tokens, floats, mask):
        emb = self._embed(tokens)
        rally_in = torch.cat([*emb, floats], dim=-1)
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        rally = self.rally_enc(self.rally_proj(rally_in) + self.pos(positions),
                               src_key_padding_mask=~mask)
        style_in = torch.cat([emb[i] for i in self.player_idx], dim=-1)
        style = self.style_enc(self.style_proj(style_in) + self.pos(positions),
                               src_key_padding_mask=~mask)
        lengths = mask.long().sum(1).clamp(min=1) - 1
        ar = torch.arange(rally.shape[0], device=rally.device)
        return rally[ar, lengths], style[ar, lengths]

    def last_gate(self, tokens, floats, mask):
        r, s = self._contexts(tokens, floats, mask)
        return self.gate(torch.cat([r, s], dim=-1))

    def forward(self, tokens, floats, mask):
        r, s = self._contexts(tokens, floats, mask)
        g = self.gate(torch.cat([r, s], dim=-1))
        fused = self.norm(g * r + (1.0 - g) * s)
        return {"action_logits": self.action_head(fused),
                "point_logits": self.point_head(fused)}
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `conda run -n aicup-tt pytest tests/test_shuttle_model.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/shuttle_model.py tests/test_shuttle_model.py
git commit -m "feat(shuttle): dual-encoder position-aware forecaster module + tests"
```

### Task L1.2: Training/OOF script adapted from train_seq_pilot.py

- [ ] **Step 1: Read `scripts/train_seq_pilot.py` (407 lines)** to reuse its dataset/loader/AMP/early-stop/OOF-writing loop. The only changes: import `ShuttleForecaster` instead of `RallyTransformer`; drop the server head/loss (loss = `CE(action) + CE(point)`, class-weighted by inverse-sqrt frequency to match macro-F1); write OOF for `action`, `point` only.

- [ ] **Step 2: Create `scripts/train_shuttle.py`** mirroring the seq pilot's CLI (`--seeds`, `--fold`, `--epochs`, `--batch-size`, `--d-model`, `--layers`, `--nhead`, `--ffn`, `--dropout`, `--lr`, `--model-name shuttle`, `--write-oof`, `--predict-test`). The training step computes:

```python
out = model(batch["tokens"].to(dev), batch["floats"].to(dev), batch["mask"].to(dev))
loss = (F.cross_entropy(out["action_logits"], batch["y_action"].to(dev), weight=w_action)
        + F.cross_entropy(out["point_logits"], batch["y_point"].to(dev), weight=w_point))
```

where `w_action`/`w_point` are `1/sqrt(class_count)` normalized. OOF/test writing reuses the seq pilot's `write_oof`/`_write_test_parquet` calls with `model="shuttle"`, targets `action`/`point`.

- [ ] **Step 3: GPU smoke test**

Run: `env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt python -m scripts.train_shuttle --seeds 11 --fold 0 --epochs 1 --batch-size 64 --d-model 64 --layers 2 --max-train 256 --max-valid 128`
Expected: prints `device=cuda`, `NVIDIA GeForce RTX 3090`, a train loss, and a valid row count. (If it falls back to CPU, the GPU env mapping is wrong — see PROGRESS "P4 Route C" for the `CUDA_VISIBLE_DEVICES` mapping.)

- [ ] **Step 4: Commit**

```bash
git add scripts/train_shuttle.py
git commit -m "feat(shuttle): training + OOF/test script (action+point, class-weighted CE)"
```

### Task L1.3: Pilot gate (seed 11 × folds 0–2)

- [ ] **Step 1: Run the pilot OOF** (mirror the seq pilot's winning config)

Run: `env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt python -m scripts.train_shuttle --seeds 11 --fold -1 --epochs 50 --batch-size 256 --d-model 256 --layers 4 --nhead 4 --ffn 768 --dropout 0.2 --lr 3e-4 --model-name shuttle_pilot --write-oof --write-partial` (restrict to folds 0–2 via the same flag the seq pilot uses; if none exists, run `--fold 0`, `--fold 1`, `--fold 2` separately).
Expected: writes `shuttle_pilot_{action,point}.parquet` for the pilot slice.

- [ ] **Step 2: Score the pilot standalone** with `scripts/score_oof.py`; compare action/point macro-F1 to the seq pilot baseline (seq best: action 0.2425, point 0.1618; lgbm15 action 0.2551, point 0.1619).

- [ ] **Step 3: Pilot decision.** If shuttle pilot is **competitive** (action ≥ ~0.245 AND not clearly worse than seq on point) → proceed to L1.4. If it is in the rejected-seq ballpark (~0.24 overall standalone with no ensemble promise) → **STOP L1**: record the pilot REJECT in PROGRESS.md, keep scripts + pilot parquets, do not run the full 25-fold. This is the seq/FT/TabPFN discipline — kill non-competitive neural bets cheaply.

- [ ] **Step 4: Commit the pilot result**

```bash
git add PROGRESS.md artifacts/oof/shuttle_pilot_*.parquet
git commit -m "feat(shuttle): pilot gate — <PROCEED/STOP> (action <a>, point <p> vs seq 0.2425/0.1618)"
```

### Task L1.4: Full 25-fold OOF + test + integration (only if pilot proceeded)

- [ ] **Step 1: Full OOF**

Run: `env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt python -u -m scripts.train_shuttle --seeds 11 22 33 44 55 --fold -1 --epochs 50 --batch-size 256 --d-model 256 --layers 4 --nhead 4 --ffn 768 --dropout 0.2 --lr 3e-4 --model-name shuttle --write-oof --write-partial`
Expected: `shuttle_{action,point}.parquet`, 74975 rows each.

- [ ] **Step 2: Test inference**

Run: `env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt python -m scripts.train_shuttle --predict-test --model-name shuttle --d-model 256 --layers 4 --nhead 4 --ffn 768 --epochs 50`
Expected: `shuttle_{action,point}_test.parquet`, 1845 rows.

- [ ] **Step 3: Integration A/B** — add `"shuttle": ("action", "point")` to `BASES`; run `build_final_perrow`; record overall; ship/reject by > 0.00168 (revert + rebuild if sub-floor).

- [ ] **Step 4: Record `## v4 L1 — ShuttleNet base` in PROGRESS.md** (standalone + ensemble result table + verdict).

- [ ] **Step 5: Commit**

```bash
git add PROGRESS.md scripts/build_final_perrow.py artifacts/oof/shuttle_*.parquet
git commit -m "feat(shuttle): full OOF + integration A/B — <SHIP/REJECT> (lift <x>)"
```

---

## Lever L5 — public leak expansion (runs LAST, after L1–L4 ship decisions)

**Files:**
- Modify/extend: `scripts/build_leakmax_submission.py`

### Task L5.1: Re-verify leak coverage

- [ ] **Step 1: Confirm overlap count**

Run: `conda run -n aicup-tt python -c "import pandas as pd; from pathlib import Path; new=pd.read_csv(next(Path.cwd().glob('AI CUP*/test_new.csv'))); old=pd.read_csv(next(Path.cwd().glob('AI CUP*/Reference_Only_Old_Test_Data/*'))); print('overlap rallies:', new['rally_uid'].isin(old['rally_uid']).sum() if 'rally_uid' in old else 'check schema')"`
Expected: prints the overlap (target ~1236/1845). Adjust the join key if the old-test schema differs (inspect its columns first).

- [ ] **Step 2: Record the verified coverage** as a one-line note in PROGRESS.md (no commit yet).

### Task L5.2: Rebuild leak-max on the locked private bases

- [ ] **Step 1: Read `scripts/build_leakmax_submission.py`** to see how it currently overrides point on the overlap rallies and applies server smoothing.

- [ ] **Step 2: Point it at the post-v4 production** — the leak-max point override should use the best post-L4 point model + the leak `serverGetPoint` feature (the `cat_sgp` path described in PROGRESS "Leak-max"). If any of L1–L4 shipped, the underlying `submission_FINAL_safe_perrow.csv` is already updated, so re-running the existing builder picks it up. Reconfirm action has **no** usable outcome-leak (the measured +0.0008 is dead — do not add an action leak path).

- [ ] **Step 3: Rebuild and guardrail-check**

Run: `conda run -n aicup-tt python -m scripts.build_leakmax_submission`
Expected: writes `artifacts/submission_FINAL_leakmax.csv` (1845 unique rallies, valid columns/ranges). The honest `submission_FINAL_safe_perrow.csv` stays leak-free.

- [ ] **Step 4: Record `## v4 L5 — public leak expansion` in PROGRESS.md** (note the local expectation; the public number requires a user upload, since uploads are daily-limited/teammate-shared).

- [ ] **Step 5: Commit**

```bash
git add PROGRESS.md scripts/build_leakmax_submission.py
git commit -m "feat(leakmax): rebuild public leak-max on post-v4 bases"
```

---

## Final wrap-up

### Task W.1: Campaign summary + full test run

- [ ] **Step 1: Run the full test suite**

Run: `conda run -n aicup-tt pytest -q`
Expected: all tests green (existing 32 + the new markov2/joint/focal/shuttle tests).

- [ ] **Step 2: Write `## Private-push v4 — FINAL SUMMARY` in PROGRESS.md** — a table of all five levers (lever | verdict | honest lift), the final production honest overall, and which submissions are current.

- [ ] **Step 3: Commit**

```bash
git add PROGRESS.md
git commit -m "docs(progress): private-push v4 final summary"
```

- [ ] **Step 4: Tell the user** the final honest overall, which levers shipped, and that public uploads (clean `submission_FINAL_safe_perrow.csv` and/or `submission_FINAL_leakmax.csv`) are theirs to make given the daily/teammate-shared upload limit.
```
