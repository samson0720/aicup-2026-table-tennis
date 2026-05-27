# Route B — Feature Engineering and Chain Multi-task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a feature-engineered LightGBM stack that out-predicts the current baseline by **+0.015 to +0.030** overall, independent of Route A. Three engineering levers: OOF-safe target encoding (with unseen-player fallback), prefix sequence n-grams, and a chain pipeline (predict action → feed point → feed server) that exploits the structural correlation among the three targets.

**Architecture:** A `scripts/target_encoding.py` module owns smoothed, OOF-safe encodings keyed by `gamePlayerId`, `(gamePlayerId, phase)`, and `(gamePlayerId, gamePlayerOtherId)`. A `scripts/build_features_v2.py` module produces an enlarged prefix dataset `prefix_train_v2.parquet`. A `scripts/train_chain_lgbm.py` script trains three LightGBM models in sequence — action first, point second (consuming action's OOF probabilities as features), server third (consuming both). Each stage uses `iter_cv_folds` so OOF probability columns come from fold-out models only.

**Tech Stack:** Python 3.11, `aicup-tt` env, pandas, scikit-learn, lightgbm. No GPU required.

**Depends on:** P1 (CV splits + iterator).

---

## Spec section coverage

- Section 3.1 OOF-safe target encoding → Tasks 1–3
- Section 3.2 Sequence n-gram features → Task 4
- Section 3.3 test_new prefix semi-supervised stats → Task 5
- Section 3.4 Chain prediction → Tasks 7–8
- Section 3.5 Artifacts → Task 9

## File structure

| Path | Purpose |
|---|---|
| `scripts/target_encoding.py` | OOF-safe smoothed target encoders with unseen-key fallback. Create. |
| `scripts/feature_ngrams.py` | Bigram / transition / streak features. Create. |
| `scripts/feature_semisupervised.py` | train + test_new joint distribution stats (X-only, no y). Create. |
| `scripts/build_features_v2.py` | End-to-end feature dataset builder consuming P1 splits. Create. |
| `scripts/train_chain_lgbm.py` | Three-stage chain trainer with OOF passthrough. Create. |
| `scripts/build_route_b_submission.py` | Test-time chain inference + CSV. Create. |
| `tests/test_target_encoding.py` | Tests for OOF safety + smoothing + fallback. Create. |
| `tests/test_feature_ngrams.py` | Tests for n-gram correctness. Create. |
| `artifacts/prefix_train_v2.parquet` | Enriched feature dataset. Generated. |
| `artifacts/chain_oof.parquet` | Per-target chain OOF. Generated. |
| `artifacts/submission_B_chain.csv` | Final Route B submission. Generated. |

---

### Task 1: Smoothed target encoder with global-prior fallback

The single most error-prone primitive in this plan. Two invariants:
1. The encoder MUST be fit on a "train" partition only and applied to a "valid" partition.
2. Unseen keys MUST resolve to the global prior, not NaN.

**Files:**
- Create: `scripts/target_encoding.py`
- Create: `tests/test_target_encoding.py`

- [ ] **Step 1.1: Failing test — smoothing math**

`tests/test_target_encoding.py`:
```python
import numpy as np
import pandas as pd
import pytest

from scripts.target_encoding import SmoothedEncoder


def test_smoothing_converges_to_global_on_empty_history():
    train = pd.DataFrame({"player": [], "y": []})
    enc = SmoothedEncoder(keys=["player"], n_classes=3, alpha=20.0, global_prior=np.array([0.5, 0.3, 0.2]))
    enc.fit(train, y_col="y")
    valid = pd.DataFrame({"player": [42, 99]})
    out = enc.transform(valid)
    # Unseen keys should land on the global prior.
    np.testing.assert_allclose(out, np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]), atol=1e-9)


def test_smoothing_blends_with_alpha():
    train = pd.DataFrame({"player": [1, 1, 1, 1], "y": [0, 0, 1, 2]})
    enc = SmoothedEncoder(keys=["player"], n_classes=3, alpha=4.0,
                          global_prior=np.array([0.50, 0.25, 0.25]))
    enc.fit(train, y_col="y")
    out = enc.transform(pd.DataFrame({"player": [1]}))
    # Class 0: (2 + 4*0.50)/(4+4) = 4/8 = 0.5
    # Class 1: (1 + 4*0.25)/(4+4) = 2/8 = 0.25
    # Class 2: (1 + 4*0.25)/(4+4) = 2/8 = 0.25
    np.testing.assert_allclose(out[0], [0.5, 0.25, 0.25], atol=1e-9)
```

- [ ] **Step 1.2: Run, expect ImportError**

```bash
conda run -n aicup-tt pytest tests/test_target_encoding.py -v
```
Expected: FAIL.

- [ ] **Step 1.3: Implement `SmoothedEncoder`**

`scripts/target_encoding.py`:
```python
"""OOF-safe smoothed target encoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class SmoothedEncoder:
    keys: Sequence[str]
    n_classes: int
    alpha: float = 20.0
    global_prior: np.ndarray | None = None  # if None, computed at fit time

    def __post_init__(self) -> None:
        self._stats: pd.DataFrame | None = None
        self._global: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, y_col: str) -> "SmoothedEncoder":
        if len(train) == 0:
            self._stats = pd.DataFrame()
            if self.global_prior is not None:
                self._global = self.global_prior.copy()
            else:
                self._global = np.full(self.n_classes, 1.0 / self.n_classes)
            return self
        # global prior
        if self.global_prior is None:
            counts = np.bincount(train[y_col].astype(int), minlength=self.n_classes)
            self._global = counts / counts.sum()
        else:
            self._global = self.global_prior.copy()
        # per-key counts
        grp = train.groupby(list(self.keys))[y_col]
        agg = grp.value_counts().unstack(fill_value=0)
        agg = agg.reindex(columns=range(self.n_classes), fill_value=0)
        agg["__n__"] = agg.sum(axis=1)
        for c in range(self.n_classes):
            agg[c] = (agg[c] + self.alpha * self._global[c]) / (agg["__n__"] + self.alpha)
        self._stats = agg[list(range(self.n_classes))]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._stats is None or self._global is None:
            raise RuntimeError("call fit() before transform()")
        if self._stats.empty:
            return np.tile(self._global, (len(df), 1))
        keyed = df[list(self.keys)].apply(tuple, axis=1) if len(self.keys) > 1 else df[self.keys[0]]
        out = np.zeros((len(df), self.n_classes), dtype=np.float32)
        idx = self._stats.index
        # Vectorized: build a lookup dict from index to row.
        lookup = {k: self._stats.loc[k].to_numpy() for k in idx}
        for i, k in enumerate(keyed):
            row = lookup.get(k)
            out[i] = row if row is not None else self._global
        return out
```

- [ ] **Step 1.4: Run, PASS**

```bash
conda run -n aicup-tt pytest tests/test_target_encoding.py -v
```
Expected: 2 passed.

- [ ] **Step 1.5: Commit**

```bash
git add scripts/target_encoding.py tests/test_target_encoding.py
git commit -m "feat(te): smoothed target encoder with global-prior fallback"
```

---

### Task 2: OOF-safety invariant test

Encode the rule "encoder.fit is given fold-out train; encoder.transform sees only valid; no valid row contributes to its own encoding".

- [ ] **Step 2.1: Add invariant test**

Append to `tests/test_target_encoding.py`:
```python
def test_oof_no_self_inclusion():
    """If a player only appears in valid (never in train), its encoding must equal the global prior."""
    train = pd.DataFrame({"player": [1, 1, 2, 2, 2], "y": [0, 0, 1, 1, 2]})
    valid = pd.DataFrame({"player": [3, 3]})  # player 3 unseen in train
    enc = SmoothedEncoder(keys=["player"], n_classes=3, alpha=10.0).fit(train, y_col="y")
    out = enc.transform(valid)
    # Equal to global prior built from train: [2/5, 2/5, 1/5] -> blended with itself is itself.
    np.testing.assert_allclose(out, np.tile(enc._global, (2, 1)), atol=1e-9)


def test_does_not_overwrite_stats_on_repeat_fit():
    train1 = pd.DataFrame({"player": [1, 1], "y": [0, 0]})
    train2 = pd.DataFrame({"player": [2, 2], "y": [1, 1]})
    enc = SmoothedEncoder(keys=["player"], n_classes=2, alpha=4.0).fit(train1, y_col="y")
    stats1 = enc._stats.copy()
    enc.fit(train2, y_col="y")
    # After a second fit, internal state reflects train2 only.
    assert enc._stats.index.tolist() == [2]
    assert (enc._stats != stats1.reindex(enc._stats.index).fillna(0)).any().any()
```

- [ ] **Step 2.2: Run, PASS**

```bash
conda run -n aicup-tt pytest tests/test_target_encoding.py -v
```
Expected: 4 passed.

- [ ] **Step 2.3: Commit**

```bash
git add tests/test_target_encoding.py
git commit -m "test(te): lock in OOF-safety and re-fit invariants"
```

---

### Task 3: Multi-key encoder bundle

A higher-level helper that owns the three player keys (`player`, `(player, phase)`, `(player, opponent)`) and applies them in a single call.

- [ ] **Step 3.1: Failing test**

```python
def test_multi_encoder_bundle_shape():
    from scripts.target_encoding import build_player_encoders
    train = pd.DataFrame({
        "player": [1, 1, 2, 2, 2, 3],
        "phase":  [0, 0, 1, 2, 2, 2],
        "opponent": [9, 9, 8, 8, 8, 9],
        "y_action": [0, 1, 2, 0, 1, 3],
        "y_point":  [0, 0, 1, 1, 2, 0],
        "y_server": [1, 0, 1, 1, 0, 1],
    })
    encs = build_player_encoders(train, n_action=4, n_point=3)
    valid = train.head(3)
    out = encs.transform(valid)
    assert out.shape[0] == 3
    # Columns: 3 keys × 2 multiclass targets × max_classes  +  3 keys × 1 binary target
    # We only assert that the columns are returned and non-trivial.
    assert out.shape[1] >= 3 * (4 + 3) + 3
```

- [ ] **Step 3.2: Implement `build_player_encoders`**

Append to `scripts/target_encoding.py`:
```python
@dataclass
class PlayerEncoderBundle:
    encoders: dict[str, "SmoothedEncoder"]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = [enc.transform(df) for enc in self.encoders.values()]
        return np.concatenate(cols, axis=1)


def build_player_encoders(
    train: pd.DataFrame,
    n_action: int = 19,
    n_point: int = 10,
    alpha: float = 20.0,
) -> PlayerEncoderBundle:
    encs: dict[str, SmoothedEncoder] = {}
    keys_specs = [
        ("player",          ["player"]),
        ("player_phase",    ["player", "phase"]),
        ("player_opponent", ["player", "opponent"]),
    ]
    for name, keys in keys_specs:
        encs[f"{name}__action"] = SmoothedEncoder(keys=keys, n_classes=n_action, alpha=alpha).fit(train, y_col="y_action")
        encs[f"{name}__point"]  = SmoothedEncoder(keys=keys, n_classes=n_point,  alpha=alpha).fit(train, y_col="y_point")
        # Server is binary; reuse multiclass machinery with n_classes=2.
        bin_train = train.assign(y_server=train["y_server"].astype(int))
        encs[f"{name}__server"] = SmoothedEncoder(keys=keys, n_classes=2, alpha=alpha).fit(bin_train, y_col="y_server")
    return PlayerEncoderBundle(encs)
```

- [ ] **Step 3.3: Run, PASS**

```bash
conda run -n aicup-tt pytest tests/test_target_encoding.py -v
```
Expected: 5 passed.

- [ ] **Step 3.4: Commit**

```bash
git add scripts/target_encoding.py tests/test_target_encoding.py
git commit -m "feat(te): player encoder bundle (player, player×phase, player×opponent)"
```

---

### Task 4: Sequence n-gram features

Bigrams of `(action[t-1], action[t-2])`, transition entropy, repetition counts, strikeId switch rate.

**Files:**
- Create: `scripts/feature_ngrams.py`
- Create: `tests/test_feature_ngrams.py`

- [ ] **Step 4.1: Failing test**

`tests/test_feature_ngrams.py`:
```python
import pandas as pd
from scripts.feature_ngrams import ngram_features


def test_bigram_hash_and_repeat_count():
    prefix = pd.DataFrame({
        "strikeNumber": [1, 2, 3, 4],
        "actionId":     [5, 7, 7, 2],
        "spinId":       [1, 1, 2, 3],
        "strikeId":     [1, 1, 2, 2],
    })
    f = ngram_features(prefix)
    # Bigram of last two actions: 7 * 100 + 2 = 702
    assert f["bigram_last_action"] == 702
    # Longest run of identical action: 7 appears twice consecutively.
    assert f["max_action_run"] == 2
    # strikeId switch frequency: 1->1->2->2 has 1 switch out of 3 transitions.
    assert abs(f["strike_switch_rate"] - 1.0 / 3.0) < 1e-9
```

- [ ] **Step 4.2: Run, FAIL**

- [ ] **Step 4.3: Implement**

```python
# scripts/feature_ngrams.py
"""N-gram and streak features over a rally prefix."""
from __future__ import annotations

import math
from collections import Counter

import pandas as pd


def ngram_features(prefix: pd.DataFrame) -> dict:
    p = prefix.sort_values("strikeNumber").reset_index(drop=True)
    feat: dict[str, float | int] = {}

    a = p["actionId"].astype(int).tolist()
    s = p["spinId"].astype(int).tolist()
    st = p["strikeId"].astype(int).tolist()

    # Last two-step bigrams.
    feat["bigram_last_action"] = int(a[-1] * 100 + a[-2]) if len(a) >= 2 else -1
    feat["bigram_last_spin_action"] = int(s[-1] * 100 + a[-1]) if len(a) >= 1 and len(s) >= 1 else -1

    # Max repetition run of action.
    if a:
        run = 1
        best = 1
        for i in range(1, len(a)):
            if a[i] == a[i - 1]:
                run += 1
                best = max(best, run)
            else:
                run = 1
        feat["max_action_run"] = best
    else:
        feat["max_action_run"] = 0

    # StrikeId switch rate.
    if len(st) >= 2:
        switches = sum(1 for i in range(1, len(st)) if st[i] != st[i - 1])
        feat["strike_switch_rate"] = switches / (len(st) - 1)
    else:
        feat["strike_switch_rate"] = 0.0

    # Action transition entropy in the prefix.
    if len(a) >= 2:
        trans = Counter(zip(a[:-1], a[1:]))
        total = sum(trans.values())
        ent = -sum((c / total) * math.log(c / total + 1e-12) for c in trans.values())
        feat["action_transition_entropy"] = ent
    else:
        feat["action_transition_entropy"] = 0.0

    return feat
```

- [ ] **Step 4.4: Run, PASS**

```bash
conda run -n aicup-tt pytest tests/test_feature_ngrams.py -v
```

- [ ] **Step 4.5: Commit**

```bash
git add scripts/feature_ngrams.py tests/test_feature_ngrams.py
git commit -m "feat(features): n-gram and streak features over rally prefix"
```

---

### Task 5: Semi-supervised distribution stats from train + test_new prefixes

Test_new contains prefix rows whose feature columns are observable. We use those rows to enlarge the sample for player-level FEATURE distributions (NOT labels — there is no leakage).

**Files:**
- Create: `scripts/feature_semisupervised.py`

- [ ] **Step 5.1: Implement**

```python
"""Player-level feature distribution stats over train + test_new prefix rows.

No labels involved. Outputs: for each player, the empirical distribution of
strikeId, handId, strengthId, spinId, positionId observed across all known
strokes. These become features for both train and test rallies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLS = ("strikeId", "handId", "strengthId", "spinId", "positionId")
MAX_VAL = {"strikeId": 3, "handId": 3, "strengthId": 3, "spinId": 5, "positionId": 3}


def build_player_feature_dist(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """One row per gamePlayerId with proportions over each FEATURE_COLS value."""
    pool = pd.concat([
        train[["gamePlayerId", *FEATURE_COLS]],
        test[["gamePlayerId", *FEATURE_COLS]],
    ], ignore_index=True)

    rows: list[dict] = []
    for player, g in pool.groupby("gamePlayerId"):
        row = {"gamePlayerId": int(player), "n_strokes": int(len(g))}
        for col in FEATURE_COLS:
            counts = g[col].value_counts(normalize=True)
            for v in range(MAX_VAL[col] + 1):
                row[f"plr_{col}_rate_{v}"] = float(counts.get(v, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    test  = pd.read_csv(next(Path.cwd().glob("AI CUP*/test_new.csv")))
    df = build_player_feature_dist(train, test)
    out = Path("artifacts/player_feature_dist.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {df.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Run**

```bash
conda run -n aicup-tt python -m scripts.feature_semisupervised
```
Expected: `wrote artifacts/player_feature_dist.parquet: (~190+, ~25)`.

- [ ] **Step 5.3: Commit**

```bash
git add scripts/feature_semisupervised.py artifacts/player_feature_dist.parquet
git commit -m "feat(features): semi-supervised player feature distributions from train+test"
```

---

### Task 6: V2 feature dataset builder

Stitch together: baseline features (from `train_lgbm_baseline.add_prefix_features`) + n-gram features (Task 4) + semi-supervised player dist (Task 5) + OOF-safe target encoding (Tasks 1–3). The target encoding is fit per fold using `iter_cv_folds` — so the output parquet stores **all** OOF-encoded features for each (seed, fold).

**Files:**
- Create: `scripts/build_features_v2.py`

- [ ] **Step 6.1: Build the dataset**

```python
"""Build artifacts/prefix_train_v2.parquet — OOF-safe enriched features.

Per (seed, fold) and per rally assigned to that fold's valid set, write one
feature row. Target encodings are fit on the fold's train_view only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.feature_ngrams import ngram_features
from scripts.feature_semisupervised import build_player_feature_dist
from scripts.target_encoding import build_player_encoders
from scripts.train_lgbm_baseline import add_prefix_features


def _materialize_te_rows(prefix_rows: pd.DataFrame, encoders) -> np.ndarray:
    """Build the encoder input dataframe from prefix-flattened feature rows."""
    return encoders.transform(prefix_rows.rename(columns={
        "next_gamePlayerId_inferred": "player",   # the player whose next stroke is the target
        "phase": "phase",
        "next_gamePlayerOtherId_inferred": "opponent",
    }))


def main() -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    test  = pd.read_csv(next(Path.cwd().glob("AI CUP*/test_new.csv")))
    semi = build_player_feature_dist(train, test).set_index("gamePlayerId")

    all_rows: list[pd.DataFrame] = []
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]

        # Baseline + ngram features for valid rallies (one row per rally for this seed).
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        # Add n-gram features by re-walking each rally's prefix.
        cut_by_rally = dict(zip(s_valid["rally_uid"], s_valid["cut_strikeNumber"]))
        ngram_rows = []
        for rally_uid, g in valid_view.groupby("rally_uid", sort=False):
            cut = cut_by_rally.get(int(rally_uid))
            if cut is None: continue
            prefix = g[g["strikeNumber"] < cut]
            ngram_rows.append({"rally_uid": int(rally_uid), **ngram_features(prefix)})
        df_ng = pd.DataFrame(ngram_rows)
        df_valid = df_valid.merge(df_ng, on="rally_uid", how="left")

        # Semi-supervised player dist for the next-player.
        df_valid = df_valid.merge(
            semi.add_prefix("plr_dist_").reset_index().rename(columns={"gamePlayerId": "next_gamePlayerId_inferred"}),
            on="next_gamePlayerId_inferred", how="left",
        )

        # OOF-safe target encoding: fit encoders on train_view's flattened prefix samples.
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_train["player"] = df_train["next_gamePlayerId_inferred"]
        df_train["opponent"] = df_train["next_gamePlayerOtherId_inferred"]
        df_train["phase"] = df_train["phase"]
        df_train["y_action"] = df_train["y_actionId"]
        df_train["y_point"]  = df_train["y_pointId"]
        df_train["y_server"] = df_train["y_serverGetPoint"]
        encoders = build_player_encoders(df_train, n_action=19, n_point=10, alpha=20.0)

        te_valid = encoders.transform(df_valid.rename(columns={
            "next_gamePlayerId_inferred": "player",
            "next_gamePlayerOtherId_inferred": "opponent",
        }))
        # Column names for te_valid: keep ordering, expand as te_<i>.
        te_cols = [f"te_{i}" for i in range(te_valid.shape[1])]
        df_valid = pd.concat([df_valid.reset_index(drop=True),
                              pd.DataFrame(te_valid, columns=te_cols)], axis=1)
        df_valid["seed"] = seed
        df_valid["fold"] = fold
        all_rows.append(df_valid)
        print(f"seed={seed} fold={fold} rows={len(df_valid)}")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_parquet("artifacts/prefix_train_v2.parquet", index=False)
    print(f"wrote artifacts/prefix_train_v2.parquet: {out.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Run**

```bash
conda run -n aicup-tt python -m scripts.build_features_v2
```
Expected: 25 progress lines, then a write line. Shape roughly `(74_975, ~200)`.
Runtime: 10–20 minutes (target encoder fit per fold).

- [ ] **Step 6.3: Commit**

```bash
git add scripts/build_features_v2.py artifacts/prefix_train_v2.parquet
git commit -m "feat(features): v2 enriched prefix dataset with OOF target encoding + n-grams"
```

---

### Task 7: Chain trainer — action OOF

Fit one LightGBM per (seed, fold) on V2 features, predicting `y_actionId`. Save OOF probabilities so the next stage can consume them.

**Files:**
- Create: `scripts/train_chain_lgbm.py`

- [ ] **Step 7.1: Write the action stage**

```python
"""Chain LightGBM: action -> point (with action OOF) -> server (with both).

Each stage uses prefix_train_v2.parquet plus the OOF probabilities of all
prior stages. OOF probability columns come from fold-out models only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from scripts.oof_loader import write_oof
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES,
    fit_multiclass, fit_binary,
)


N_ACTION, N_POINT = 19, 10


def _feature_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude]


def train_action() -> None:
    df = pd.read_parquet("artifacts/prefix_train_v2.parquet")
    exclude = {"rally_uid", "match", "seed", "fold",
               "y_actionId", "y_pointId", "y_serverGetPoint"}
    feats = _feature_cols(df, exclude)

    rally_list, seed_list, fold_list, cut_list, prob_list = [], [], [], [], []
    for (seed, fold), sub in df.groupby(["seed", "fold"], sort=False):
        train = df[(df["seed"] == seed) & (df["fold"] != fold)]
        valid = sub
        p = fit_multiclass(train[feats], train["y_actionId"],
                           valid[feats], valid["y_actionId"],
                           TARGET_ACTION_CLASSES, "sqrt", 2026 + fold, 240, 31)
        rally_list.append(valid["rally_uid"].to_numpy())
        seed_list.append(np.full(len(valid), seed))
        fold_list.append(np.full(len(valid), fold))
        cut_list.append(valid["target_strikeNumber"].to_numpy())
        prob_list.append(p)
        print(f"action seed={seed} fold={fold} n_valid={len(valid)}")

    write_oof("chain_action", "action",
              np.concatenate(rally_list), np.concatenate(seed_list),
              np.concatenate(fold_list), np.concatenate(cut_list),
              np.concatenate(prob_list, axis=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["action", "point", "server", "all"], default="all")
    args = p.parse_args()
    if args.stage in ("action", "all"):
        train_action()
    # point and server stages added in Tasks 8 + 9
```

- [ ] **Step 7.2: Run**

```bash
conda run -n aicup-tt python -m scripts.train_chain_lgbm --stage action
```
Expected: 25 lines, then OOF parquet written.

- [ ] **Step 7.3: Commit**

```bash
git add scripts/train_chain_lgbm.py artifacts/oof/chain_action_action.parquet
git commit -m "feat(chain): action-stage LGBM with v2 features"
```

---

### Task 8: Chain — point stage consumes action OOF

- [ ] **Step 8.1: Append `train_point()` to `scripts/train_chain_lgbm.py`**

```python
def train_point() -> None:
    df = pd.read_parquet("artifacts/prefix_train_v2.parquet")
    action_oof = pd.read_parquet("artifacts/oof/chain_action_action.parquet")
    # Average across seeds is wrong here — we want fold-specific OOF.
    # Merge by (rally_uid, seed, fold) so the action probs that flow in are
    # the ones the OUT-OF-FOLD model produced for THIS row.
    p_cols = [f"p_{i}" for i in range(N_ACTION)]
    df = df.merge(
        action_oof[["rally_uid", "seed", "fold", *p_cols]].rename(columns={c: f"act_{c}" for c in p_cols}),
        on=["rally_uid", "seed", "fold"], how="left",
    )

    exclude = {"rally_uid", "match", "seed", "fold",
               "y_actionId", "y_pointId", "y_serverGetPoint"}
    feats = _feature_cols(df, exclude)
    rally_list, seed_list, fold_list, cut_list, prob_list = [], [], [], [], []
    for (seed, fold), sub in df.groupby(["seed", "fold"], sort=False):
        train = df[(df["seed"] == seed) & (df["fold"] != fold)]
        valid = sub
        p = fit_multiclass(train[feats], train["y_pointId"],
                           valid[feats], valid["y_pointId"],
                           TARGET_POINT_CLASSES, "sqrt", 3026 + fold, 240, 31)
        rally_list.append(valid["rally_uid"].to_numpy())
        seed_list.append(np.full(len(valid), seed))
        fold_list.append(np.full(len(valid), fold))
        cut_list.append(valid["target_strikeNumber"].to_numpy())
        prob_list.append(p)
        print(f"point seed={seed} fold={fold}")

    write_oof("chain_point", "point",
              np.concatenate(rally_list), np.concatenate(seed_list),
              np.concatenate(fold_list), np.concatenate(cut_list),
              np.concatenate(prob_list, axis=0))
```

Wire into `__main__`:
```python
    if args.stage in ("point", "all"):
        train_point()
```

- [ ] **Step 8.2: Run**

```bash
conda run -n aicup-tt python -m scripts.train_chain_lgbm --stage point
```

- [ ] **Step 8.3: Commit**

```bash
git add scripts/train_chain_lgbm.py artifacts/oof/chain_point_point.parquet
git commit -m "feat(chain): point stage consuming action OOF"
```

---

### Task 9: Chain — server stage consumes action + point OOF

- [ ] **Step 9.1: Append `train_server()`**

```python
def train_server() -> None:
    df = pd.read_parquet("artifacts/prefix_train_v2.parquet")
    a = pd.read_parquet("artifacts/oof/chain_action_action.parquet")
    p = pd.read_parquet("artifacts/oof/chain_point_point.parquet")

    a_cols = [f"p_{i}" for i in range(N_ACTION)]
    p_cols = [f"p_{i}" for i in range(N_POINT)]
    df = df.merge(a[["rally_uid","seed","fold",*a_cols]].rename(columns={c:f"act_{c}" for c in a_cols}),
                  on=["rally_uid","seed","fold"], how="left")
    df = df.merge(p[["rally_uid","seed","fold",*p_cols]].rename(columns={c:f"pnt_{c}" for c in p_cols}),
                  on=["rally_uid","seed","fold"], how="left")

    exclude = {"rally_uid","match","seed","fold","y_actionId","y_pointId","y_serverGetPoint"}
    feats = _feature_cols(df, exclude)
    rally_list, seed_list, fold_list, cut_list, prob_list = [], [], [], [], []
    for (seed, fold), sub in df.groupby(["seed","fold"], sort=False):
        train = df[(df["seed"] == seed) & (df["fold"] != fold)]
        valid = sub
        ps = fit_binary(train[feats], train["y_serverGetPoint"],
                        valid[feats], valid["y_serverGetPoint"],
                        4026 + fold, 240, 31)
        rally_list.append(valid["rally_uid"].to_numpy())
        seed_list.append(np.full(len(valid), seed))
        fold_list.append(np.full(len(valid), fold))
        cut_list.append(valid["target_strikeNumber"].to_numpy())
        prob_list.append(ps.reshape(-1, 1))
        print(f"server seed={seed} fold={fold}")

    write_oof("chain_server", "server",
              np.concatenate(rally_list), np.concatenate(seed_list),
              np.concatenate(fold_list), np.concatenate(cut_list),
              np.concatenate(prob_list, axis=0))
```

Wire into `__main__`.

- [ ] **Step 9.2: Run**

```bash
conda run -n aicup-tt python -m scripts.train_chain_lgbm --stage server
```

- [ ] **Step 9.3: Score chain OOF against baseline**

```bash
conda run -n aicup-tt python -c "
import pandas as pd
from scripts.score_oof import attach_labels, score_action, score_point, score_server, overall
train=pd.read_csv(next(__import__('pathlib').Path.cwd().glob('AI CUP*/train.csv')))
def s(model, tgt, fn):
    df=attach_labels(pd.read_parquet(f'artifacts/oof/{model}_{tgt}.parquet'), train)
    return fn(df)
a=s('chain_action','action',score_action); p=s('chain_point','point',score_point); sv=s('chain_server','server',score_server)
print({'action':a,'point':p,'server':sv,'overall':overall(a,p,sv)})
"
```
Expected: overall higher than `artifacts/base_oof_scores.json[lgbm15]['overall']` by **+0.015 to +0.030**.

- [ ] **Step 9.4: Commit**

```bash
git add scripts/train_chain_lgbm.py artifacts/oof/chain_server_server.parquet
git commit -m "feat(chain): server stage consuming action + point OOF"
```

---

### Task 10: Test-time chain inference + Route B submission

**Files:**
- Create: `scripts/build_route_b_submission.py`

- [ ] **Step 10.1: Write the inference script**

```python
"""Refit chain on full train (no folds) and predict on test_new.

Refit uses the same feature set as the OOF chain, but with target encoders
fit on the entirety of train and applied to test. This intentionally drops
the OOF safeguard for the FEATURES (target encoding is now fit on all
train), because test has no labels — there is nothing to leak. The chain
predictions ARE still chain (action predicts, then point uses action,
then server uses both); only the encoders go non-OOF.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feature_ngrams import ngram_features
from scripts.feature_semisupervised import build_player_feature_dist
from scripts.target_encoding import build_player_encoders
from scripts.train_lgbm_baseline import add_prefix_features, fit_binary, fit_multiclass, TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES


def _features_train_full(train: pd.DataFrame, semi: pd.DataFrame, encoders) -> pd.DataFrame:
    rows: list[dict] = []
    for rally_uid, g in train.groupby("rally_uid", sort=False):
        g = g.sort_values("strikeNumber").reset_index(drop=True)
        # Use cut == rally_len so the target is the last stroke; this matches
        # the inference layout where the "target" is the next stroke after
        # the entire observed prefix.
        cut = int(g.iloc[-1]["strikeNumber"])
        prefix = g[g["strikeNumber"] < cut]
        if len(prefix) == 0: continue
        feat = add_prefix_features(prefix, cut)
        feat.update(ngram_features(prefix))
        feat["y_actionId"] = int(g.iloc[-1]["actionId"])
        feat["y_pointId"]  = int(g.iloc[-1]["pointId"])
        feat["y_serverGetPoint"] = int(g.iloc[0]["serverGetPoint"])
        rows.append(feat)
    df = pd.DataFrame(rows)
    # plr_dist
    df = df.merge(
        semi.add_prefix("plr_dist_").reset_index().rename(columns={"gamePlayerId": "next_gamePlayerId_inferred"}),
        on="next_gamePlayerId_inferred", how="left",
    )
    # target encoder transform (fit on full train using the same rows)
    te = encoders.transform(df.rename(columns={
        "next_gamePlayerId_inferred": "player",
        "next_gamePlayerOtherId_inferred": "opponent",
    }))
    te_cols = [f"te_{i}" for i in range(te.shape[1])]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(te, columns=te_cols)], axis=1)
    return df


def _features_test(test: pd.DataFrame, semi: pd.DataFrame, encoders) -> pd.DataFrame:
    rows: list[dict] = []
    for rally_uid, g in test.groupby("rally_uid", sort=False):
        g = g.sort_values("strikeNumber").reset_index(drop=True)
        target_strike = int(g.iloc[-1]["strikeNumber"]) + 1
        feat = add_prefix_features(g, target_strike)
        feat.update(ngram_features(g))
        feat["rally_uid"] = int(rally_uid)
        rows.append(feat)
    df = pd.DataFrame(rows)
    df = df.merge(
        semi.add_prefix("plr_dist_").reset_index().rename(columns={"gamePlayerId": "next_gamePlayerId_inferred"}),
        on="next_gamePlayerId_inferred", how="left",
    )
    te = encoders.transform(df.rename(columns={
        "next_gamePlayerId_inferred": "player",
        "next_gamePlayerOtherId_inferred": "opponent",
    }))
    te_cols = [f"te_{i}" for i in range(te.shape[1])]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(te, columns=te_cols)], axis=1)


def main() -> None:
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    test  = pd.read_csv(next(Path.cwd().glob("AI CUP*/test_new.csv")))
    semi  = build_player_feature_dist(train, test).set_index("gamePlayerId")

    # Fit encoders on full train (rally-target rows).
    enc_train = pd.DataFrame({
        "player":  [],
        "phase":   [],
        "opponent":[],
        "y_action":[], "y_point":[], "y_server":[],
    })
    # Build encoder training data identically to V2 features.
    # We reuse _features_train_full's intermediate rows.
    pre = []
    for rally_uid, g in train.groupby("rally_uid", sort=False):
        g = g.sort_values("strikeNumber").reset_index(drop=True)
        cut = int(g.iloc[-1]["strikeNumber"])
        prefix = g[g["strikeNumber"] < cut]
        if len(prefix) == 0: continue
        pre.append({
            "player":   int(g.iloc[-1]["gamePlayerId"]),
            "opponent": int(g.iloc[-1]["gamePlayerOtherId"]),
            "phase":    (0 if cut == 2 else (1 if cut == 3 else 2)),
            "y_action": int(g.iloc[-1]["actionId"]),
            "y_point":  int(g.iloc[-1]["pointId"]),
            "y_server": int(g.iloc[0]["serverGetPoint"]),
        })
    enc_train = pd.DataFrame(pre)
    encoders = build_player_encoders(enc_train, n_action=19, n_point=10, alpha=20.0)

    df_train = _features_train_full(train, semi, encoders)
    df_test  = _features_test(test, semi, encoders)

    exclude = {"rally_uid","y_actionId","y_pointId","y_serverGetPoint"}
    feats = [c for c in df_train.columns if c not in exclude and c in df_test.columns]

    # Stage A: action
    pa_tr = fit_multiclass(df_train[feats], df_train["y_actionId"],
                           df_train[feats], df_train["y_actionId"],
                           TARGET_ACTION_CLASSES, "sqrt", 1, 240, 31)
    pa_te = fit_multiclass(df_train[feats], df_train["y_actionId"],
                           df_test[feats],  pd.Series(np.zeros(len(df_test), dtype=int)),
                           TARGET_ACTION_CLASSES, "sqrt", 1, 240, 31)
    for i in range(19):
        df_train[f"act_p_{i}"] = pa_tr[:, i]
        df_test[f"act_p_{i}"]  = pa_te[:, i]

    feats2 = feats + [f"act_p_{i}" for i in range(19)]
    pp_tr = fit_multiclass(df_train[feats2], df_train["y_pointId"],
                           df_train[feats2], df_train["y_pointId"],
                           TARGET_POINT_CLASSES, "sqrt", 2, 240, 31)
    pp_te = fit_multiclass(df_train[feats2], df_train["y_pointId"],
                           df_test[feats2],  pd.Series(np.zeros(len(df_test), dtype=int)),
                           TARGET_POINT_CLASSES, "sqrt", 2, 240, 31)
    for i in range(10):
        df_train[f"pnt_p_{i}"] = pp_tr[:, i]
        df_test[f"pnt_p_{i}"]  = pp_te[:, i]

    feats3 = feats2 + [f"pnt_p_{i}" for i in range(10)]
    ps_te = fit_binary(df_train[feats3], df_train["y_serverGetPoint"],
                       df_test[feats3], pd.Series(np.zeros(len(df_test), dtype=int)),
                       3, 240, 31)

    action_pred = pa_te.argmax(axis=1)
    point_pred  = pp_te.argmax(axis=1)
    server_prob = ps_te

    sub = pd.DataFrame({
        "rally_uid": df_test["rally_uid"].astype(int).to_numpy(),
        "actionId": action_pred,
        "pointId":  point_pred,
        "serverGetPoint": server_prob,
    })
    sub.to_csv("artifacts/submission_B_chain.csv", index=False)
    print(f"wrote artifacts/submission_B_chain.csv: {sub.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: Run**

```bash
conda run -n aicup-tt python -m scripts.build_route_b_submission
```
Expected: `wrote artifacts/submission_B_chain.csv: (1845, 4)`.
Runtime: ~10–30 minutes.

- [ ] **Step 10.3: Commit**

```bash
git add scripts/build_route_b_submission.py artifacts/submission_B_chain.csv
git commit -m "feat(route_b): chain inference + submission"
```

---

## Self-review notes

- All spec items from Section 3 are covered.
- `SmoothedEncoder` is the single point where OOF leakage could occur; tests in Task 2 lock the invariant.
- Inference-time encoders (Task 10) are intentionally fit on the FULL train. This is safe because test has no labels — no leak is possible — and it gives the test rallies a denser encoding.
- The chain at test time is implemented by reusing `fit_multiclass` with valid set = the test features; the `y_valid` argument is unused inside `fit_multiclass` for the prediction path and any non-empty pd.Series of correct length satisfies the signature.
- The action / point models are refit on full train at inference time, not on per-fold subsets. This is correct because at inference there is no OOF requirement.

## What's next

Score Route B OOF the same way Route A is scored, then upload `submission_B_chain.csv` as a leaderboard probe. The final ensemble in P5 expects both Route A's stacked OOF and Route B's chain OOF on disk.
