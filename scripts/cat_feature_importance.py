"""CatBoost feature importance + stability (mean/std across folds) for action+point.

Trains cat on seed 11 x folds 0/1/2, collects per-fold feature_importances_,
and saves a keep-list (drops features with low mean importance for BOTH targets).
Used to build a pruned feature set. GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.train_catboost_baseline import cat_feature_indices, fit_full_multiclass, prepare_x
from scripts.train_lgbm_baseline import TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES, feature_columns

KEEP_FRAC = 0.60  # keep top 60% of features (by max importance across action/point)


def run() -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    imp_a, imp_p = [], []
    feats = None
    for seed, fold, tv, _ in iter_cv_folds(train, splits):
        if seed != 11 or fold not in (0, 1, 2):
            continue
        st = splits[(splits.seed == seed) & (splits.fold != fold)]
        dtr = build_one_sample_per_rally(tv, st)
        if dtr.empty:
            continue
        feats = [c for c in feature_columns(dtr)]
        cat_idx = cat_feature_indices(feats)
        cat_cols = [feats[i] for i in cat_idx]
        x = prepare_x(dtr[feats], cat_cols)
        ma = fit_full_multiclass(x, dtr["y_actionId"], TARGET_ACTION_CLASSES, cat_idx, "sqrt", 9000 + fold, 300, task_type="GPU", devices="0")
        mp = fit_full_multiclass(x, dtr["y_pointId"], TARGET_POINT_CLASSES, cat_idx, "sqrt", 9100 + fold, 300, task_type="GPU", devices="0")
        imp_a.append(np.asarray(ma.feature_importances_, dtype=float))
        imp_p.append(np.asarray(mp.feature_importances_, dtype=float))
        print(f"fold {fold} importances collected", flush=True)

    A = np.vstack(imp_a); P = np.vstack(imp_p)
    df = pd.DataFrame({
        "feature": feats,
        "a_mean": A.mean(0), "a_std": A.std(0),
        "p_mean": P.mean(0), "p_std": P.std(0),
    })
    # normalize per target so a "useful for either" rule is fair
    df["a_norm"] = df["a_mean"] / df["a_mean"].sum()
    df["p_norm"] = df["p_mean"] / df["p_mean"].sum()
    df["keep_score"] = df[["a_norm", "p_norm"]].max(1)
    df = df.sort_values("keep_score", ascending=False).reset_index(drop=True)
    n_keep = int(np.ceil(KEEP_FRAC * len(df)))
    keep = df["feature"].head(n_keep).tolist()
    drop = df["feature"].tail(len(df) - n_keep).tolist()

    Path("artifacts/cat_keep_features.json").write_text(json.dumps(keep, indent=2))
    print(f"\ntotal feats {len(df)}, keep {len(keep)}, drop {len(drop)}")
    print("=== bottom 15 (dropped, low importance) ===")
    print(df.tail(15)[["feature", "a_mean", "p_mean"]].to_string(index=False))
    print("\n=== most UNSTABLE kept (high std/mean) ===")
    kept = df.head(n_keep).copy()
    kept["a_cv"] = kept["a_std"] / (kept["a_mean"] + 1e-9)
    print(kept.sort_values("a_cv", ascending=False).head(8)[["feature", "a_mean", "a_std"]].to_string(index=False))


if __name__ == "__main__":
    run()
