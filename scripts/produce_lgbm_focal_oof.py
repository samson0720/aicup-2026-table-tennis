"""OOF producer for LightGBM with focal-loss custom objective.

Produces artifacts/oof/lgbm_focal_{action,point}.parquet (OOF) and
artifacts/oof/lgbm_focal_{action,point}_test.parquet (test predictions).

Usage:
  python -m scripts.produce_lgbm_focal_oof               # full 25-fold OOF
  python -m scripts.produce_lgbm_focal_oof --predict-test  # + test parquets
  python -m scripts.produce_lgbm_focal_oof --gamma 2.0    # custom focal gamma

LightGBM >= 4.0: pass custom objective via params dict (not fobj= kwarg).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    build_prefix_dataset,
    feature_columns,
)
from scripts.focal_loss import multiclass_focal_objective, softmax

# Fixed class counts — never label-encode; absent classes get zero gradient
N_CLASSES = {"action": 19, "point": 10}
YCOL = {"action": "y_actionId", "point": "y_pointId"}

BASE_PARAMS = dict(
    learning_rate=0.035,
    num_leaves=15,
    min_child_samples=30,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.05,
    reg_lambda=0.2,
    verbosity=-1,
    n_jobs=-1,
)


def _train_predict(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_va: pd.DataFrame,
    target: str,
    gamma: float,
    num_boost_round: int = 180,
) -> np.ndarray:
    """Train focal-loss LightGBM and return probability matrix for x_va."""
    n = N_CLASSES[target]
    dtrain = lgb.Dataset(x_tr, label=y_tr.astype(int))
    obj = multiclass_focal_objective(n, gamma)
    # LightGBM >= 4.0: pass objective inside params dict
    params = {**BASE_PARAMS, "num_class": n, "objective": obj}
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    # With custom objective predict() returns raw scores
    raw = booster.predict(x_va)
    # LightGBM returns shape (n_samples, num_class) for multiclass custom obj
    if raw.ndim == 1:
        raw = raw.reshape(-1, n)
    probs = softmax(raw)
    return probs


def _stack(rs, ss, fs, cs, ps):
    return (
        np.concatenate(rs),
        np.concatenate(ss),
        np.concatenate(fs),
        np.concatenate(cs),
        np.concatenate(ps, axis=0),
    )


def run_oof(gamma: float = 2.0, predict_test: bool = False, model_name: str = "lgbm_focal") -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))

    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point")}

    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty:
            continue
        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        x_train, x_valid = df_train[feats], df_valid[feats]

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()

        for tgt in ("action", "point"):
            probs = _train_predict(x_train, df_train[YCOL[tgt]], x_valid, tgt, gamma)
            bag[tgt]["r"].append(rally)
            bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid)
            bag[tgt]["c"].append(cut)
            bag[tgt]["p"].append(probs)

        print(f"{model_name} seed={seed} fold={fold} valid_n={len(rally)}", flush=True)

    for tgt in ("action", "point"):
        r, s, f, c, p = _stack(
            bag[tgt]["r"], bag[tgt]["s"], bag[tgt]["f"], bag[tgt]["c"], bag[tgt]["p"]
        )
        out = write_oof(model_name, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)

    if predict_test:
        _run_predict_test(train, gamma, model_name)


def _run_predict_test(train: pd.DataFrame, gamma: float, model_name: str) -> None:
    """Fit on full prefix-train data and write test parquets."""
    data_dir = next(Path.cwd().glob("AI CUP*"))
    test = pd.read_csv(data_dir / "test_new.csv")

    cache = Path("artifacts/prefix_train_baseline.parquet")
    if cache.exists():
        df_train = pd.read_parquet(cache)
    else:
        df_train = build_prefix_dataset(train)
        df_train.to_parquet(cache, index=False)

    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(df_train) if c in test_features.columns]
    x_train = df_train[feats]
    x_test = test_features[feats]
    rally = test_features["rally_uid"].to_numpy()

    for tgt in ("action", "point"):
        probs = _train_predict(x_train, df_train[YCOL[tgt]], x_test, tgt, gamma)
        out = _write_test_parquet(model_name, tgt, rally, probs)
        print(f"wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM focal-loss OOF producer")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma (default 2.0)")
    parser.add_argument("--predict-test", action="store_true", help="Also write test parquets")
    parser.add_argument("--model-name", default="lgbm_focal", help="Base name for output files")
    args = parser.parse_args()

    run_oof(gamma=args.gamma, predict_test=args.predict_test, model_name=args.model_name)


if __name__ == "__main__":
    main()
