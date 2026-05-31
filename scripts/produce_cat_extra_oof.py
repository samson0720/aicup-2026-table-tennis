"""Produce cat_extra OOF + test parquets (CatBoost + another_data, GPU).

Mirrors produce_extra_lgbm_oof but uses CatBoost on GPU. Appends another_data
new-match rows to each fold's training set. Writes:
  artifacts/oof/cat_extra_{action,point,server}.parquet
  artifacts/oof/cat_extra_{action,point,server}_test.parquet

Usage:
  conda run -n aicup-tt python -m scripts.produce_cat_extra_oof
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
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

MODEL_NAME = "cat_extra"
ITERATIONS = 400
DEPTH = 6
TASK_TYPE = "GPU"
DEVICES = "0"


def _stack(*arrays):
    return np.concatenate(arrays)


def run() -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    official_matches = set(train["match"].astype(str).unique())
    extra_pairs = load_extra_pairs(official_matches)

    print(f"{MODEL_NAME}: task_type={TASK_TYPE} devices={DEVICES}", flush=True)

    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty:
            continue

        df_train = pd.concat([df_train, extra_pairs], ignore_index=True)

        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        cat_idx = cat_feature_indices(feats)
        cat_cols = [feats[i] for i in cat_idx]
        x_train = prepare_x(df_train[feats], cat_cols)
        x_valid = prepare_x(df_valid[feats], cat_cols)

        pa = fit_multiclass(x_train, df_train["y_actionId"], x_valid,
                            TARGET_ACTION_CLASSES, cat_idx, "sqrt", 9000 + fold, ITERATIONS,
                            depth=DEPTH, task_type=TASK_TYPE, devices=DEVICES)
        pp = fit_multiclass(x_train, df_train["y_pointId"], x_valid,
                            TARGET_POINT_CLASSES, cat_idx, "sqrt", 9100 + fold, ITERATIONS,
                            depth=DEPTH, task_type=TASK_TYPE, devices=DEVICES)
        ps = fit_binary(x_train, df_train["y_serverGetPoint"], x_valid,
                        cat_idx, 9200 + fold, ITERATIONS,
                        depth=DEPTH, task_type=TASK_TYPE, devices=DEVICES).reshape(-1, 1)

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()
        for tgt, p in (("action", pa), ("point", pp), ("server", ps)):
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"{MODEL_NAME} seed={seed} fold={fold} n_train={len(df_train)} n_valid={len(rally)}", flush=True)

    for tgt in ("action", "point", "server"):
        r = np.concatenate(bag[tgt]["r"]); s = np.concatenate(bag[tgt]["s"])
        f = np.concatenate(bag[tgt]["f"]); c = np.concatenate(bag[tgt]["c"])
        p = np.concatenate(bag[tgt]["p"], axis=0)
        out = write_oof(MODEL_NAME, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)

    # Test predictions (full train + extra)
    print(f"\n[{MODEL_NAME}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df_train = build_one_sample_per_rally(train, splits)
    full_df_train = pd.concat([full_df_train, extra_pairs], ignore_index=True)

    test = build_test_dataset(pd.read_csv(dd / "test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df_train) if c in test.columns]
    cat_idx = cat_feature_indices(feats)
    cat_cols = [feats[i] for i in cat_idx]
    x_full = prepare_x(full_df_train[feats], cat_cols)
    x_test = prepare_x(test[feats], cat_cols)
    rally_test = test["rally_uid"].to_numpy()

    from scripts.train_catboost_baseline import fit_full_multiclass, fit_full_binary
    ma = fit_full_multiclass(x_full, full_df_train["y_actionId"],
                             TARGET_ACTION_CLASSES, cat_idx, "sqrt", 2026, ITERATIONS,
                             depth=DEPTH, task_type=TASK_TYPE, devices=DEVICES)
    mp = fit_full_multiclass(x_full, full_df_train["y_pointId"],
                             TARGET_POINT_CLASSES, cat_idx, "sqrt", 3026, ITERATIONS,
                             depth=DEPTH, task_type=TASK_TYPE, devices=DEVICES)
    ms = fit_full_binary(x_full, full_df_train["y_serverGetPoint"],
                         cat_idx, 4026, ITERATIONS,
                         depth=DEPTH, task_type=TASK_TYPE, devices=DEVICES)

    from scripts.make_lgbm_submission import align_proba
    pa_test = ma.predict_proba(x_test)
    pp_test = mp.predict_proba(x_test)
    ps_test = ms.predict_proba(x_test)[:, 1:]

    _write_test_parquet(MODEL_NAME, "action", rally_test, pa_test)
    _write_test_parquet(MODEL_NAME, "point", rally_test, pp_test)
    _write_test_parquet(MODEL_NAME, "server", rally_test, ps_test)
    print(f"[{MODEL_NAME}] test predictions written.", flush=True)


if __name__ == "__main__":
    run()
