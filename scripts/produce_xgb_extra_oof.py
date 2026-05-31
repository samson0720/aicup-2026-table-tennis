"""Produce xgb_extra OOF + test parquets (XGBoost GPU + another_data).

Mirrors produce_extra_lgbm_oof but uses XGBoost on GPU. Appends another_data
new-match rows to each fold's training set. Writes:
  artifacts/oof/xgb_extra_{action,point,server}.parquet
  artifacts/oof/xgb_extra_{action,point,server}_test.parquet

Usage:
  conda run -n aicup-tt python -m scripts.produce_xgb_extra_oof
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
from scripts.produce_xgb_oof import fit_mc, fit_bin
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    feature_columns,
)

MODEL_NAME = "xgb_extra"
ITERATIONS = 600
DEVICE = "cuda"


def _stack(*arrays):
    return np.concatenate(arrays)


def run() -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    official_matches = set(train["match"].astype(str).unique())
    extra_pairs = load_extra_pairs(official_matches)

    print(f"{MODEL_NAME}: device={DEVICE} iterations={ITERATIONS}", flush=True)

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
        x_train = df_train[feats].fillna(0).to_numpy(np.float32)
        x_valid = df_valid[feats].fillna(0).to_numpy(np.float32)

        pa = fit_mc(x_train, df_train["y_actionId"], x_valid, 19, 9000 + fold, ITERATIONS, DEVICE)
        pp = fit_mc(x_train, df_train["y_pointId"], x_valid, 10, 9100 + fold, ITERATIONS, DEVICE)
        ps = fit_bin(x_train, df_train["y_serverGetPoint"], x_valid, 9200 + fold, ITERATIONS, DEVICE).reshape(-1, 1)

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed); fid = np.full(len(rally), fold)
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
    x_full = full_df_train[feats].fillna(0).to_numpy(np.float32)
    x_test = test[feats].fillna(0).to_numpy(np.float32)
    rally_test = test["rally_uid"].to_numpy()

    le_a = LabelEncoder().fit(full_df_train["y_actionId"])
    clf_a = xgb.XGBClassifier(objective="multi:softprob", tree_method="hist", device=DEVICE,
                               n_estimators=ITERATIONS, max_depth=6, learning_rate=0.05,
                               reg_lambda=5.0, subsample=0.9, colsample_bytree=0.9,
                               random_state=2026, verbosity=0)
    clf_a.fit(x_full, le_a.transform(full_df_train["y_actionId"]))
    raw_a = clf_a.predict_proba(x_test)
    pa_test = np.zeros((len(x_test), 19), dtype=np.float32)
    for i, c in enumerate(le_a.classes_):
        pa_test[:, int(c)] = raw_a[:, i]

    le_p = LabelEncoder().fit(full_df_train["y_pointId"])
    clf_p = xgb.XGBClassifier(objective="multi:softprob", tree_method="hist", device=DEVICE,
                               n_estimators=ITERATIONS, max_depth=6, learning_rate=0.05,
                               reg_lambda=5.0, subsample=0.9, colsample_bytree=0.9,
                               random_state=3026, verbosity=0)
    clf_p.fit(x_full, le_p.transform(full_df_train["y_pointId"]))
    raw_p = clf_p.predict_proba(x_test)
    pp_test = np.zeros((len(x_test), 10), dtype=np.float32)
    for i, c in enumerate(le_p.classes_):
        pp_test[:, int(c)] = raw_p[:, i]

    clf_s = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist", device=DEVICE,
                               n_estimators=ITERATIONS, max_depth=6, learning_rate=0.05,
                               reg_lambda=5.0, subsample=0.9, colsample_bytree=0.9,
                               random_state=4026, verbosity=0)
    clf_s.fit(x_full, full_df_train["y_serverGetPoint"].to_numpy())
    ps_test = clf_s.predict_proba(x_test)[:, 1:]

    _write_test_parquet(MODEL_NAME, "action", rally_test, pa_test)
    _write_test_parquet(MODEL_NAME, "point", rally_test, pp_test)
    _write_test_parquet(MODEL_NAME, "server", rally_test, ps_test)
    print(f"[{MODEL_NAME}] test predictions written.", flush=True)


if __name__ == "__main__":
    run()
