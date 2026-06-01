"""Markov (BackoffClassifier) + another_data augmentation OOF + test parquets.

Appends another_data training pairs to each fold's count table, giving better
probability estimates for rare action/point transitions. CPU-only (pure counting).
Writes: artifacts/oof/markov_extra_{action,point,server}.parquet  (+test)

Usage:
  conda run -n aicup-tt python -m scripts.produce_markov_extra_oof
"""
from __future__ import annotations
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
    TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES, feature_columns,
)
from scripts.train_markov_ensemble import (
    ACTION_CONTEXTS, POINT_CONTEXTS, SERVER_CONTEXTS,
    BackoffClassifier, BackoffBinary,
)

MODEL_NAME = "markov_extra"


def _build_markov(df_train, df_valid, feats):
    x_train, x_valid = df_train[feats], df_valid[feats]
    phase_alpha_action = {0: 55.0, 1: 35.0, 2: 18.0}
    phase_alpha_point  = {0: 60.0, 1: 40.0, 2: 20.0}
    phase_alpha_server = {0: 80.0, 1: 55.0, 2: 35.0}
    action_model = BackoffClassifier(TARGET_ACTION_CLASSES, ACTION_CONTEXTS, phase_alpha_action, 25.0)
    point_model  = BackoffClassifier(TARGET_POINT_CLASSES,  POINT_CONTEXTS,  phase_alpha_point,  25.0)
    server_model = BackoffBinary(SERVER_CONTEXTS, phase_alpha_server, 40.0)
    p_action = action_model.fit(x_train, df_train["y_actionId"]).predict_proba(x_valid)
    p_point  = point_model.fit(x_train, df_train["y_pointId"]).predict_proba(x_valid)
    p_server = server_model.fit(x_train, df_train["y_serverGetPoint"]).predict_proba(x_valid)
    return p_action, p_point, p_server.reshape(-1, 1)


def main():
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    extra_pairs = load_extra_pairs(set(train["match"].astype(str).unique()))
    print(f"[{MODEL_NAME}] extra_pairs: {len(extra_pairs)} rows", flush=True)

    bag = {t: {"r":[],"s":[],"f":[],"c":[],"p":[]} for t in ("action","point","server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"]==seed)&(splits["fold"]!=fold)]
        s_valid = splits[(splits["seed"]==seed)&(splits["fold"]==fold)]
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty: continue
        df_train = pd.concat([df_train, extra_pairs], ignore_index=True)
        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        pa, pp, ps = _build_markov(df_train, df_valid, feats)
        rally = df_valid["rally_uid"].to_numpy()
        sid,fid,cut = np.full(len(rally),seed), np.full(len(rally),fold), df_valid["target_strikeNumber"].to_numpy()
        for tgt,p in [("action",pa),("point",pp),("server",ps)]:
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"{MODEL_NAME} seed={seed} fold={fold} n_train={len(df_train)} n_valid={len(rally)}", flush=True)

    for tgt in ("action","point","server"):
        r,s,f,c = (np.concatenate(bag[tgt][k]) for k in ["r","s","f","c"])
        p = np.concatenate(bag[tgt]["p"], axis=0)
        out = write_oof(MODEL_NAME, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)

    # Test predictions (full train + extra)
    print(f"\n[{MODEL_NAME}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df_train = build_one_sample_per_rally(train, splits)
    full_df_train = pd.concat([full_df_train, extra_pairs], ignore_index=True)
    test_features = build_test_dataset(pd.read_csv(dd/"test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df_train) if c in test_features.columns]
    pa, pp, ps = _build_markov(full_df_train, test_features, feats)
    rally_test = test_features["rally_uid"].to_numpy()
    _write_test_parquet(MODEL_NAME, "action", rally_test, pa)
    _write_test_parquet(MODEL_NAME, "point", rally_test, pp)
    _write_test_parquet(MODEL_NAME, "server", rally_test, ps)
    print(f"[{MODEL_NAME}] test predictions written.", flush=True)


if __name__ == "__main__":
    main()
