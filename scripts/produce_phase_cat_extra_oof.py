"""Phase-specific CatBoost + another_data augmentation OOF + test parquets (GPU).

Combines phase-splitting (train_phase_lgbm) with CatBoost GPU training and
another_data augmentation. Writes:
  artifacts/oof/phase_cat_extra_{action,point,server}.parquet
  artifacts/oof/phase_cat_extra_{action,point,server}_test.parquet

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n aicup-tt python -m scripts.produce_phase_cat_extra_oof
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
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    feature_columns,
)
from scripts.train_catboost_baseline import (
    cat_feature_indices,
    fit_binary,
    fit_full_binary,
    fit_full_multiclass,
    fit_multiclass,
    prepare_x,
    align_multiclass,
)

MODEL_NAME = "phase_cat_extra"
ITERATIONS = 400
DEPTH = 6
TASK_TYPE = "GPU"
DEVICES = "0"
MIN_PHASE_ROWS = 100


def _phase_predict_cat(df_train: pd.DataFrame, df_valid: pd.DataFrame,
                       feats: list[str], cat_idx: list[int], cat_cols: list[str]) -> tuple:
    x_train = prepare_x(df_train[feats].reset_index(drop=True), cat_cols)
    x_valid = prepare_x(df_valid[feats].reset_index(drop=True), cat_cols)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    p_action = np.zeros((len(df_valid), len(TARGET_ACTION_CLASSES)), dtype=np.float64)
    p_point = np.zeros((len(df_valid), len(TARGET_POINT_CLASSES)), dtype=np.float64)
    p_server = np.zeros(len(df_valid), dtype=np.float64)

    train_phase = df_train["phase"].to_numpy()
    valid_phase = df_valid["phase"].to_numpy()
    for phase in sorted(set(int(p) for p in valid_phase)):
        val_mask = valid_phase == phase
        if not val_mask.any():
            continue
        trn_mask = train_phase == phase
        if int(trn_mask.sum()) < MIN_PHASE_ROWS:
            trn_mask = np.ones_like(train_phase, dtype=bool)

        xt = x_train.iloc[list(np.where(trn_mask)[0])]
        xv = x_valid.iloc[list(np.where(val_mask)[0])]
        ya_t = df_train.loc[trn_mask, "y_actionId"]
        yp_t = df_train.loc[trn_mask, "y_pointId"]
        ys_t = df_train.loc[trn_mask, "y_serverGetPoint"]

        p_action[val_mask] = fit_multiclass(
            xt, ya_t, xv, TARGET_ACTION_CLASSES, cat_idx, "sqrt",
            9000 + int(phase) * 10, ITERATIONS, DEPTH, TASK_TYPE, DEVICES,
        )
        p_point[val_mask] = fit_multiclass(
            xt, yp_t, xv, TARGET_POINT_CLASSES, cat_idx, "sqrt",
            9100 + int(phase) * 10, ITERATIONS, DEPTH, TASK_TYPE, DEVICES,
        )
        p_server[val_mask] = fit_binary(
            xt, ys_t, xv, cat_idx,
            9200 + int(phase) * 10, ITERATIONS, DEPTH, TASK_TYPE, DEVICES,
        )
    return p_action, p_point, p_server


def main() -> None:
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

        pa, pp, ps = _phase_predict_cat(df_train, df_valid, feats, cat_idx, cat_cols)

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()
        for tgt, p in [("action", pa), ("point", pp), ("server", ps.reshape(-1, 1))]:
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"{MODEL_NAME} seed={seed} fold={fold} n_train={len(df_train)} n_valid={len(rally)}", flush=True)

    for tgt in ("action", "point", "server"):
        r = np.concatenate(bag[tgt]["r"]); s = np.concatenate(bag[tgt]["s"])
        f = np.concatenate(bag[tgt]["f"]); c = np.concatenate(bag[tgt]["c"])
        p = np.concatenate(bag[tgt]["p"], axis=0)
        out = write_oof(MODEL_NAME, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)

    # Test predictions (full train + extra, phase-specific)
    print(f"\n[{MODEL_NAME}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df_train = build_one_sample_per_rally(train, splits)
    full_df_train = pd.concat([full_df_train, extra_pairs], ignore_index=True)
    full_df_train = full_df_train.reset_index(drop=True)

    test_features = build_test_dataset(pd.read_csv(dd / "test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df_train) if c in test_features.columns]
    cat_idx = cat_feature_indices(feats)
    cat_cols = [feats[i] for i in cat_idx]
    x_train_full = prepare_x(full_df_train[feats].reset_index(drop=True), cat_cols)
    x_test = prepare_x(test_features[feats].reset_index(drop=True), cat_cols)

    n_test = len(test_features)
    p_action = np.zeros((n_test, len(TARGET_ACTION_CLASSES)), dtype=np.float64)
    p_point = np.zeros((n_test, len(TARGET_POINT_CLASSES)), dtype=np.float64)
    p_server = np.zeros(n_test, dtype=np.float64)

    train_phase = full_df_train["phase"].to_numpy()
    test_phase = test_features["phase"].to_numpy()
    for phase in sorted(set(int(p) for p in test_phase)):
        test_mask = test_phase == phase
        if not test_mask.any():
            continue
        train_mask = train_phase == phase
        if int(train_mask.sum()) < MIN_PHASE_ROWS:
            train_mask = np.ones_like(train_phase, dtype=bool)

        xt = x_train_full.iloc[list(np.where(train_mask)[0])]
        ya = full_df_train.loc[train_mask, "y_actionId"]
        yp = full_df_train.loc[train_mask, "y_pointId"]
        ys = full_df_train.loc[train_mask, "y_serverGetPoint"]
        xv = x_test.iloc[list(np.where(test_mask)[0])]

        am = fit_full_multiclass(xt, ya, TARGET_ACTION_CLASSES, cat_idx, "sqrt",
                                 8200 + int(phase), ITERATIONS, DEPTH, TASK_TYPE, DEVICES)
        pm = fit_full_multiclass(xt, yp, TARGET_POINT_CLASSES, cat_idx, "sqrt",
                                 8300 + int(phase), ITERATIONS, DEPTH, TASK_TYPE, DEVICES)
        sm = fit_full_binary(xt, ys, cat_idx, 8400 + int(phase), ITERATIONS, DEPTH, TASK_TYPE, DEVICES)

        p_action[test_mask] = am.predict_proba(xv)
        p_point[test_mask] = pm.predict_proba(xv)
        p_server[test_mask] = sm.predict_proba(xv)[:, 1]
        print(f"phase={phase}: n_train={int(train_mask.sum())}, n_test={int(test_mask.sum())}", flush=True)

    rally_test = test_features["rally_uid"].to_numpy()
    _write_test_parquet(MODEL_NAME, "action", rally_test, p_action)
    _write_test_parquet(MODEL_NAME, "point", rally_test, p_point)
    _write_test_parquet(MODEL_NAME, "server", rally_test, p_server.reshape(-1, 1))
    print(f"[{MODEL_NAME}] test predictions written.", flush=True)


if __name__ == "__main__":
    main()
