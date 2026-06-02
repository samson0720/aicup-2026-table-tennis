"""DART-boosted LGBM (31 leaves) + another_data OOF + test parquets.

DART (Dropouts meet Multiple Additive Regression Trees) provides better
regularization than standard gradient boosting by randomly dropping trees
during training. Pilot (seed11 x folds 0-2) showed +0.017 action argmax
vs standard lgbm31. Gate against 19-base ensemble to see ensemble lift.

CPU-only. Writes:
  artifacts/oof/dart_lgbm31_extra_{action,point,server}.parquet  (+test)

Usage:
  conda run -n aicup-tt python -m scripts.produce_dart_lgbm_extra_oof
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES, feature_columns,
    class_weights, categorical_feature_names,
)

MODEL_NAME = "dart_lgbm31_extra"
NUM_LEAVES = 31
N_ESTIMATORS = 180


def _fit_dart(x_tr, y_tr, x_va, classes, weight_mode, seed):
    cw = class_weights(y_tr, classes, weight_mode)
    sw = y_tr.map(cw).fillna(1.0) if cw else None
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(classes),
        n_estimators=N_ESTIMATORS, learning_rate=0.035, num_leaves=NUM_LEAVES,
        min_child_samples=30, boosting_type="dart", drop_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.05, reg_lambda=0.2,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    cat_feats = categorical_feature_names(list(x_tr.columns))
    model.fit(x_tr, y_tr, sample_weight=sw, categorical_feature=cat_feats)
    raw = model.predict_proba(x_va)
    out = np.zeros((raw.shape[0], len(classes)), dtype=np.float64)
    for i, c in enumerate(model.classes_):
        out[:, int(c)] = raw[:, i]
    return out


def _fit_dart_binary(x_tr, y_tr, x_va, seed):
    pos = max(int((y_tr == 1).sum()), 1); neg = max(int((y_tr == 0).sum()), 1)
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=N_ESTIMATORS, learning_rate=0.035,
        num_leaves=NUM_LEAVES, min_child_samples=30, boosting_type="dart", drop_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.05, reg_lambda=0.2,
        scale_pos_weight=neg / pos, random_state=seed, n_jobs=-1, verbose=-1,
    )
    cat_feats = categorical_feature_names(list(x_tr.columns))
    model.fit(x_tr, y_tr, categorical_feature=cat_feats)
    return model.predict_proba(x_va)[:, 1]


def main() -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    extra_pairs = load_extra_pairs(set(train["match"].astype(str).unique()))
    print(f"{MODEL_NAME}: DART boosting, {NUM_LEAVES} leaves, {N_ESTIMATORS} estimators", flush=True)

    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        df_tr = build_one_sample_per_rally(train_view, s_train)
        df_va = build_one_sample_per_rally(valid_view, s_valid)
        if df_tr.empty or df_va.empty: continue
        df_tr = pd.concat([df_tr, extra_pairs], ignore_index=True)
        feats = [c for c in feature_columns(df_tr) if c in df_va.columns]

        pa = _fit_dart(df_tr[feats], df_tr["y_actionId"], df_va[feats], TARGET_ACTION_CLASSES, "sqrt", seed+fold*100)
        pp = _fit_dart(df_tr[feats], df_tr["y_pointId"], df_va[feats], TARGET_POINT_CLASSES, "sqrt", seed+fold*100+1000)
        ps = _fit_dart_binary(df_tr[feats], df_tr["y_serverGetPoint"], df_va[feats], seed+fold*100+2000)

        rally = df_va["rally_uid"].to_numpy()
        sid, fid, cut = np.full(len(rally), seed), np.full(len(rally), fold), df_va["target_strikeNumber"].to_numpy()
        for tgt, p in [("action", pa), ("point", pp), ("server", ps.reshape(-1, 1))]:
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"{MODEL_NAME} seed={seed} fold={fold} n_train={len(df_tr)} n_valid={len(rally)}", flush=True)

    for tgt in ("action", "point", "server"):
        r,s,f,c,p = (np.concatenate(bag[tgt][k]) for k in "rsfcp")
        out = write_oof(MODEL_NAME, tgt, r, s, f, c, p if tgt != "server" else p.reshape(-1,1) if p.ndim == 1 else p)
        print(f"wrote {out}: rows={len(r)}", flush=True)

    # Test predictions (full train + extra_data)
    print(f"\n[{MODEL_NAME}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df = build_one_sample_per_rally(train, splits)
    full_df = pd.concat([full_df, extra_pairs], ignore_index=True).reset_index(drop=True)
    test_feat = build_test_dataset(pd.read_csv(dd / "test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df) if c in test_feat.columns]

    pa = _fit_dart(full_df[feats], full_df["y_actionId"], test_feat[feats], TARGET_ACTION_CLASSES, "sqrt", 99999)
    pp = _fit_dart(full_df[feats], full_df["y_pointId"], test_feat[feats], TARGET_POINT_CLASSES, "sqrt", 99998)
    ps = _fit_dart_binary(full_df[feats], full_df["y_serverGetPoint"], test_feat[feats], 99997)

    rally_test = test_feat["rally_uid"].to_numpy()
    _write_test_parquet(MODEL_NAME, "action", rally_test, pa)
    _write_test_parquet(MODEL_NAME, "point", rally_test, pp)
    _write_test_parquet(MODEL_NAME, "server", rally_test, ps.reshape(-1, 1))
    print(f"[{MODEL_NAME}] test predictions written.", flush=True)


if __name__ == "__main__":
    main()
