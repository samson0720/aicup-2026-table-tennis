"""Global XGBoost max_depth=8 + another_data OOF + test parquets (GPU).

Same as xgb_extra but max_depth=8 (vs 6). Tests whether the depth=8 advantage
from phase_xgb8_extra also transfers to the global (non-phase-split) XGBoost.
Writes:
  artifacts/oof/xgb8_extra_{action,point,server}.parquet  (+test)

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n aicup-tt python -m scripts.produce_xgb8_extra_oof
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
from scripts.produce_xgb_oof import _sample_weight
from scripts.train_lgbm_baseline import TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES, feature_columns

MODEL_NAME = "xgb8_extra"
N_ESTIMATORS = 600
MAX_DEPTH = 8
DEVICE = "cuda"


def fit_mc(x_tr, y_tr, x_va, n_cls, seed):
    le = LabelEncoder().fit(y_tr)
    clf = xgb.XGBClassifier(objective="multi:softprob", tree_method="hist", device=DEVICE,
                            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=0.05,
                            reg_lambda=5.0, subsample=0.9, colsample_bytree=0.9,
                            random_state=seed, verbosity=0)
    clf.fit(x_tr, le.transform(y_tr), sample_weight=_sample_weight(y_tr, n_cls, "sqrt"))
    raw = clf.predict_proba(x_va)
    out = np.zeros((raw.shape[0], n_cls), dtype=np.float64)
    for enc_i, orig_cls in enumerate(le.classes_):
        out[:, int(orig_cls)] = raw[:, enc_i]
    s = out.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return out / s


def fit_bin(x_tr, y_tr, x_va, seed):
    pos = max(int((y_tr == 1).sum()), 1); neg = max(int((y_tr == 0).sum()), 1)
    clf = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist", device=DEVICE,
                            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=0.05,
                            reg_lambda=5.0, subsample=0.9, colsample_bytree=0.9,
                            scale_pos_weight=neg / pos, random_state=seed, verbosity=0)
    clf.fit(x_tr, y_tr)
    return clf.predict_proba(x_va)[:, 1]


def main():
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    extra_pairs = load_extra_pairs(set(train["match"].astype(str).unique()))
    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty: continue
        df_train = pd.concat([df_train, extra_pairs], ignore_index=True)
        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        x_tr = df_train[feats].fillna(0.0).to_numpy()
        x_va = df_valid[feats].fillna(0.0).to_numpy()
        pa = fit_mc(x_tr, df_train["y_actionId"], x_va, len(TARGET_ACTION_CLASSES), 2026 + fold)
        pp = fit_mc(x_tr, df_train["y_pointId"], x_va, len(TARGET_POINT_CLASSES), 3026 + fold)
        ps = fit_bin(x_tr, df_train["y_serverGetPoint"], x_va, 4026 + fold)
        rally = df_valid["rally_uid"].to_numpy()
        sid, fid = np.full(len(rally), seed), np.full(len(rally), fold)
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
    print(f"\n[{MODEL_NAME}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df_train = build_one_sample_per_rally(train, splits)
    full_df_train = pd.concat([full_df_train, extra_pairs], ignore_index=True)
    test = build_test_dataset(pd.read_csv(dd / "test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df_train) if c in test.columns]
    x_full = full_df_train[feats].fillna(0.0).to_numpy()
    x_test = test[feats].fillna(0.0).to_numpy()
    rally_test = test["rally_uid"].to_numpy()
    pa_t = fit_mc(x_full, full_df_train["y_actionId"], x_test, len(TARGET_ACTION_CLASSES), 2026)
    pp_t = fit_mc(x_full, full_df_train["y_pointId"], x_test, len(TARGET_POINT_CLASSES), 3026)
    ps_t = fit_bin(x_full, full_df_train["y_serverGetPoint"], x_test, 4026).reshape(-1, 1)
    _write_test_parquet(MODEL_NAME, "action", rally_test, pa_t)
    _write_test_parquet(MODEL_NAME, "point", rally_test, pp_t)
    _write_test_parquet(MODEL_NAME, "server", rally_test, ps_t)
    print(f"[{MODEL_NAME}] test predictions written.", flush=True)


if __name__ == "__main__":
    main()
