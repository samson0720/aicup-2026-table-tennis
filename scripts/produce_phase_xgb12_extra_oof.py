"""Phase-specific XGBoost (max_depth=12) + another_data OOF + test parquets (GPU).

Continues the depth ladder: depth=8(+0.00571), depth=10(+0.00156), testing depth=12.
Writes: artifacts/oof/phase_xgb12_extra_{action,point,server}.parquet  (+test)

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt python -m scripts.produce_phase_xgb12_extra_oof
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

MODEL_NAME = "phase_xgb12_extra"
N_ESTIMATORS = 600
MAX_DEPTH = 12
DEVICE = "cuda"
MIN_PHASE_ROWS = 100

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

def _phase_predict(df_train, df_valid, feats):
    x_tr = df_train[feats].fillna(0.0).to_numpy()
    x_va = df_valid[feats].fillna(0.0).to_numpy()
    df_train = df_train.reset_index(drop=True); df_valid = df_valid.reset_index(drop=True)
    p_action = np.zeros((len(df_valid), len(TARGET_ACTION_CLASSES)), dtype=np.float64)
    p_point = np.zeros((len(df_valid), len(TARGET_POINT_CLASSES)), dtype=np.float64)
    p_server = np.zeros(len(df_valid), dtype=np.float64)
    train_phase = df_train["phase"].to_numpy(); valid_phase = df_valid["phase"].to_numpy()
    for phase in sorted(set(int(p) for p in valid_phase)):
        val_mask = valid_phase == phase
        if not val_mask.any(): continue
        trn_mask = train_phase == phase
        if int(trn_mask.sum()) < MIN_PHASE_ROWS: trn_mask = np.ones_like(train_phase, dtype=bool)
        xt, xv = x_tr[trn_mask], x_va[val_mask]
        p_action[val_mask] = fit_mc(xt, df_train.loc[trn_mask,"y_actionId"], xv, len(TARGET_ACTION_CLASSES), 2026+int(phase)*10)
        p_point[val_mask] = fit_mc(xt, df_train.loc[trn_mask,"y_pointId"], xv, len(TARGET_POINT_CLASSES), 3026+int(phase)*10)
        p_server[val_mask] = fit_bin(xt, df_train.loc[trn_mask,"y_serverGetPoint"], xv, 4026+int(phase)*10)
    return p_action, p_point, p_server

def main():
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    extra_pairs = load_extra_pairs(set(train["match"].astype(str).unique()))
    bag = {t: {"r":[],"s":[],"f":[],"c":[],"p":[]} for t in ("action","point","server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"]==seed)&(splits["fold"]!=fold)]
        s_valid = splits[(splits["seed"]==seed)&(splits["fold"]==fold)]
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty: continue
        df_train = pd.concat([df_train, extra_pairs], ignore_index=True)
        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        pa, pp, ps = _phase_predict(df_train, df_valid, feats)
        rally = df_valid["rally_uid"].to_numpy()
        sid,fid,cut = np.full(len(rally),seed), np.full(len(rally),fold), df_valid["target_strikeNumber"].to_numpy()
        for tgt,p in [("action",pa),("point",pp),("server",ps.reshape(-1,1))]:
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"{MODEL_NAME} seed={seed} fold={fold} n_train={len(df_train)} n_valid={len(rally)}", flush=True)
    for tgt in ("action","point","server"):
        r,s,f,c = np.concatenate(bag[tgt]["r"]),np.concatenate(bag[tgt]["s"]),np.concatenate(bag[tgt]["f"]),np.concatenate(bag[tgt]["c"])
        p = np.concatenate(bag[tgt]["p"],axis=0)
        out = write_oof(MODEL_NAME, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)
    print(f"\n[{MODEL_NAME}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df_train = build_one_sample_per_rally(train, splits)
    full_df_train = pd.concat([full_df_train, extra_pairs], ignore_index=True)
    full_df_train = full_df_train.reset_index(drop=True)
    test_features = build_test_dataset(pd.read_csv(dd/"test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df_train) if c in test_features.columns]
    x_tr_full = full_df_train[feats].fillna(0.0).to_numpy()
    x_test = test_features[feats].fillna(0.0).to_numpy()
    n_test = len(test_features)
    p_action = np.zeros((n_test, len(TARGET_ACTION_CLASSES)), dtype=np.float64)
    p_point = np.zeros((n_test, len(TARGET_POINT_CLASSES)), dtype=np.float64)
    p_server = np.zeros(n_test, dtype=np.float64)
    train_phase = full_df_train["phase"].to_numpy(); test_phase = test_features["phase"].to_numpy()
    for phase in sorted(set(int(p) for p in test_phase)):
        test_mask = test_phase==phase
        if not test_mask.any(): continue
        train_mask = train_phase==phase
        if int(train_mask.sum())<MIN_PHASE_ROWS: train_mask = np.ones_like(train_phase,dtype=bool)
        xt,xv = x_tr_full[train_mask], x_test[test_mask]
        ya,yp,ys = full_df_train.loc[train_mask,"y_actionId"],full_df_train.loc[train_mask,"y_pointId"],full_df_train.loc[train_mask,"y_serverGetPoint"]
        p_action[test_mask] = fit_mc(xt,ya,xv,len(TARGET_ACTION_CLASSES),8200+int(phase))
        p_point[test_mask] = fit_mc(xt,yp,xv,len(TARGET_POINT_CLASSES),8300+int(phase))
        p_server[test_mask] = fit_bin(xt,ys,xv,8400+int(phase))
        print(f"phase={phase}: n_train={int(train_mask.sum())}, n_test={int(test_mask.sum())}", flush=True)
    rally_test = test_features["rally_uid"].to_numpy()
    _write_test_parquet(MODEL_NAME,"action",rally_test,p_action)
    _write_test_parquet(MODEL_NAME,"point",rally_test,p_point)
    _write_test_parquet(MODEL_NAME,"server",rally_test,p_server.reshape(-1,1))
    print(f"[{MODEL_NAME}] test predictions written.", flush=True)

if __name__ == "__main__":
    main()
