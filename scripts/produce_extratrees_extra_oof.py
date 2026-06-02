"""ExtraTreesClassifier + another_data OOF + test parquets.

Pilot seed11 x folds 0-2: action argmax F1 ~0.297 (vs lgbm31 ~0.267, ShuttleNet ~0.268).
Genuinely different algorithm (random threshold bagging vs GBDT boosting).
May provide orthogonal diversity in the 19-base ensemble.

CPU-only. Writes:
  artifacts/oof/extratrees_extra_{action,point,server}.parquet  (+test)

Usage:
  conda run -n aicup-tt python -m scripts.produce_extratrees_extra_oof
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES, feature_columns,
)

MODEL_NAME = "extratrees_extra"
N_ESTIMATORS = 300
MIN_SAMPLES_LEAF = 5


def _fit(X_tr, y_tr, X_va, seed, is_binary=False):
    clf = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS, max_features="sqrt",
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=seed, n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_va)
    if is_binary:
        return proba[:, list(clf.classes_).index(1)]
    # align to fixed class order
    n_cls = len(TARGET_ACTION_CLASSES) if not is_binary else 1
    return proba


def main():
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    extra_pairs = load_extra_pairs(set(train["match"].astype(str).unique()))
    print(f"{MODEL_NAME}: ExtraTrees n={N_ESTIMATORS} min_samples_leaf={MIN_SAMPLES_LEAF}", flush=True)

    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}

    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        df_tr = build_one_sample_per_rally(train_view, s_train)
        df_va = build_one_sample_per_rally(valid_view, s_valid)
        if df_tr.empty or df_va.empty:
            continue
        df_tr = pd.concat([df_tr, extra_pairs], ignore_index=True)
        feats = [c for c in feature_columns(df_tr) if c in df_va.columns]
        X_tr = df_tr[feats].fillna(-1).values
        X_va = df_va[feats].fillna(-1).values

        # Action
        clf_a = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
            min_samples_leaf=MIN_SAMPLES_LEAF, random_state=seed+fold*100,
            n_jobs=-1, class_weight="balanced_subsample")
        clf_a.fit(X_tr, df_tr["y_actionId"].astype(int).values)
        raw_a = clf_a.predict_proba(X_va)
        pa = np.zeros((len(X_va), len(TARGET_ACTION_CLASSES)))
        for i, c in enumerate(clf_a.classes_):
            pa[:, int(c)] = raw_a[:, i]

        # Point
        clf_p = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
            min_samples_leaf=MIN_SAMPLES_LEAF, random_state=seed+fold*100+1000,
            n_jobs=-1, class_weight="balanced_subsample")
        clf_p.fit(X_tr, df_tr["y_pointId"].astype(int).values)
        raw_p = clf_p.predict_proba(X_va)
        pp = np.zeros((len(X_va), len(TARGET_POINT_CLASSES)))
        for i, c in enumerate(clf_p.classes_):
            pp[:, int(c)] = raw_p[:, i]

        # Server
        clf_s = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
            min_samples_leaf=MIN_SAMPLES_LEAF, random_state=seed+fold*100+2000,
            n_jobs=-1, class_weight="balanced_subsample")
        clf_s.fit(X_tr, df_tr["y_serverGetPoint"].astype(int).values)
        ps = clf_s.predict_proba(X_va)[:, list(clf_s.classes_).index(1)]

        rally = df_va["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_va["target_strikeNumber"].to_numpy()
        for tgt, p in [("action", pa), ("point", pp), ("server", ps.reshape(-1, 1))]:
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"{MODEL_NAME} seed={seed} fold={fold} n_train={len(df_tr)} n_valid={len(rally)}", flush=True)

    for tgt in ("action", "point", "server"):
        r = np.concatenate(bag[tgt]["r"]); s = np.concatenate(bag[tgt]["s"])
        f = np.concatenate(bag[tgt]["f"]); c = np.concatenate(bag[tgt]["c"])
        p = np.concatenate(bag[tgt]["p"], axis=0)
        out = write_oof(MODEL_NAME, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)

    # Test predictions
    print(f"\n[{MODEL_NAME}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df = build_one_sample_per_rally(train, splits)
    full_df = pd.concat([full_df, extra_pairs], ignore_index=True).reset_index(drop=True)
    test_feat = build_test_dataset(pd.read_csv(dd / "test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df) if c in test_feat.columns]
    X_tr_full = full_df[feats].fillna(-1).values
    X_te = test_feat[feats].fillna(-1).values

    clf_a = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=99999, n_jobs=-1, class_weight="balanced_subsample")
    clf_a.fit(X_tr_full, full_df["y_actionId"].astype(int).values)
    raw_a = clf_a.predict_proba(X_te)
    pa_t = np.zeros((len(X_te), len(TARGET_ACTION_CLASSES)))
    for i, c in enumerate(clf_a.classes_): pa_t[:, int(c)] = raw_a[:, i]

    clf_p = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=99998, n_jobs=-1, class_weight="balanced_subsample")
    clf_p.fit(X_tr_full, full_df["y_pointId"].astype(int).values)
    raw_p = clf_p.predict_proba(X_te)
    pp_t = np.zeros((len(X_te), len(TARGET_POINT_CLASSES)))
    for i, c in enumerate(clf_p.classes_): pp_t[:, int(c)] = raw_p[:, i]

    clf_s = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=99997, n_jobs=-1, class_weight="balanced_subsample")
    clf_s.fit(X_tr_full, full_df["y_serverGetPoint"].astype(int).values)
    ps_t = clf_s.predict_proba(X_te)[:, list(clf_s.classes_).index(1)]

    rally_test = test_feat["rally_uid"].to_numpy()
    _write_test_parquet(MODEL_NAME, "action", rally_test, pa_t)
    _write_test_parquet(MODEL_NAME, "point", rally_test, pp_t)
    _write_test_parquet(MODEL_NAME, "server", rally_test, ps_t.reshape(-1, 1))
    print(f"[{MODEL_NAME}] done.", flush=True)


if __name__ == "__main__":
    main()
