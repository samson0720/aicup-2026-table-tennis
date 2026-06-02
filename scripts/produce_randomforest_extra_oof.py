"""RandomForestClassifier + another_data OOF + test parquets.

RF uses optimal split thresholds (vs random in ExtraTrees) + bootstrap sampling.
Pilot: RF_300 action argmax F1=0.30233 vs ET_300 0.29766 (+0.00467).
May provide diversity that complements ExtraTrees (random thresholds) already in ensemble.

CPU-only. Writes:
  artifacts/oof/randomforest_extra_{action,point,server}.parquet (+test)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
from scripts.train_lgbm_baseline import TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES, feature_columns

MODEL_NAME = "randomforest_extra"
N_ESTIMATORS = 300
MIN_SAMPLES_LEAF = 5

def main():
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    extra_pairs = load_extra_pairs(set(train["match"].astype(str).unique()))
    print(f"{MODEL_NAME}: RF n={N_ESTIMATORS} min_sl={MIN_SAMPLES_LEAF}", flush=True)

    bag = {t: {"r":[],"s":[],"f":[],"c":[],"p":[]} for t in ("action","point","server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_tr = splits[(splits["seed"]==seed)&(splits["fold"]!=fold)]
        s_va = splits[(splits["seed"]==seed)&(splits["fold"]==fold)]
        df_tr = build_one_sample_per_rally(train_view, s_tr)
        df_va = build_one_sample_per_rally(valid_view, s_va)
        if df_tr.empty or df_va.empty: continue
        df_tr = pd.concat([df_tr, extra_pairs], ignore_index=True)
        feats = [c for c in feature_columns(df_tr) if c in df_va.columns]
        X_tr = df_tr[feats].fillna(-1).values; X_va = df_va[feats].fillna(-1).values

        for tgt, classes, y_col in [("action", TARGET_ACTION_CLASSES, "y_actionId"),
                                     ("point", TARGET_POINT_CLASSES, "y_pointId")]:
            clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
                min_samples_leaf=MIN_SAMPLES_LEAF, random_state=seed+fold*100+(0 if tgt=="action" else 1000),
                n_jobs=-1, class_weight="balanced_subsample")
            clf.fit(X_tr, df_tr[y_col].astype(int).values)
            raw = clf.predict_proba(X_va)
            p = np.zeros((len(X_va), len(classes)))
            for i, c in enumerate(clf.classes_): p[:, int(c)] = raw[:, i]
            rally = df_va["rally_uid"].to_numpy()
            sid,fid,cut = np.full(len(rally),seed),np.full(len(rally),fold),df_va["target_strikeNumber"].to_numpy()
            bag[tgt]["r"].append(rally);bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid);bag[tgt]["c"].append(cut);bag[tgt]["p"].append(p)

        clf_s = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
            min_samples_leaf=MIN_SAMPLES_LEAF, random_state=seed+fold*100+2000, n_jobs=-1,
            class_weight="balanced_subsample")
        clf_s.fit(X_tr, df_tr["y_serverGetPoint"].astype(int).values)
        ps = clf_s.predict_proba(X_va)[:, list(clf_s.classes_).index(1)]
        rally = df_va["rally_uid"].to_numpy()
        sid,fid,cut = np.full(len(rally),seed),np.full(len(rally),fold),df_va["target_strikeNumber"].to_numpy()
        bag["server"]["r"].append(rally);bag["server"]["s"].append(sid)
        bag["server"]["f"].append(fid);bag["server"]["c"].append(cut);bag["server"]["p"].append(ps.reshape(-1,1))
        print(f"{MODEL_NAME} seed={seed} fold={fold}", flush=True)

    for tgt in ("action","point","server"):
        r,s,f,c,p=(np.concatenate(bag[tgt][k]) for k in "rsfcp")
        write_oof(MODEL_NAME,tgt,r,s,f,c,p)

    print(f"\n[{MODEL_NAME}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df = build_one_sample_per_rally(train, splits)
    full_df = pd.concat([full_df, extra_pairs], ignore_index=True).reset_index(drop=True)
    test_feat = build_test_dataset(pd.read_csv(dd/"test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df) if c in test_feat.columns]
    X_tr_full = full_df[feats].fillna(-1).values; X_te = test_feat[feats].fillna(-1).values

    for tgt, classes, y_col in [("action", TARGET_ACTION_CLASSES, "y_actionId"),
                                  ("point", TARGET_POINT_CLASSES, "y_pointId")]:
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
            min_samples_leaf=MIN_SAMPLES_LEAF, random_state=99999+(0 if tgt=="action" else 1), n_jobs=-1,
            class_weight="balanced_subsample")
        clf.fit(X_tr_full, full_df[y_col].astype(int).values)
        raw = clf.predict_proba(X_te)
        out = np.zeros((len(X_te), len(classes)))
        for i, c in enumerate(clf.classes_): out[:, int(c)] = raw[:, i]
        _write_test_parquet(MODEL_NAME, tgt, test_feat["rally_uid"].to_numpy(), out)

    clf_s = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=99997, n_jobs=-1,
        class_weight="balanced_subsample")
    clf_s.fit(X_tr_full, full_df["y_serverGetPoint"].astype(int).values)
    ps_t = clf_s.predict_proba(X_te)[:, list(clf_s.classes_).index(1)]
    _write_test_parquet(MODEL_NAME, "server", test_feat["rally_uid"].to_numpy(), ps_t.reshape(-1,1))
    print(f"[{MODEL_NAME}] done.", flush=True)

if __name__ == "__main__":
    main()
