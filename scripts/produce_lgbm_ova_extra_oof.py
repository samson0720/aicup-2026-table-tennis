"""LightGBM multiclassova (one-vs-all) + another_data OOF + test parquets.

Pilot seed11x3folds: OVA action argmax F1 = 0.21725 vs multiclass 0.20809 (+0.00916).
OVA trains 19 binary classifiers instead of one multiclass — better for imbalanced
rare action classes. May provide orthogonal signal to existing 19-base ensemble.

CPU-only. Writes:
  artifacts/oof/lgbm_ova_extra_{action,point,server}.parquet (+test)

Usage:
  conda run -n aicup-tt python -m scripts.produce_lgbm_ova_extra_oof
"""
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import align_proba, build_test_dataset, fit_binary_full, fit_multiclass_full
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES, feature_columns,
    class_weights, categorical_feature_names, fit_binary,
)

MODEL_NAME = "lgbm_ova_extra"
N_ESTIMATORS = 180
NUM_LEAVES = 31


def fit_ova(x_tr, y_tr, x_va, classes, seed):
    cw = class_weights(y_tr, classes, "sqrt")
    sw = y_tr.map(cw).fillna(1.0)
    cat_feats = categorical_feature_names(list(x_tr.columns))
    model = lgb.LGBMClassifier(
        objective="multiclassova", num_class=len(classes),
        n_estimators=N_ESTIMATORS, learning_rate=0.035, num_leaves=NUM_LEAVES,
        min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.05, reg_lambda=0.2, random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(x_tr, y_tr, sample_weight=sw, categorical_feature=cat_feats)
    raw = model.predict_proba(x_va)
    out = np.zeros((raw.shape[0], len(classes)), dtype=np.float64)
    for i, c in enumerate(model.classes_):
        out[:, int(c)] = raw[:, i]
    s = out.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return out / s


def main():
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    extra_pairs = load_extra_pairs(set(train["match"].astype(str).unique()))
    print(f"{MODEL_NAME}: multiclassova, {NUM_LEAVES} leaves", flush=True)

    bag = {t: {"r":[],"s":[],"f":[],"c":[],"p":[]} for t in ("action","point","server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"]==seed)&(splits["fold"]!=fold)]
        s_valid = splits[(splits["seed"]==seed)&(splits["fold"]==fold)]
        df_tr = build_one_sample_per_rally(train_view, s_train)
        df_va = build_one_sample_per_rally(valid_view, s_valid)
        if df_tr.empty or df_va.empty: continue
        df_tr = pd.concat([df_tr, extra_pairs], ignore_index=True)
        feats = [c for c in feature_columns(df_tr) if c in df_va.columns]

        pa = fit_ova(df_tr[feats], df_tr["y_actionId"], df_va[feats], TARGET_ACTION_CLASSES, seed+fold*100)
        pp = fit_ova(df_tr[feats], df_tr["y_pointId"], df_va[feats], TARGET_POINT_CLASSES, seed+fold*100+1000)
        ps = fit_binary(df_tr[feats], df_tr["y_serverGetPoint"], df_va[feats],
                        df_va["y_serverGetPoint"], seed+fold*100+2000, N_ESTIMATORS, NUM_LEAVES)

        rally = df_va["rally_uid"].to_numpy()
        sid,fid,cut = np.full(len(rally),seed),np.full(len(rally),fold),df_va["target_strikeNumber"].to_numpy()
        for tgt,p in [("action",pa),("point",pp),("server",ps.reshape(-1,1))]:
            bag[tgt]["r"].append(rally);bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid);bag[tgt]["c"].append(cut);bag[tgt]["p"].append(p)
        print(f"{MODEL_NAME} seed={seed} fold={fold} n_train={len(df_tr)}", flush=True)

    for tgt in ("action","point","server"):
        r,s,f,c,p=(np.concatenate(bag[tgt][k]) for k in "rsfcp")
        write_oof(MODEL_NAME,tgt,r,s,f,c,p)

    print(f"\n[{MODEL_NAME}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df = build_one_sample_per_rally(train, splits)
    full_df = pd.concat([full_df, extra_pairs], ignore_index=True).reset_index(drop=True)
    test_feat = build_test_dataset(pd.read_csv(dd/"test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df) if c in test_feat.columns]
    cat_feats = categorical_feature_names(feats)

    for tgt, classes in [("action", TARGET_ACTION_CLASSES), ("point", TARGET_POINT_CLASSES)]:
        y_col = "y_actionId" if tgt == "action" else "y_pointId"
        cw = class_weights(full_df[y_col], classes, "sqrt")
        sw = full_df[y_col].map(cw).fillna(1.0)
        m = lgb.LGBMClassifier(objective="multiclassova", num_class=len(classes),
            n_estimators=N_ESTIMATORS, learning_rate=0.035, num_leaves=NUM_LEAVES,
            min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.05, reg_lambda=0.2, random_state=99999, n_jobs=-1, verbose=-1)
        m.fit(full_df[feats], full_df[y_col], sample_weight=sw, categorical_feature=cat_feats)
        raw = m.predict_proba(test_feat[feats])
        out = np.zeros((len(test_feat), len(classes)))
        for i, c in enumerate(m.classes_): out[:, int(c)] = raw[:, i]
        s = out.sum(axis=1, keepdims=True); s[s==0] = 1.0
        _write_test_parquet(MODEL_NAME, tgt, test_feat["rally_uid"].to_numpy(), out/s)

    # server (binary)
    m_s = fit_binary_full(full_df[feats], full_df["y_serverGetPoint"],
                          seed=99997, n_estimators=N_ESTIMATORS, num_leaves=NUM_LEAVES)
    ps_t = m_s.predict_proba(align_proba(m_s, test_feat[feats], [0,1]).reshape(-1,2) if False else test_feat[feats])[:, 1]
    _write_test_parquet(MODEL_NAME, "server", test_feat["rally_uid"].to_numpy(), ps_t.reshape(-1,1))
    print(f"[{MODEL_NAME}] done.", flush=True)

if __name__ == "__main__":
    main()
