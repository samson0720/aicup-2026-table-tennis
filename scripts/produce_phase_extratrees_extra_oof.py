"""Phase-specific ExtraTreesClassifier + another_data OOF + test parquets.

Combines phase splitting (3 phases) with ExtraTrees + another_data augmentation.
Run only if plain extratrees_extra passes the ensemble gate.

CPU-only. Writes:
  artifacts/oof/phase_extratrees_extra_{action,point,server}.parquet (+test)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
from scripts.train_lgbm_baseline import TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES, feature_columns

MODEL_NAME = "phase_extratrees_extra"
N_ESTIMATORS = 300
MIN_SAMPLES_LEAF = 5
MIN_PHASE_ROWS = 100

def _phase_fit(df_tr, df_va, feats, seed_base):
    X_tr = df_tr[feats].fillna(-1).values; X_va = df_va[feats].fillna(-1).values
    df_tr = df_tr.reset_index(drop=True); df_va = df_va.reset_index(drop=True)
    pa = np.zeros((len(df_va), len(TARGET_ACTION_CLASSES)))
    pp = np.zeros((len(df_va), len(TARGET_POINT_CLASSES)))
    ps = np.zeros(len(df_va))
    tp = df_tr["phase"].to_numpy(); vp = df_va["phase"].to_numpy()
    for phase in sorted(set(int(p) for p in vp)):
        vm = vp == phase
        if not vm.any(): continue
        tm = tp == phase
        if int(tm.sum()) < MIN_PHASE_ROWS: tm = np.ones_like(tp, dtype=bool)
        xt = X_tr[tm]; xv = X_va[vm]
        for tgt, arr, classes in [("action", pa, TARGET_ACTION_CLASSES), ("point", pp, TARGET_POINT_CLASSES)]:
            y_tgt = "y_actionId" if tgt == "action" else "y_pointId"
            clf = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
                min_samples_leaf=MIN_SAMPLES_LEAF, random_state=seed_base+int(phase)*10+("action"==tgt),
                n_jobs=-1, class_weight="balanced_subsample")
            clf.fit(xt, df_tr.loc[tm, y_tgt].astype(int).values)
            raw = clf.predict_proba(xv)
            for i, c in enumerate(clf.classes_): arr[vm, int(c)] = raw[:, i]
        clf_s = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
            min_samples_leaf=MIN_SAMPLES_LEAF, random_state=seed_base+int(phase)*10+2,
            n_jobs=-1, class_weight="balanced_subsample")
        clf_s.fit(xt, df_tr.loc[tm, "y_serverGetPoint"].astype(int).values)
        ps[vm] = clf_s.predict_proba(xv)[:, list(clf_s.classes_).index(1)]
    return pa, pp, ps

def main():
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    extra = load_extra_pairs(set(train["match"].astype(str).unique()))
    print(f"{MODEL_NAME}: phase-specific ExtraTrees", flush=True)
    bag = {t: {"r":[],"s":[],"f":[],"c":[],"p":[]} for t in ("action","point","server")}
    for seed, fold, tv, vv in iter_cv_folds(train, splits):
        st = splits[(splits["seed"]==seed)&(splits["fold"]!=fold)]
        sv = splits[(splits["seed"]==seed)&(splits["fold"]==fold)]
        df_tr = build_one_sample_per_rally(tv, st)
        df_va = build_one_sample_per_rally(vv, sv)
        if df_tr.empty or df_va.empty: continue
        df_tr = pd.concat([df_tr, extra], ignore_index=True)
        feats = [c for c in feature_columns(df_tr) if c in df_va.columns]
        pa, pp, ps = _phase_fit(df_tr, df_va, feats, seed*100+fold*10)
        rally = df_va["rally_uid"].to_numpy()
        sid,fid,cut = np.full(len(rally),seed),np.full(len(rally),fold),df_va["target_strikeNumber"].to_numpy()
        for tgt,p in [("action",pa),("point",pp),("server",ps.reshape(-1,1))]:
            bag[tgt]["r"].append(rally);bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid);bag[tgt]["c"].append(cut);bag[tgt]["p"].append(p)
        print(f"{MODEL_NAME} seed={seed} fold={fold}", flush=True)
    for tgt in ("action","point","server"):
        r,s,f,c,p=(np.concatenate(bag[tgt][k]) for k in "rsfcp")
        write_oof(MODEL_NAME,tgt,r,s,f,c,p)
    print(f"\n[{MODEL_NAME}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df = build_one_sample_per_rally(train,splits)
    full_df = pd.concat([full_df,extra],ignore_index=True).reset_index(drop=True)
    test_feat = build_test_dataset(pd.read_csv(dd/"test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df) if c in test_feat.columns]
    pa,pp,ps = _phase_fit(full_df,test_feat,feats,99000)
    rally_test = test_feat["rally_uid"].to_numpy()
    _write_test_parquet(MODEL_NAME,"action",rally_test,pa)
    _write_test_parquet(MODEL_NAME,"point",rally_test,pp)
    _write_test_parquet(MODEL_NAME,"server",rally_test,ps.reshape(-1,1))
    print(f"[{MODEL_NAME}] done.", flush=True)

if __name__ == "__main__":
    main()
