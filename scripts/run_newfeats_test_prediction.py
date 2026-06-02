"""Re-run only the test prediction for phase_lgbm_newfeats_extra (OOF already exists).

Usage:
  conda run -n aicup-tt python -m scripts.run_newfeats_test_prediction
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import align_proba, build_test_dataset, fit_binary_full, fit_multiclass_full
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
from scripts.train_lgbm_baseline import TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES, feature_columns
from scripts.produce_phase_lgbm_newfeats_extra_oof import add_new_features, MODEL_NAME, N_ESTIMATORS, NUM_LEAVES, MIN_PHASE_ROWS

def main():
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    official_matches = set(train["match"].astype(str).unique())
    extra_pairs = load_extra_pairs(official_matches)

    dd = next(Path.cwd().glob("AI CUP*"))
    full_df_train = build_one_sample_per_rally(train, splits)
    full_df_train = pd.concat([full_df_train, extra_pairs], ignore_index=True)
    full_df_train = add_new_features(full_df_train)

    test_raw = build_test_dataset(pd.read_csv(dd / "test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    test_raw = add_new_features(test_raw)

    rally_test = test_raw["rally_uid"].to_numpy()
    feats_full = [c for c in feature_columns(full_df_train) if c in test_raw.columns]
    x_full = full_df_train[feats_full].fillna(-1)
    x_test = test_raw[feats_full].fillna(-1)
    train_phase = full_df_train["phase"].to_numpy()
    test_phase = test_raw["phase"].to_numpy()

    p_action = np.zeros((len(test_raw), len(TARGET_ACTION_CLASSES)))
    p_point = np.zeros((len(test_raw), len(TARGET_POINT_CLASSES)))
    p_server = np.zeros(len(test_raw))

    for phase in sorted(set(int(p) for p in test_phase)):
        tst_mask = test_phase == phase
        trn_mask = train_phase == phase
        if int(trn_mask.sum()) < MIN_PHASE_ROWS:
            trn_mask = np.ones(len(train_phase), dtype=bool)
        xtr_p = x_full.iloc[trn_mask]
        xte_p = x_test.iloc[tst_mask]
        ytr = full_df_train.iloc[trn_mask]

        am = fit_multiclass_full(xtr_p, ytr["y_actionId"], TARGET_ACTION_CLASSES, "sqrt", 8200 + phase, N_ESTIMATORS, NUM_LEAVES)
        pm = fit_multiclass_full(xtr_p, ytr["y_pointId"], TARGET_POINT_CLASSES, "sqrt", 8300 + phase, N_ESTIMATORS, NUM_LEAVES)
        sm = fit_binary_full(xtr_p, ytr["y_serverGetPoint"], 8400 + phase, N_ESTIMATORS, NUM_LEAVES)

        p_action[tst_mask] = align_proba(am, xte_p, TARGET_ACTION_CLASSES)
        p_point[tst_mask] = align_proba(pm, xte_p, TARGET_POINT_CLASSES)
        p_server[tst_mask] = sm.predict_proba(xte_p)[:, list(sm.classes_).index(1)]
        print(f"phase={phase} done", flush=True)

    _write_test_parquet(MODEL_NAME, "action", rally_test, p_action)
    _write_test_parquet(MODEL_NAME, "point", rally_test, p_point)
    _write_test_parquet(MODEL_NAME, "server", rally_test, p_server.reshape(-1, 1))
    print(f"[{MODEL_NAME}] Test predictions done. {len(rally_test)} rows.", flush=True)

if __name__ == "__main__":
    main()
