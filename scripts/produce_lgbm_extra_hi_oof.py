"""lgbm15_extra_hi / lgbm31_extra_hi — same as produce_extra_lgbm_oof but 300 iterations.

Tests whether more trees improve the another_data-augmented LGBM ensemble contribution.
Writes:
  artifacts/oof/lgbm15_extra_hi_{action,point,server}.parquet  (+test)
  artifacts/oof/lgbm31_extra_hi_{action,point,server}.parquet  (+test)

Usage:
  conda run -n aicup-tt python -m scripts.produce_lgbm_extra_hi_oof --model lgbm15
  conda run -n aicup-tt python -m scripts.produce_lgbm_extra_hi_oof --model lgbm31
  conda run -n aicup-tt python -m scripts.produce_lgbm_extra_hi_oof  # both
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import align_proba, build_test_dataset, fit_binary_full, fit_multiclass_full
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    feature_columns,
    fit_binary,
    fit_multiclass,
)

N_ESTIMATORS = 300


def run(num_leaves: int, model_name: str) -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    official_matches = set(train["match"].astype(str).unique())
    extra_pairs = load_extra_pairs(official_matches)

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
        x_train, x_valid = df_train[feats], df_valid[feats]

        pa = fit_multiclass(x_train, df_train["y_actionId"], x_valid, df_valid["y_actionId"],
                            TARGET_ACTION_CLASSES, "sqrt", 2026 + fold, N_ESTIMATORS, num_leaves)
        pp = fit_multiclass(x_train, df_train["y_pointId"], x_valid, df_valid["y_pointId"],
                            TARGET_POINT_CLASSES, "sqrt", 3026 + fold, N_ESTIMATORS, num_leaves)
        ps = fit_binary(x_train, df_train["y_serverGetPoint"], x_valid, df_valid["y_serverGetPoint"],
                        4026 + fold, N_ESTIMATORS, num_leaves)

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()
        for tgt, p in [("action", pa), ("point", pp), ("server", ps.reshape(-1, 1))]:
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"{model_name} seed={seed} fold={fold} n_train={len(df_train)} n_valid={len(rally)}", flush=True)

    for tgt in ("action", "point", "server"):
        r = np.concatenate(bag[tgt]["r"]); s = np.concatenate(bag[tgt]["s"])
        f = np.concatenate(bag[tgt]["f"]); c = np.concatenate(bag[tgt]["c"])
        p = np.concatenate(bag[tgt]["p"], axis=0)
        out = write_oof(model_name, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)

    # Test predictions (full train + extra)
    print(f"\n[{model_name}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df_train = build_one_sample_per_rally(train, splits)
    full_df_train = pd.concat([full_df_train, extra_pairs], ignore_index=True)

    test = build_test_dataset(pd.read_csv(dd / "test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df_train) if c in test.columns]
    x_full = full_df_train[feats]
    x_test = test[feats]
    rally_test = test["rally_uid"].to_numpy()

    ma = fit_multiclass_full(x_full, full_df_train["y_actionId"],
                             TARGET_ACTION_CLASSES, "sqrt", 2026, N_ESTIMATORS, num_leaves)
    mp = fit_multiclass_full(x_full, full_df_train["y_pointId"],
                             TARGET_POINT_CLASSES, "sqrt", 3026, N_ESTIMATORS, num_leaves)
    ms = fit_binary_full(x_full, full_df_train["y_serverGetPoint"], 4026, N_ESTIMATORS, num_leaves)

    _write_test_parquet(model_name, "action", rally_test, align_proba(ma, x_test, TARGET_ACTION_CLASSES))
    _write_test_parquet(model_name, "point", rally_test, align_proba(mp, x_test, TARGET_POINT_CLASSES))
    _write_test_parquet(model_name, "server", rally_test, ms.predict_proba(x_test)[:, 1:])
    print(f"[{model_name}] test predictions written.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgbm15", "lgbm31", "both"], default="both")
    args = ap.parse_args()
    if args.model in ("lgbm15", "both"):
        run(15, "lgbm15_extra_hi")
    if args.model in ("lgbm31", "both"):
        run(31, "lgbm31_extra_hi")


if __name__ == "__main__":
    main()
