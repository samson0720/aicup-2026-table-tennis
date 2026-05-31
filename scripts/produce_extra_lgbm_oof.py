"""Produce lgbm15_extra and lgbm31_extra OOF + test parquets.

Identical to produce_base_oof.run_lgbm but appends another_data new-match
rows to each fold's training set.  Writes to:
  artifacts/oof/lgbm15_extra_{action,point,server}.parquet
  artifacts/oof/lgbm15_extra_{action,point,server}_test.parquet
  artifacts/oof/lgbm31_extra_{action,point,server}.parquet
  artifacts/oof/lgbm31_extra_{action,point,server}_test.parquet

Usage:
  conda run -n aicup-tt python -m scripts.produce_extra_lgbm_oof
  conda run -n aicup-tt python -m scripts.produce_extra_lgbm_oof --model lgbm15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import align_proba, build_test_dataset, fit_multiclass_full, fit_binary_full
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    build_prefix_dataset,
    feature_columns,
    fit_binary,
    fit_multiclass,
)

EXTRA_DATA_PATH = Path("another_data/train.csv")


def load_extra_pairs(official_matches: set[str]) -> pd.DataFrame:
    df = pd.read_csv(EXTRA_DATA_PATH)
    df = df.rename(columns={"strickNumber": "strikeNumber", "strickId": "strikeId"})
    df = df[df["let"] == 0].copy()
    df = df[~df["match"].astype(str).isin(official_matches)].copy()
    df = df.drop(columns=[c for c in ("serveId", "serveNumber", "let") if c in df.columns])
    sgp = df.groupby("rally_uid")["serverGetPoint"].first()
    df["serverGetPoint"] = df["rally_uid"].map(sgp)
    pairs = build_prefix_dataset(df)
    print(f"[extra] {len(pairs)} training pairs from {df['match'].nunique()} new matches", flush=True)
    return pairs


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
                            TARGET_ACTION_CLASSES, "sqrt", 2026 + fold, 180, num_leaves)
        pp = fit_multiclass(x_train, df_train["y_pointId"], x_valid, df_valid["y_pointId"],
                            TARGET_POINT_CLASSES, "sqrt", 3026 + fold, 180, num_leaves)
        ps = fit_binary(x_train, df_train["y_serverGetPoint"], x_valid, df_valid["y_serverGetPoint"],
                        4026 + fold, 180, num_leaves)

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()
        for tgt, p in [("action", pa), ("point", pp), ("server", ps.reshape(-1, 1))]:
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"{model_name} seed={seed} fold={fold} n_train={len(df_train)} n_valid={len(rally)}", flush=True)

    for tgt in ("action", "point", "server"):
        r, s, f, c, p = (
            np.concatenate(bag[tgt]["r"]), np.concatenate(bag[tgt]["s"]),
            np.concatenate(bag[tgt]["f"]), np.concatenate(bag[tgt]["c"]),
            np.concatenate(bag[tgt]["p"], axis=0),
        )
        out = write_oof(model_name, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)

    # --- Test predictions (full train + extra) ---
    print(f"\n[{model_name}] Building test predictions (full train + extra)...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    # Full train for features
    full_df_train = build_one_sample_per_rally(train, splits)
    full_df_train = pd.concat([full_df_train, extra_pairs], ignore_index=True)

    test = build_test_dataset(pd.read_csv(dd / "test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df_train) if c in test.columns]
    x_full = full_df_train[feats]
    x_test = test[feats]
    rally_test = test["rally_uid"].to_numpy()

    ma = fit_multiclass_full(x_full, full_df_train["y_actionId"],
                             TARGET_ACTION_CLASSES, "sqrt", 2026, 180, num_leaves)
    mp = fit_multiclass_full(x_full, full_df_train["y_pointId"],
                             TARGET_POINT_CLASSES, "sqrt", 3026, 180, num_leaves)
    ms = fit_binary_full(x_full, full_df_train["y_serverGetPoint"], 4026, 180, num_leaves)

    pa_test = align_proba(ma, x_test, TARGET_ACTION_CLASSES)
    pp_test = align_proba(mp, x_test, TARGET_POINT_CLASSES)
    ps_test = ms.predict_proba(x_test)[:, 1:]  # shape (n, 1)

    _write_test_parquet(model_name, "action", rally_test, pa_test)
    _write_test_parquet(model_name, "point", rally_test, pp_test)
    _write_test_parquet(model_name, "server", rally_test, ps_test)
    print(f"[{model_name}] test predictions written.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm15", "lgbm31", "both"], default="both")
    args = parser.parse_args()

    models = {"lgbm15": 15, "lgbm31": 31, "both": None}
    if args.model == "lgbm15" or args.model == "both":
        run(15, "lgbm15_extra")
    if args.model == "lgbm31" or args.model == "both":
        run(31, "lgbm31_extra")


if __name__ == "__main__":
    main()
