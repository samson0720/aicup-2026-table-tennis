"""Produce lgbm15_testpfx and lgbm31_testpfx OOFs.

Augments each fold's training set with (prefix→next_stroke) pairs from
test_new.csv visible prefix strokes (same distribution as test targets).

Key difference from produce_extra_lgbm_oof.py:
  - test prefix pairs are from the SAME test matches/players
  - serverGetPoint is unknown for test → dummy=0; server model is trained
    WITHOUT the test prefix rows (only action/point augmented)

Writes:
  artifacts/oof/lgbm15_testpfx_{action,point,server}.parquet
  artifacts/oof/lgbm15_testpfx_{action,point,server}_test.parquet
  artifacts/oof/lgbm31_testpfx_{action,point,server}.parquet
  artifacts/oof/lgbm31_testpfx_{action,point,server}_test.parquet

Usage:
  conda run -n aicup-tt python -m scripts.produce_testpfx_lgbm_oof
  conda run -n aicup-tt python -m scripts.produce_testpfx_lgbm_oof --model lgbm15
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
    TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES,
    build_prefix_dataset, feature_columns,
    fit_binary, fit_multiclass,
)


def load_test_prefix_pairs() -> pd.DataFrame:
    """Build (prefix→next) pairs from test_new.csv visible strokes."""
    test = pd.read_csv(next(Path.cwd().glob("AI CUP*/test_new.csv")))
    multi = test.groupby("rally_uid").filter(lambda g: len(g) >= 2).copy()
    multi["serverGetPoint"] = 0  # dummy — excluded from server training
    pairs = build_prefix_dataset(multi)
    print(f"[test_pairs] {len(pairs)} pairs from {multi['rally_uid'].nunique()} test rallies", flush=True)
    return pairs


def run(num_leaves: int, model_name: str) -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    test_pairs = load_test_prefix_pairs()

    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty:
            continue

        # action/point: augment with test prefix pairs
        df_train_ap = pd.concat([df_train, test_pairs], ignore_index=True)
        # server: NO augmentation (dummy serverGetPoint would hurt)
        df_train_sv = df_train

        feats = [c for c in feature_columns(df_train_ap) if c in df_valid.columns]
        x_train_ap = df_train_ap[feats]
        x_train_sv = df_train_sv[feats]
        x_valid = df_valid[feats]

        pa = fit_multiclass(x_train_ap, df_train_ap["y_actionId"], x_valid, df_valid["y_actionId"],
                            TARGET_ACTION_CLASSES, "sqrt", 2026 + fold, 180, num_leaves)
        pp = fit_multiclass(x_train_ap, df_train_ap["y_pointId"], x_valid, df_valid["y_pointId"],
                            TARGET_POINT_CLASSES, "sqrt", 3026 + fold, 180, num_leaves)
        ps = fit_binary(x_train_sv, df_train_sv["y_serverGetPoint"], x_valid, df_valid["y_serverGetPoint"],
                        4026 + fold, 180, num_leaves)

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()
        for tgt, p in [("action", pa), ("point", pp), ("server", ps.reshape(-1, 1))]:
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"{model_name} seed={seed} fold={fold} n_ap={len(df_train_ap)} n_sv={len(df_train_sv)} n_valid={len(rally)}", flush=True)

    for tgt in ("action", "point", "server"):
        r, s, f, c, p = (
            np.concatenate(bag[tgt]["r"]), np.concatenate(bag[tgt]["s"]),
            np.concatenate(bag[tgt]["f"]), np.concatenate(bag[tgt]["c"]),
            np.concatenate(bag[tgt]["p"], axis=0),
        )
        out = write_oof(model_name, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)

    # Test predictions (full train + test_pairs for action/point; train-only for server)
    print(f"\n[{model_name}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df = build_one_sample_per_rally(train, splits)
    full_df_ap = pd.concat([full_df, test_pairs], ignore_index=True)

    test = build_test_dataset(pd.read_csv(dd / "test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df_ap) if c in test.columns]
    x_test = test[feats]
    rally_test = test["rally_uid"].to_numpy()

    ma = fit_multiclass_full(full_df_ap[feats], full_df_ap["y_actionId"],
                             TARGET_ACTION_CLASSES, "sqrt", 2026, 180, num_leaves)
    mp = fit_multiclass_full(full_df_ap[feats], full_df_ap["y_pointId"],
                             TARGET_POINT_CLASSES, "sqrt", 3026, 180, num_leaves)
    # server: train-only (no augmentation)
    ms = fit_binary_full(full_df[feats], full_df["y_serverGetPoint"], 4026, 180, num_leaves)

    _write_test_parquet(model_name, "action", rally_test, align_proba(ma, x_test, TARGET_ACTION_CLASSES))
    _write_test_parquet(model_name, "point",  rally_test, align_proba(mp, x_test, TARGET_POINT_CLASSES))
    _write_test_parquet(model_name, "server", rally_test, ms.predict_proba(x_test)[:, 1:])
    print(f"[{model_name}] done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm15", "lgbm31", "both"], default="both")
    args = parser.parse_args()
    if args.model in ("lgbm15", "both"):
        run(15, "lgbm15_testpfx")
    if args.model in ("lgbm31", "both"):
        run(31, "lgbm31_testpfx")


if __name__ == "__main__":
    main()
