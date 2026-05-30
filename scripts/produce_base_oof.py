"""Re-run base models on cv_splits.parquet and write OOF parquets.

Usage:
  python -m scripts.produce_base_oof --model lgbm15
  python -m scripts.produce_base_oof --model lgbm31
  python -m scripts.produce_base_oof --model markov
  python -m scripts.produce_base_oof --model phase_lgbm

(player_stats is dropped per audit F3 in PROGRESS.md.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally  # P1 Task 9
from scripts.oof_loader import write_oof
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    feature_columns,
    fit_binary,
    fit_multiclass,
)


def _stack(rs, ss, fs, cs, ps):
    return (
        np.concatenate(rs),
        np.concatenate(ss),
        np.concatenate(fs),
        np.concatenate(cs),
        np.concatenate(ps, axis=0),
    )


def run_lgbm(num_leaves: int, model_name: str, leak_sgp: bool = False) -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))

    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty:
            continue
        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        x_train, x_valid = df_train[feats], df_valid[feats]

        # leak-as-feature (Track A): feed known rally outcome serverGetPoint to the
        # action/point models only (server stays leak-free). Mirrors produce_catboost_oof.
        x_train_ap, x_valid_ap = x_train, x_valid
        if leak_sgp:
            x_train_ap = x_train.copy(); x_train_ap["_sgp"] = df_train["y_serverGetPoint"].to_numpy().astype(float)
            x_valid_ap = x_valid.copy(); x_valid_ap["_sgp"] = df_valid["y_serverGetPoint"].to_numpy().astype(float)

        pa = fit_multiclass(
            x_train_ap, df_train["y_actionId"], x_valid_ap, df_valid["y_actionId"],
            TARGET_ACTION_CLASSES, "sqrt", 2026 + fold, 180, num_leaves,
        )
        pp = fit_multiclass(
            x_train_ap, df_train["y_pointId"], x_valid_ap, df_valid["y_pointId"],
            TARGET_POINT_CLASSES, "sqrt", 3026 + fold, 180, num_leaves,
        )
        ps = fit_binary(
            x_train, df_train["y_serverGetPoint"], x_valid, df_valid["y_serverGetPoint"],
            4026 + fold, 180, num_leaves,
        )

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()
        bag["action"]["r"].append(rally); bag["action"]["s"].append(sid)
        bag["action"]["f"].append(fid); bag["action"]["c"].append(cut); bag["action"]["p"].append(pa)
        bag["point"]["r"].append(rally); bag["point"]["s"].append(sid)
        bag["point"]["f"].append(fid); bag["point"]["c"].append(cut); bag["point"]["p"].append(pp)
        bag["server"]["r"].append(rally); bag["server"]["s"].append(sid)
        bag["server"]["f"].append(fid); bag["server"]["c"].append(cut); bag["server"]["p"].append(ps.reshape(-1, 1))
        print(f"{model_name} seed={seed} fold={fold} valid_n={len(rally)}", flush=True)

    for tgt in ("action", "point", "server"):
        r, s, f, c, p = _stack(bag[tgt]["r"], bag[tgt]["s"], bag[tgt]["f"], bag[tgt]["c"], bag[tgt]["p"])
        out = write_oof(model_name, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)


def run_markov() -> None:
    from scripts.train_markov_ensemble import markov_oof  # added by P2 T3 refactor

    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        out = markov_oof(train_view, valid_view, s_train, s_valid)
        rally = out["rally_uid"]; cut = out["cut"]
        sid = np.full(len(rally), seed); fid = np.full(len(rally), fold)
        for tgt in ("action", "point", "server"):
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(out[tgt])
        print(f"markov seed={seed} fold={fold} valid_n={len(rally)}", flush=True)
    for tgt in ("action", "point", "server"):
        r, s, f, c, p = _stack(bag[tgt]["r"], bag[tgt]["s"], bag[tgt]["f"], bag[tgt]["c"], bag[tgt]["p"])
        write_oof("markov", tgt, r, s, f, c, p)
        print(f"wrote markov_{tgt}: rows={len(r)}", flush=True)


def run_phase_lgbm() -> None:
    from scripts.train_phase_lgbm import phase_lgbm_oof  # added by P2 T4 refactor

    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        out = phase_lgbm_oof(train_view, valid_view, s_train, s_valid)
        rally = out["rally_uid"]; cut = out["cut"]
        sid = np.full(len(rally), seed); fid = np.full(len(rally), fold)
        for tgt in ("action", "point", "server"):
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(out[tgt])
        print(f"phase_lgbm seed={seed} fold={fold} valid_n={len(rally)}", flush=True)
    for tgt in ("action", "point", "server"):
        r, s, f, c, p = _stack(bag[tgt]["r"], bag[tgt]["s"], bag[tgt]["f"], bag[tgt]["c"], bag[tgt]["p"])
        write_oof("phase_lgbm", tgt, r, s, f, c, p)
        print(f"wrote phase_lgbm_{tgt}: rows={len(r)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        choices=["lgbm15", "lgbm31", "markov", "phase_lgbm", "lgbm_sgp"],
    )
    args = parser.parse_args()

    if args.model == "lgbm15":
        run_lgbm(15, "lgbm15")
    elif args.model == "lgbm_sgp":
        run_lgbm(15, "lgbm_sgp", leak_sgp=True)
    elif args.model == "lgbm31":
        run_lgbm(31, "lgbm31")
    elif args.model == "markov":
        run_markov()
    elif args.model == "phase_lgbm":
        run_phase_lgbm()


if __name__ == "__main__":
    main()
