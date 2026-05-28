"""Produce CatBoost OOF parquets on cv_splits.parquet (per-row, honest).

Mirrors produce_base_oof.run_lgbm but uses the CatBoost helpers from
train_catboost_baseline. CPU only. Writes artifacts/oof/cat_{target}.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.oof_loader import write_oof
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    feature_columns,
)
from scripts.train_catboost_baseline import (
    cat_feature_indices,
    fit_binary,
    fit_multiclass,
    prepare_x,
)


def _stack(rs, ss, fs, cs, ps):
    return (np.concatenate(rs), np.concatenate(ss), np.concatenate(fs),
            np.concatenate(cs), np.concatenate(ps, axis=0))


def run(args) -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    task_type = "GPU" if args.gpu else "CPU"
    devices = "0" if args.gpu else None
    print(f"catboost task_type={task_type} devices={devices}", flush=True)

    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        if args.seeds and seed not in args.seeds:
            continue
        if args.folds and fold not in args.folds:
            continue
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty:
            continue
        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        cat_idx = cat_feature_indices(feats)
        cat_cols = [feats[i] for i in cat_idx]
        x_train = prepare_x(df_train[feats], cat_cols)
        x_valid = prepare_x(df_valid[feats], cat_cols)

        pa = fit_multiclass(x_train, df_train["y_actionId"], x_valid,
                            TARGET_ACTION_CLASSES, cat_idx, "sqrt", 9000 + fold, args.iterations,
                            task_type=task_type, devices=devices)
        pp = fit_multiclass(x_train, df_train["y_pointId"], x_valid,
                            TARGET_POINT_CLASSES, cat_idx, "sqrt", 9100 + fold, args.iterations,
                            task_type=task_type, devices=devices)
        ps = fit_binary(x_train, df_train["y_serverGetPoint"], x_valid,
                        cat_idx, 9200 + fold, args.iterations,
                        task_type=task_type, devices=devices).reshape(-1, 1)

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()
        for tgt, p in (("action", pa), ("point", pp), ("server", ps)):
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"cat seed={seed} fold={fold} valid_n={len(rally)}", flush=True)

    for tgt in ("action", "point", "server"):
        r, s, f, c, p = _stack(bag[tgt]["r"], bag[tgt]["s"], bag[tgt]["f"], bag[tgt]["c"], bag[tgt]["p"])
        out = write_oof("cat", tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--gpu", action="store_true", help="train CatBoost on the GPU (3090 = device 0)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
