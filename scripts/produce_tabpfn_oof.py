"""Produce TabPFN OOF parquets on cv_splits.parquet (per-row, honest).

TabPFN v2 (open weights) caps at ~10 classes and ~10k context samples, so it
covers POINT (10 classes) and SERVER (binary) only -- action (19 classes) stays
with the GBDT bases. Per-fold training context is subsampled to <=10k. GPU.

Runs in the isolated `aicup-tt-tabpfn` conda env (pip tabpfn==2.2.1), NOT the
main aicup-tt env. Writes artifacts/oof/tabpfn_{point,server}.parquet, read by
build_final_perrow in the main env.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.oof_loader import write_oof
from scripts.train_lgbm_baseline import feature_columns

MAX_CTX = 10000


def _features(df_train, df_valid):
    feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
    x_tr = df_train[feats].fillna(0).to_numpy(np.float32)
    x_va = df_valid[feats].fillna(0).to_numpy(np.float32)
    return x_tr, x_va


def _subsample(x, y, seed):
    if len(x) <= MAX_CTX:
        return x, y
    idx = np.random.default_rng(seed).choice(len(x), MAX_CTX, replace=False)
    return x[idx], y[idx]


def _fit_predict(x_tr, y_tr, x_va, n_cls, seed):
    clf = TabPFNClassifier(device="cuda", random_state=seed)
    clf.fit(x_tr, y_tr)
    raw = clf.predict_proba(x_va)
    classes = [int(c) for c in clf.classes_]
    out = np.zeros((raw.shape[0], n_cls), dtype=np.float64)
    for i, c in enumerate(classes):
        if 0 <= c < n_cls:
            out[:, c] = raw[:, i]
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def _stack(rs, ss, fs, cs, ps):
    return (np.concatenate(rs), np.concatenate(ss), np.concatenate(fs),
            np.concatenate(cs), np.concatenate(ps, axis=0))


def run(args) -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))

    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("point", "server")}
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
        x_tr, x_va = _features(df_train, df_valid)

        yp = df_train["y_pointId"].to_numpy()
        xs, ys = _subsample(x_tr, yp, seed * 100 + fold)
        pp = _fit_predict(xs, ys, x_va, 10, seed * 100 + fold)

        yse = df_train["y_serverGetPoint"].to_numpy()
        xs2, ys2 = _subsample(x_tr, yse, seed * 100 + fold + 50)
        ps_full = _fit_predict(xs2, ys2, x_va, 2, seed * 100 + fold + 50)
        ps = ps_full[:, 1].reshape(-1, 1)

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()
        for tgt, p in (("point", pp), ("server", ps)):
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"tabpfn seed={seed} fold={fold} valid_n={len(rally)}", flush=True)

    for tgt in ("point", "server"):
        r, s, f, c, p = _stack(bag[tgt]["r"], bag[tgt]["s"], bag[tgt]["f"], bag[tgt]["c"], bag[tgt]["p"])
        out = write_oof("tabpfn", tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
