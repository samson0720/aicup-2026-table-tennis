"""Player-conditional transition-probability features (idea 1).

Smoothed backoff P(next_target | last_stroke, next_player), which markov (player-
agnostic) does not capture. OOF-safe: count tables fit on each fold's train,
applied to its valid; for test, fit on full train. Writes markovp_{action,point}
OOF + test parquets to feed into the GBDT stack as extra bases.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.train_lgbm_baseline import build_prefix_dataset

N = {"action": 19, "point": 10}
LASTCOL = {"action": "last1_actionId", "point": "last1_pointId"}
YCOL = {"action": "y_actionId", "point": "y_pointId"}
PLAYER = "next_gamePlayerId_inferred"
ALPHA = 8.0


def fit_tables(df, target):
    n = N[target]
    y = df[YCOL[target]].to_numpy().astype(int)
    la = df[LASTCOL[target]].to_numpy()
    pl = df[PLAYER].to_numpy()
    glob = np.bincount(y, minlength=n).astype(float)
    glob = glob / glob.sum() if glob.sum() else np.full(n, 1.0 / n)
    d1 = collections.defaultdict(lambda: np.zeros(n))
    for yy, l in zip(y, la):
        d1[l][yy] += 1
    by_last = {l: (c + ALPHA * glob) / (c.sum() + ALPHA) for l, c in d1.items()}
    d2 = collections.defaultdict(lambda: np.zeros(n))
    for yy, l, p in zip(y, la, pl):
        d2[(p, l)][yy] += 1
    by_pl = {}
    for (p, l), c in d2.items():
        parent = by_last.get(l, glob)
        by_pl[(p, l)] = (c + ALPHA * parent) / (c.sum() + ALPHA)
    return glob, by_last, by_pl


def predict(df, target, tables):
    glob, by_last, by_pl = tables
    n = N[target]
    la = df[LASTCOL[target]].to_numpy()
    pl = df[PLAYER].to_numpy()
    out = np.zeros((len(df), n))
    for i, (l, p) in enumerate(zip(la, pl)):
        if (p, l) in by_pl:
            out[i] = by_pl[(p, l)]
        elif l in by_last:
            out[i] = by_last[l]
        else:
            out[i] = glob
    return out


def run_oof(args) -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point")}
    for seed, fold, tv, vv in iter_cv_folds(train, splits):
        st = splits[(splits.seed == seed) & (splits.fold != fold)]
        sv = splits[(splits.seed == seed) & (splits.fold == fold)]
        dtr = build_one_sample_per_rally(tv, st); dva = build_one_sample_per_rally(vv, sv)
        if dtr.empty or dva.empty:
            continue
        rally = dva["rally_uid"].to_numpy(); sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold); cut = dva["target_strikeNumber"].to_numpy()
        for t in ("action", "point"):
            p = predict(dva, t, fit_tables(dtr, t))
            bag[t]["r"].append(rally); bag[t]["s"].append(sid)
            bag[t]["f"].append(fid); bag[t]["c"].append(cut); bag[t]["p"].append(p)
        print(f"markovp seed={seed} fold={fold} n={len(rally)}", flush=True)
    for t in ("action", "point"):
        r = np.concatenate(bag[t]["r"]); s = np.concatenate(bag[t]["s"])
        f = np.concatenate(bag[t]["f"]); c = np.concatenate(bag[t]["c"])
        p = np.concatenate(bag[t]["p"], axis=0)
        out = write_oof("markovp", t, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)


def run_test() -> None:
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    cache = Path("artifacts/prefix_train_baseline.parquet")
    df_train = pd.read_parquet(cache) if cache.exists() else build_prefix_dataset(train)
    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)
    rally = test_features["rally_uid"].to_numpy()
    for t in ("action", "point"):
        p = predict(test_features, t, fit_tables(df_train, t))
        _write_test_parquet("markovp", t, rally, p)
        print(f"wrote markovp_{t}_test: {p.shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-test", action="store_true")
    args = ap.parse_args()
    if args.predict_test:
        run_test()
    else:
        run_oof(args)


if __name__ == "__main__":
    main()
