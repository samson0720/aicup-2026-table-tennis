"""Higher-order player x context Markov (v4 L3). Extends markovp with a
last-2-gram level and a (player, 2-gram) level via Dirichlet backoff. OOF-safe:
fit on each fold's train, apply to its valid; fit on full train for test."""
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
LAST1 = {"action": "last1_actionId", "point": "last1_pointId"}
LAST2 = {"action": "last2_actionId", "point": "last2_pointId"}
YCOL = {"action": "y_actionId", "point": "y_pointId"}
PLAYER = "next_gamePlayerId_inferred"
ALPHA = 8.0


def _smooth(counts, parent, alpha=ALPHA):
    return (counts + alpha * parent) / (counts.sum() + alpha)


def fit_tables2(df, target):
    n = N[target]
    y = df[YCOL[target]].to_numpy().astype(int)
    l1 = df[LAST1[target]].to_numpy()
    l2 = df[LAST2[target]].to_numpy()
    pl = df[PLAYER].to_numpy()
    glob = np.bincount(y, minlength=n).astype(float)
    glob = glob / glob.sum() if glob.sum() else np.full(n, 1.0 / n)
    d1 = collections.defaultdict(lambda: np.zeros(n))
    for yy, a in zip(y, l1):
        d1[a][yy] += 1
    by_l1 = {a: _smooth(c, glob) for a, c in d1.items()}
    d2 = collections.defaultdict(lambda: np.zeros(n))
    for yy, a, b in zip(y, l1, l2):
        d2[(a, b)][yy] += 1
    by_l2 = {(a, b): _smooth(c, by_l1.get(a, glob)) for (a, b), c in d2.items()}
    d3 = collections.defaultdict(lambda: np.zeros(n))
    for yy, p, a in zip(y, pl, l1):
        d3[(p, a)][yy] += 1
    by_pl1 = {(p, a): _smooth(c, by_l1.get(a, glob)) for (p, a), c in d3.items()}
    d4 = collections.defaultdict(lambda: np.zeros(n))
    for yy, p, a, b in zip(y, pl, l1, l2):
        d4[(p, a, b)][yy] += 1
    by_pl2 = {}
    for (p, a, b), c in d4.items():
        parent = by_pl1.get((p, a), by_l2.get((a, b), by_l1.get(a, glob)))
        by_pl2[(p, a, b)] = _smooth(c, parent)
    return glob, by_l1, by_l2, by_pl1, by_pl2


def predict2(df, target, tables):
    glob, by_l1, by_l2, by_pl1, by_pl2 = tables
    n = N[target]
    l1 = df[LAST1[target]].to_numpy()
    l2 = df[LAST2[target]].to_numpy()
    pl = df[PLAYER].to_numpy()
    out = np.zeros((len(df), n))
    for i, (a, b, p) in enumerate(zip(l1, l2, pl)):
        if (p, a, b) in by_pl2:
            out[i] = by_pl2[(p, a, b)]
        elif (p, a) in by_pl1:
            out[i] = by_pl1[(p, a)]
        elif (a, b) in by_l2:
            out[i] = by_l2[(a, b)]
        elif a in by_l1:
            out[i] = by_l1[a]
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
            p = predict2(dva, t, fit_tables2(dtr, t))
            bag[t]["r"].append(rally); bag[t]["s"].append(sid)
            bag[t]["f"].append(fid); bag[t]["c"].append(cut); bag[t]["p"].append(p)
        print(f"markov2 seed={seed} fold={fold} n={len(rally)}", flush=True)
    for t in ("action", "point"):
        r = np.concatenate(bag[t]["r"]); s = np.concatenate(bag[t]["s"])
        f = np.concatenate(bag[t]["f"]); c = np.concatenate(bag[t]["c"])
        p = np.concatenate(bag[t]["p"], axis=0)
        out = write_oof("markov2", t, r, s, f, c, p)
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
        p = predict2(test_features, t, fit_tables2(df_train, t))
        _write_test_parquet("markov2", t, rally, p)
        print(f"wrote markov2_{t}_test: {p.shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-test", action="store_true")
    args = ap.parse_args()
    run_test() if args.predict_test else run_oof(args)


if __name__ == "__main__":
    main()
