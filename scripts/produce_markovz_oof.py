"""Player x landing-ZONE transition features (Idea 2, real-geometry dividend).

Smoothed backoff P(next_action | next_player, incoming_zone) where incoming_zone =
the cell the upcoming hitter must play from = last1_pointId (verified 3x3 grid; see
[[aicup-pointid-geometry]]). The thesis: WHERE the ball lands drives shot selection
(deep/long -> defend/transition, short -> attack) per player, a signal markovp
(conditioned on the last ACTION) does not capture for the action target.

NOTE the point target conditioned on (player, last1_pointId) is ALREADY markovp's
point base -- so this producer's novel contribution is the ACTION target only; it is
integrated action-only. Mirrors produce_markovp_oof exactly otherwise (OOF-safe count
tables fit per fold-train; full-train for test; Dirichlet alpha=8 backoff global ->
zone -> player x zone). CPU, fast.
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
ZONECOL = "last1_pointId"  # incoming landing zone for BOTH targets (the novel conditioning)
YCOL = {"action": "y_actionId", "point": "y_pointId"}
PLAYER = "next_gamePlayerId_inferred"
ALPHA = 8.0
# Default to the novel action target only (point would duplicate markovp's point base).
TARGETS = ("action",)


def fit_tables(df, target):
    n = N[target]
    y = df[YCOL[target]].to_numpy().astype(int)
    z = df[ZONECOL].to_numpy()
    pl = df[PLAYER].to_numpy()
    glob = np.bincount(y, minlength=n).astype(float)
    glob = glob / glob.sum() if glob.sum() else np.full(n, 1.0 / n)
    d1 = collections.defaultdict(lambda: np.zeros(n))
    for yy, zz in zip(y, z):
        d1[zz][yy] += 1
    by_zone = {zz: (c + ALPHA * glob) / (c.sum() + ALPHA) for zz, c in d1.items()}
    d2 = collections.defaultdict(lambda: np.zeros(n))
    for yy, zz, p in zip(y, z, pl):
        d2[(p, zz)][yy] += 1
    by_pz = {}
    for (p, zz), c in d2.items():
        parent = by_zone.get(zz, glob)
        by_pz[(p, zz)] = (c + ALPHA * parent) / (c.sum() + ALPHA)
    return glob, by_zone, by_pz


def predict(df, target, tables):
    glob, by_zone, by_pz = tables
    n = N[target]
    z = df[ZONECOL].to_numpy()
    pl = df[PLAYER].to_numpy()
    out = np.zeros((len(df), n))
    for i, (zz, p) in enumerate(zip(z, pl)):
        if (p, zz) in by_pz:
            out[i] = by_pz[(p, zz)]
        elif zz in by_zone:
            out[i] = by_zone[zz]
        else:
            out[i] = glob
    return out


def run_oof(args) -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in TARGETS}
    for seed, fold, tv, vv in iter_cv_folds(train, splits):
        st = splits[(splits.seed == seed) & (splits.fold != fold)]
        sv = splits[(splits.seed == seed) & (splits.fold == fold)]
        dtr = build_one_sample_per_rally(tv, st); dva = build_one_sample_per_rally(vv, sv)
        if dtr.empty or dva.empty:
            continue
        rally = dva["rally_uid"].to_numpy(); sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold); cut = dva["target_strikeNumber"].to_numpy()
        for t in TARGETS:
            p = predict(dva, t, fit_tables(dtr, t))
            bag[t]["r"].append(rally); bag[t]["s"].append(sid)
            bag[t]["f"].append(fid); bag[t]["c"].append(cut); bag[t]["p"].append(p)
        print(f"markovz seed={seed} fold={fold} n={len(rally)}", flush=True)
    for t in TARGETS:
        r = np.concatenate(bag[t]["r"]); s = np.concatenate(bag[t]["s"])
        f = np.concatenate(bag[t]["f"]); c = np.concatenate(bag[t]["c"])
        p = np.concatenate(bag[t]["p"], axis=0)
        out = write_oof("markovz", t, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)


def run_test() -> None:
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    cache = Path("artifacts/prefix_train_baseline.parquet")
    df_train = pd.read_parquet(cache) if cache.exists() else build_prefix_dataset(train)
    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)
    rally = test_features["rally_uid"].to_numpy()
    for t in TARGETS:
        p = predict(test_features, t, fit_tables(df_train, t))
        _write_test_parquet("markovz", t, rally, p)
        print(f"wrote markovz_{t}_test: {p.shape}", flush=True)


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
