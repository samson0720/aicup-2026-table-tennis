"""Structured (action, point) joint base (v4 L4). New signal vs the chain:
the marginalized joint P(point) = sum_a P(point|a) P_hat(a). OOF-safe: the
conditional P(point|a) is fit on each fold's train; P_hat(a) is read from an
existing action OOF base (cat). Evaluated only through the full downstream
pipeline (build_final_perrow), never argmax-only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import read_oof, write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.train_lgbm_baseline import build_prefix_dataset

N_ACTION, N_POINT = 19, 10
ALPHA = 4.0


def fit_point_given_action(df, n_action=N_ACTION, n_point=N_POINT, alpha=ALPHA):
    a = df["y_actionId"].to_numpy().astype(int)
    p = df["y_pointId"].to_numpy().astype(int)
    glob = np.bincount(p, minlength=n_point).astype(float)
    glob = glob / glob.sum() if glob.sum() else np.full(n_point, 1.0 / n_point)
    cond = np.zeros((n_action, n_point))
    for cls in range(n_action):
        c = np.bincount(p[a == cls], minlength=n_point).astype(float)
        cond[cls] = (c + alpha * glob) / (c.sum() + alpha)
    return cond


def marginalize_point(phat_action, cond):
    out = phat_action @ cond
    return out / out.sum(axis=1, keepdims=True)


def _phat_action(model, seed, fold, rally_order):
    df = read_oof(model, "action")
    df = df[(df.seed == seed) & (df.fold == fold)].set_index("rally_uid")
    cols = [f"p_{c}" for c in range(N_ACTION)]
    return df.reindex(rally_order)[cols].to_numpy()


def run_oof(args) -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    bag = {"r": [], "s": [], "f": [], "c": [], "p": []}
    for seed, fold, tv, vv in iter_cv_folds(train, splits):
        st = splits[(splits.seed == seed) & (splits.fold != fold)]
        sv = splits[(splits.seed == seed) & (splits.fold == fold)]
        dtr = build_one_sample_per_rally(tv, st); dva = build_one_sample_per_rally(vv, sv)
        if dtr.empty or dva.empty:
            continue
        rally = dva["rally_uid"].to_numpy()
        cond = fit_point_given_action(dtr)
        phat = _phat_action(args.action_base, seed, fold, rally)
        phat = np.nan_to_num(phat, nan=1.0 / N_ACTION)
        p = marginalize_point(phat, cond)
        bag["r"].append(rally); bag["s"].append(np.full(len(rally), seed))
        bag["f"].append(np.full(len(rally), fold)); bag["c"].append(dva["target_strikeNumber"].to_numpy())
        bag["p"].append(p)
        print(f"joint seed={seed} fold={fold} n={len(rally)}", flush=True)
    r = np.concatenate(bag["r"]); s = np.concatenate(bag["s"]); f = np.concatenate(bag["f"])
    c = np.concatenate(bag["c"]); p = np.concatenate(bag["p"], axis=0)
    print("wrote", write_oof("joint", "point", r, s, f, c, p), "rows=", len(r), flush=True)


def run_test() -> None:
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    cache = Path("artifacts/prefix_train_baseline.parquet")
    df_train = pd.read_parquet(cache) if cache.exists() else build_prefix_dataset(train)
    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)
    rally = test_features["rally_uid"].to_numpy()
    cond = fit_point_given_action(df_train)
    phat = pd.read_parquet("artifacts/oof/cat_action_test.parquet").set_index("rally_uid")
    phat = phat.loc[rally, [f"p_{c}" for c in range(N_ACTION)]].to_numpy()
    p = marginalize_point(phat, cond)
    _write_test_parquet("joint", "point", rally, p)
    print(f"wrote joint_point_test: {p.shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-base", default="cat")
    ap.add_argument("--predict-test", action="store_true")
    args = ap.parse_args()
    run_test() if args.predict_test else run_oof(args)


if __name__ == "__main__":
    main()
