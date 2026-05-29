"""XGBoost GPU OOF producer (per-row, honest). Mirrors produce_catboost_oof.

Numeric features (like the LGBM bases) + sqrt class sample-weights. A different
GBDT family for ensemble diversity. Writes artifacts/oof/<model>_{target}.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.oof_loader import write_oof
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    class_weights,
    feature_columns,
)


def _sample_weight(y: pd.Series, n_cls: int, mode: str):
    cw = class_weights(y, list(range(n_cls)), mode)
    if cw is None:
        return None
    return y.map(lambda c: cw.get(int(c), 1.0)).to_numpy()


def fit_mc(x_tr, y_tr, x_va, n_cls, seed, n_est, device):
    # XGBoost's sklearn wrapper needs contiguous labels; a fold can miss a class.
    le = LabelEncoder().fit(y_tr)
    clf = xgb.XGBClassifier(objective="multi:softprob", tree_method="hist",
                            device=device, n_estimators=n_est, max_depth=6, learning_rate=0.05,
                            reg_lambda=5.0, subsample=0.9, colsample_bytree=0.9,
                            random_state=seed, verbosity=0)
    clf.fit(x_tr, le.transform(y_tr), sample_weight=_sample_weight(y_tr, n_cls, "sqrt"))
    raw = clf.predict_proba(x_va)
    out = np.zeros((raw.shape[0], n_cls), dtype=np.float64)
    for enc_i, orig_cls in enumerate(le.classes_):
        out[:, int(orig_cls)] = raw[:, enc_i]
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def fit_bin(x_tr, y_tr, x_va, seed, n_est, device):
    pos = max(int((y_tr == 1).sum()), 1)
    neg = max(int((y_tr == 0).sum()), 1)
    clf = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist", device=device,
                            n_estimators=n_est, max_depth=6, learning_rate=0.05, reg_lambda=5.0,
                            scale_pos_weight=neg / pos, random_state=seed, verbosity=0)
    clf.fit(x_tr, y_tr)
    return clf.predict_proba(x_va)[:, list(clf.classes_).index(1)]


def _stack(rs, ss, fs, cs, ps):
    return (np.concatenate(rs), np.concatenate(ss), np.concatenate(fs),
            np.concatenate(cs), np.concatenate(ps, axis=0))


def run(args) -> None:
    device = "cuda" if args.gpu else "cpu"
    print(f"xgb device={device}", flush=True)
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))

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
        x_train = df_train[feats].fillna(0).to_numpy(np.float32)
        x_valid = df_valid[feats].fillna(0).to_numpy(np.float32)

        pa = fit_mc(x_train, df_train["y_actionId"], x_valid, 19, 9000 + fold, args.iterations, device)
        pp = fit_mc(x_train, df_train["y_pointId"], x_valid, 10, 9100 + fold, args.iterations, device)
        ps = fit_bin(x_train, df_train["y_serverGetPoint"], x_valid, 9200 + fold, args.iterations, device).reshape(-1, 1)

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed); fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()
        for tgt, p in (("action", pa), ("point", pp), ("server", ps)):
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"xgb seed={seed} fold={fold} valid_n={len(rally)}", flush=True)

    for tgt in ("action", "point", "server"):
        r, s, f, c, p = _stack(bag[tgt]["r"], bag[tgt]["s"], bag[tgt]["f"], bag[tgt]["c"], bag[tgt]["p"])
        out = write_oof(args.model_name, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--model-name", default="xgb")
    run(p.parse_args())


if __name__ == "__main__":
    main()
