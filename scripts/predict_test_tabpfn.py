"""Full-train TabPFN test inference -> artifacts/oof/tabpfn_{point,server}_test.parquet.

Point (10 classes) + server (binary) only (TabPFN v2 class cap). Training context
subsampled to <=10k (TabPFN sample cap). Single-cut per test rally. Runs in the
isolated aicup-tt-tabpfn env. GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

from scripts.make_lgbm_submission import build_test_dataset
from scripts.predict_test_base import _write_test_parquet
from scripts.train_lgbm_baseline import build_prefix_dataset, feature_columns

MAX_CTX = 10000


def _full_train_features(train: pd.DataFrame) -> pd.DataFrame:
    cache = Path("artifacts/prefix_train_baseline.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    df = build_prefix_dataset(train)
    df.to_parquet(cache, index=False)
    return df


def _subsample(x, y, seed):
    if len(x) <= MAX_CTX:
        return x, y
    idx = np.random.default_rng(seed).choice(len(x), MAX_CTX, replace=False)
    return x[idx], y[idx]


def _fit_predict(x_tr, y_tr, x_te, n_cls, seed):
    clf = TabPFNClassifier(device="cuda", random_state=seed)
    clf.fit(x_tr, y_tr)
    raw = clf.predict_proba(x_te)
    classes = [int(c) for c in clf.classes_]
    out = np.zeros((raw.shape[0], n_cls), dtype=np.float64)
    for i, c in enumerate(classes):
        if 0 <= c < n_cls:
            out[:, c] = raw[:, i]
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def run() -> None:
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")

    df_train = _full_train_features(train)
    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(df_train) if c in test_features.columns]
    x_train = df_train[feats].fillna(0).to_numpy(np.float32)
    x_test = test_features[feats].fillna(0).to_numpy(np.float32)
    rally = test_features["rally_uid"].to_numpy()

    yp = df_train["y_pointId"].to_numpy()
    xs, ys = _subsample(x_train, yp, 7000)
    p_point = _fit_predict(xs, ys, x_test, 10, 7000)

    yse = df_train["y_serverGetPoint"].to_numpy()
    xs2, ys2 = _subsample(x_train, yse, 7050)
    p_server = _fit_predict(xs2, ys2, x_test, 2, 7050)[:, 1].reshape(-1, 1)

    print(f"tabpfn test: point {p_point.shape}, server {p_server.shape}", flush=True)
    _write_test_parquet("tabpfn", "point", rally, p_point)
    _write_test_parquet("tabpfn", "server", rally, p_server)
    print("wrote tabpfn_{point,server}_test.parquet", flush=True)


if __name__ == "__main__":
    run()
