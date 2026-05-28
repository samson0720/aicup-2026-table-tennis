"""Full-train CatBoost test inference -> artifacts/oof/cat_{target}_test.parquet.

Mirrors predict_test_base.predict_test_lgbm with the CatBoost full-train fits.
Single-cut per test rally; distribution-matched to the OOF models. GPU by
default (3090 = device 0); pass --cpu to force CPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_lgbm_submission import build_test_dataset
from scripts.predict_test_base import _write_test_parquet
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    build_prefix_dataset,
    feature_columns,
)
from scripts.train_catboost_baseline import (
    align_multiclass,
    cat_feature_indices,
    fit_full_binary,
    fit_full_multiclass,
    prepare_x,
)


def _full_train_features(train: pd.DataFrame) -> pd.DataFrame:
    cache = Path("artifacts/prefix_train_baseline.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    df = build_prefix_dataset(train)
    df.to_parquet(cache, index=False)
    return df


def run(iterations: int = 600, gpu: bool = True) -> None:
    task_type = "GPU" if gpu else "CPU"
    devices = "0" if gpu else None
    print(f"cat test: task_type={task_type} devices={devices} iterations={iterations}", flush=True)

    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")

    df_train = _full_train_features(train)
    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(df_train) if c in test_features.columns]
    cat_idx = cat_feature_indices(feats)
    cat_cols = [feats[i] for i in cat_idx]
    x_train = prepare_x(df_train[feats], cat_cols)
    x_test = prepare_x(test_features[feats], cat_cols)

    action_model = fit_full_multiclass(x_train, df_train["y_actionId"], TARGET_ACTION_CLASSES, cat_idx, "sqrt", 9000, iterations, task_type=task_type, devices=devices)
    point_model = fit_full_multiclass(x_train, df_train["y_pointId"], TARGET_POINT_CLASSES, cat_idx, "sqrt", 9100, iterations, task_type=task_type, devices=devices)
    server_model = fit_full_binary(x_train, df_train["y_serverGetPoint"], cat_idx, 9200, iterations, task_type=task_type, devices=devices)

    rally = test_features["rally_uid"].to_numpy()
    p_action = align_multiclass(action_model, x_test, TARGET_ACTION_CLASSES)
    p_point = align_multiclass(point_model, x_test, TARGET_POINT_CLASSES)
    pos_idx = list(server_model.classes_).index(1)
    p_server = server_model.predict_proba(x_test)[:, pos_idx].reshape(-1, 1)

    print(f"cat test: action {p_action.shape}, point {p_point.shape}, server {p_server.shape}", flush=True)
    _write_test_parquet("cat", "action", rally, p_action)
    _write_test_parquet("cat", "point", rally, p_point)
    _write_test_parquet("cat", "server", rally, p_server)
    print("wrote cat_{action,point,server}_test.parquet", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(iterations=args.iterations, gpu=not args.cpu)


if __name__ == "__main__":
    main()
