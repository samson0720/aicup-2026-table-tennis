"""Full-train XGBoost test inference -> artifacts/oof/<model>_{target}_test.parquet.

Mirrors predict_test_catboost with XGBoost GPU fits (numeric features). Single-cut
per test rally, distribution-matched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_lgbm_submission import build_test_dataset
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_xgb_oof import fit_bin, fit_mc
from scripts.train_lgbm_baseline import build_prefix_dataset, feature_columns


def _full_train_features(train: pd.DataFrame) -> pd.DataFrame:
    cache = Path("artifacts/prefix_train_baseline.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    df = build_prefix_dataset(train)
    df.to_parquet(cache, index=False)
    return df


def run(iterations: int = 600, model_name: str = "xgb", gpu: bool = True) -> None:
    device = "cuda" if gpu else "cpu"
    print(f"{model_name} test: device={device} iterations={iterations}", flush=True)
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")

    df_train = _full_train_features(train)
    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(df_train) if c in test_features.columns]
    x_train = df_train[feats].fillna(0).to_numpy(np.float32)
    x_test = test_features[feats].fillna(0).to_numpy(np.float32)
    rally = test_features["rally_uid"].to_numpy()

    p_action = fit_mc(x_train, df_train["y_actionId"], x_test, 19, 9000, iterations, device)
    p_point = fit_mc(x_train, df_train["y_pointId"], x_test, 10, 9100, iterations, device)
    p_server = fit_bin(x_train, df_train["y_serverGetPoint"], x_test, 9200, iterations, device).reshape(-1, 1)

    print(f"{model_name} test: action {p_action.shape}, point {p_point.shape}, server {p_server.shape}", flush=True)
    _write_test_parquet(model_name, "action", rally, p_action)
    _write_test_parquet(model_name, "point", rally, p_point)
    _write_test_parquet(model_name, "server", rally, p_server)
    print(f"wrote {model_name}_{{action,point,server}}_test.parquet", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--model-name", default="xgb")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(iterations=args.iterations, model_name=args.model_name, gpu=not args.cpu)


if __name__ == "__main__":
    main()
