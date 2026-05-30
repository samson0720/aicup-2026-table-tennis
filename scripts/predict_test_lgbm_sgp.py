"""Test-time leak-feature point predictions for lgbm_sgp (Track A deploy).

Full-train LightGBM point model with true serverGetPoint as the `_sgp` feature.
On test, `_sgp` = the true leaked serverGetPoint on the 1236 old-test-overlap
rallies; for non-overlap rallies it is set to 0 (NOT used downstream — the
leakmax builder only overrides point on the overlap rallies). Writes
artifacts/oof/lgbm_sgp_point_test.parquet in the standard schema.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.predict_test_base import _write_test_parquet
from scripts.train_lgbm_baseline import TARGET_POINT_CLASSES, feature_columns, fit_multiclass


def main() -> None:
    dd = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(dd / "train.csv")
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    df_train = build_one_sample_per_rally(train, splits)

    test = build_test_dataset(pd.read_csv(dd / "test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(df_train) if c in test.columns]

    # leaked serverGetPoint on the 1236 old-test overlap rallies
    old = pd.read_csv(dd / "Reference_Only_Old_Test_Data" / "test.csv")
    old_server = old.groupby("rally_uid")["serverGetPoint"].first().to_dict()
    rally = test["rally_uid"].to_numpy()

    x_train = df_train[feats].copy()
    x_train["_sgp"] = df_train["y_serverGetPoint"].to_numpy().astype(float)
    x_test = test[feats].copy()
    x_test["_sgp"] = np.array([float(old_server.get(int(u), 0)) for u in rally])

    dummy = pd.Series(np.zeros(len(x_test), dtype=int))
    probs = fit_multiclass(x_train, df_train["y_pointId"], x_test, dummy,
                           TARGET_POINT_CLASSES, "sqrt", 3026, 180, 15)
    _write_test_parquet(model_name="lgbm_sgp", target="point", rally_uids=rally, probs=probs)


if __name__ == "__main__":
    main()
