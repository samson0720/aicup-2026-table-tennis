from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from scripts.train_lgbm_baseline import (
        TARGET_ACTION_CLASSES,
        TARGET_POINT_CLASSES,
        add_prefix_features,
        build_prefix_dataset,
        class_weights,
        feature_columns,
    )
except ImportError:  # pragma: no cover
    from train_lgbm_baseline import (
        TARGET_ACTION_CLASSES,
        TARGET_POINT_CLASSES,
        add_prefix_features,
        build_prefix_dataset,
        class_weights,
        feature_columns,
    )


def find_data_dir() -> Path:
    matches = [p for p in Path.cwd().glob("AI CUP*") if p.is_dir()]
    if not matches:
        raise FileNotFoundError("Could not find AI CUP data directory")
    return matches[0]


def build_test_dataset(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rally_uid, grp in test.groupby("rally_uid", sort=False):
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        target_strike_number = int(grp["strikeNumber"].iloc[-1]) + 1
        feats = add_prefix_features(grp, target_strike_number)
        feats["rally_uid"] = int(rally_uid)
        rows.append(feats)
    return pd.DataFrame(rows)


def align_proba(model: lgb.LGBMClassifier, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    raw = model.predict_proba(x)
    aligned = np.zeros((len(x), len(classes)), dtype=float)
    for src_idx, cls in enumerate(model.classes_):
        if int(cls) in classes:
            aligned[:, classes.index(int(cls))] = raw[:, src_idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    missing = row_sum.squeeze() == 0
    if missing.any():
        aligned[missing, :] = 1.0 / len(classes)
        row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / row_sum


def fit_multiclass_full(
    x: pd.DataFrame,
    y: pd.Series,
    classes: list[int],
    weight_mode: str,
    seed: int,
    n_estimators: int,
    num_leaves: int,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=num_leaves,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=seed,
        n_jobs=-1,
        class_weight=class_weights(y, classes, weight_mode),
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def fit_binary_full(
    x: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_estimators: int,
    num_leaves: int,
) -> lgb.LGBMClassifier:
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=num_leaves,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=neg / pos,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-mode", default="none", choices=["none", "sqrt", "balanced"])
    parser.add_argument("--estimators", type=int, default=220)
    parser.add_argument("--num-leaves", type=int, default=31)
    args = parser.parse_args()

    data_dir = find_data_dir()
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    old_test_path = data_dir / "Reference_Only_Old_Test_Data" / "test.csv"

    prefix_path = out_dir / "prefix_train_baseline.parquet"
    if prefix_path.exists():
        train_prefix = pd.read_parquet(prefix_path)
    else:
        train_prefix = build_prefix_dataset(train)
        train_prefix.to_parquet(prefix_path, index=False)

    test_features = build_test_dataset(test)
    feats = feature_columns(train_prefix)
    x_train = train_prefix[feats]
    x_test = test_features[feats]

    action_model = fit_multiclass_full(
        x_train,
        train_prefix["y_actionId"],
        TARGET_ACTION_CLASSES,
        args.weight_mode,
        2026,
        args.estimators,
        args.num_leaves,
    )
    point_model = fit_multiclass_full(
        x_train,
        train_prefix["y_pointId"],
        TARGET_POINT_CLASSES,
        args.weight_mode,
        3026,
        args.estimators,
        args.num_leaves,
    )
    server_model = fit_binary_full(
        x_train,
        train_prefix["y_serverGetPoint"],
        4026,
        args.estimators,
        args.num_leaves,
    )

    action_pred = align_proba(action_model, x_test, TARGET_ACTION_CLASSES).argmax(axis=1)
    point_pred = align_proba(point_model, x_test, TARGET_POINT_CLASSES).argmax(axis=1)
    server_pred = server_model.predict_proba(x_test)[:, 1]

    clean = pd.DataFrame(
        {
            "rally_uid": test_features["rally_uid"].astype(int),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.clip(server_pred, 1e-5, 1 - 1e-5),
        }
    )

    clean_path = out_dir / "submission_lgbm_clean.csv"
    clean.to_csv(clean_path, index=False)

    leaderboard = clean.copy()
    if old_test_path.exists():
        old = pd.read_csv(old_test_path)
        old_server = old.groupby("rally_uid")["serverGetPoint"].first().to_dict()
        mask = leaderboard["rally_uid"].isin(old_server)
        leaderboard.loc[mask, "serverGetPoint"] = leaderboard.loc[mask, "rally_uid"].map(
            lambda uid: 0.95 if int(old_server[int(uid)]) == 1 else 0.05
        )

    leaderboard_path = out_dir / "submission_lgbm_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    print(f"Wrote {clean_path} rows={len(clean)}")
    print(f"Wrote {leaderboard_path} rows={len(leaderboard)}")
    print(clean.head().to_string(index=False))


if __name__ == "__main__":
    main()
