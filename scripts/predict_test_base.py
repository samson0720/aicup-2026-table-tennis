"""Train each base model on FULL train and predict test_new rallies.

Output schema (one row per test rally, sorted by rally_uid):
  artifacts/oof/<model>_<target>_test.parquet
  rally_uid + p_0..p_{n-1} (or rally_uid + p_1 for server)

The same OOF schema is used; consumers (final stacker) can read OOF
and test parquets identically.

Models supported:
  lgbm15, lgbm31, markov, phase_lgbm
(player_stats dropped per audit F3.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.oof_loader import OOF_DIR
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    build_prefix_dataset,
    feature_columns,
)
from scripts.make_lgbm_submission import (
    align_proba,
    build_test_dataset,
    fit_binary_full,
    fit_multiclass_full,
)


def _write_test_parquet(model: str, target: str, rally_uid: np.ndarray, probs: np.ndarray) -> Path:
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    out = OOF_DIR / f"{model}_{target}_test.parquet"
    if target == "server":
        df = pd.DataFrame({"rally_uid": rally_uid.astype(np.int64), "p_1": probs[:, 0].astype(np.float32)})
    else:
        n_class = probs.shape[1]
        cols = {"rally_uid": rally_uid.astype(np.int64)}
        for c in range(n_class):
            cols[f"p_{c}"] = probs[:, c].astype(np.float32)
        df = pd.DataFrame(cols)
    df.to_parquet(out, index=False)
    return out


def _full_train_features(train: pd.DataFrame) -> pd.DataFrame:
    """Builds prefix_train if the cached parquet is absent."""
    cache = Path("artifacts/prefix_train_baseline.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    df = build_prefix_dataset(train)
    df.to_parquet(cache, index=False)
    return df


def predict_test_lgbm(num_leaves: int, model_name: str) -> None:
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")

    df_train = _full_train_features(train)
    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)

    feats = [c for c in feature_columns(df_train) if c in test_features.columns]
    x_train = df_train[feats]
    x_test = test_features[feats]

    action_model = fit_multiclass_full(
        x_train, df_train["y_actionId"], TARGET_ACTION_CLASSES, "sqrt",
        seed=2026, n_estimators=240, num_leaves=num_leaves,
    )
    point_model = fit_multiclass_full(
        x_train, df_train["y_pointId"], TARGET_POINT_CLASSES, "sqrt",
        seed=3026, n_estimators=240, num_leaves=num_leaves,
    )
    server_model = fit_binary_full(
        x_train, df_train["y_serverGetPoint"],
        seed=4026, n_estimators=240, num_leaves=num_leaves,
    )

    rally = test_features["rally_uid"].to_numpy()
    p_action = align_proba(action_model, x_test, TARGET_ACTION_CLASSES)
    p_point = align_proba(point_model, x_test, TARGET_POINT_CLASSES)
    p_server_raw = server_model.predict_proba(x_test)
    pos_idx = list(server_model.classes_).index(1)
    p_server = p_server_raw[:, pos_idx].reshape(-1, 1)

    print(f"{model_name} test: action {p_action.shape}, point {p_point.shape}, server {p_server.shape}", flush=True)
    _write_test_parquet(model_name, "action", rally, p_action)
    _write_test_parquet(model_name, "point",  rally, p_point)
    _write_test_parquet(model_name, "server", rally, p_server)
    print(f"wrote {model_name}_{{action,point,server}}_test.parquet", flush=True)


def predict_test_markov() -> None:
    """Markov base predictions for test rallies.

    Fit BackoffClassifier on the FULL train prefix dataset, predict on the
    test feature rows. BackoffClassifier is deterministic (count-based), so a
    single fit suffices.
    """
    try:
        from scripts.train_markov_ensemble import (
            ACTION_CONTEXTS, POINT_CONTEXTS, SERVER_CONTEXTS,
            BackoffClassifier, BackoffBinary,
        )
    except ImportError:  # pragma: no cover
        from train_markov_ensemble import (
            ACTION_CONTEXTS, POINT_CONTEXTS, SERVER_CONTEXTS,
            BackoffClassifier, BackoffBinary,
        )

    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    df_train = _full_train_features(train)
    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)

    phase_alpha_action = {0: 55.0, 1: 35.0, 2: 18.0}
    phase_alpha_point = {0: 60.0, 1: 40.0, 2: 20.0}
    phase_alpha_server = {0: 80.0, 1: 55.0, 2: 35.0}

    feats = [c for c in feature_columns(df_train) if c in test_features.columns]
    x_train, x_test = df_train[feats], test_features[feats]

    p_action = BackoffClassifier(
        TARGET_ACTION_CLASSES, ACTION_CONTEXTS, phase_alpha_action, 25.0
    ).fit(x_train, df_train["y_actionId"]).predict_proba(x_test)
    p_point = BackoffClassifier(
        TARGET_POINT_CLASSES, POINT_CONTEXTS, phase_alpha_point, 25.0
    ).fit(x_train, df_train["y_pointId"]).predict_proba(x_test)
    p_server = BackoffBinary(
        SERVER_CONTEXTS, phase_alpha_server, 40.0
    ).fit(x_train, df_train["y_serverGetPoint"]).predict_proba(x_test).reshape(-1, 1)

    rally = test_features["rally_uid"].to_numpy()
    print(f"markov test: action {p_action.shape}, point {p_point.shape}, server {p_server.shape}", flush=True)
    _write_test_parquet("markov", "action", rally, p_action)
    _write_test_parquet("markov", "point",  rally, p_point)
    _write_test_parquet("markov", "server", rally, p_server)
    print("wrote markov_{action,point,server}_test.parquet", flush=True)


def predict_test_phase_lgbm() -> None:
    """Phase-specific LGBM test predictions.

    For each phase bucket, train a separate LGBM on the phase subset of
    train (fall back to all train if <100 phase rows), predict on the
    test rallies whose target phase is that phase.
    """
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    df_train = _full_train_features(train)
    test_features = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)

    feats = [c for c in feature_columns(df_train) if c in test_features.columns]
    x_train, x_test = df_train[feats], test_features[feats]

    n_test = len(test_features)
    p_action = np.zeros((n_test, len(TARGET_ACTION_CLASSES)), dtype=np.float64)
    p_point = np.zeros((n_test, len(TARGET_POINT_CLASSES)),  dtype=np.float64)
    p_server = np.zeros(n_test, dtype=np.float64)

    train_phase = df_train["phase"].to_numpy()
    test_phase = test_features["phase"].to_numpy()
    for phase in sorted(set(int(p) for p in test_phase)):
        test_mask = test_phase == phase
        if not test_mask.any():
            continue
        train_mask = train_phase == phase
        if int(train_mask.sum()) < 100:
            train_mask = np.ones_like(train_phase, dtype=bool)

        xt = x_train.iloc[train_mask]
        ya = df_train.loc[train_mask, "y_actionId"]
        yp = df_train.loc[train_mask, "y_pointId"]
        ys = df_train.loc[train_mask, "y_serverGetPoint"]
        xv = x_test.iloc[test_mask]

        action_model = fit_multiclass_full(xt, ya, TARGET_ACTION_CLASSES, "sqrt",
                                           seed=8200 + int(phase), n_estimators=240, num_leaves=31)
        point_model = fit_multiclass_full(xt, yp, TARGET_POINT_CLASSES, "sqrt",
                                          seed=8300 + int(phase), n_estimators=240, num_leaves=31)
        server_model = fit_binary_full(xt, ys, seed=8400 + int(phase), n_estimators=240, num_leaves=31)

        p_action[test_mask] = align_proba(action_model, xv, TARGET_ACTION_CLASSES)
        p_point[test_mask]  = align_proba(point_model, xv, TARGET_POINT_CLASSES)
        pos_idx = list(server_model.classes_).index(1)
        p_server[test_mask] = server_model.predict_proba(xv)[:, pos_idx]
        print(f"phase_lgbm phase={phase}: n_train={int(train_mask.sum())}, n_test={int(test_mask.sum())}", flush=True)

    rally = test_features["rally_uid"].to_numpy()
    _write_test_parquet("phase_lgbm", "action", rally, p_action)
    _write_test_parquet("phase_lgbm", "point",  rally, p_point)
    _write_test_parquet("phase_lgbm", "server", rally, p_server.reshape(-1, 1))
    print("wrote phase_lgbm_{action,point,server}_test.parquet", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        choices=["lgbm15", "lgbm31", "markov", "phase_lgbm"],
    )
    args = parser.parse_args()

    if args.model == "lgbm15":
        predict_test_lgbm(15, "lgbm15")
    elif args.model == "lgbm31":
        predict_test_lgbm(31, "lgbm31")
    elif args.model == "markov":
        predict_test_markov()
    elif args.model == "phase_lgbm":
        predict_test_phase_lgbm()


if __name__ == "__main__":
    main()
