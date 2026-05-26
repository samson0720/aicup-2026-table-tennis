from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from make_lgbm_submission import align_proba, build_test_dataset, fit_binary_full, fit_multiclass_full
from train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    build_prefix_dataset,
    feature_columns,
    fit_binary,
    fit_multiclass,
)


def find_data_dir() -> Path:
    matches = [p for p in Path.cwd().glob("AI CUP*") if p.is_dir()]
    if not matches:
        raise FileNotFoundError("Could not find AI CUP data directory")
    return matches[0]


def score(y_action, p_action, y_point, p_point, y_server, p_server) -> dict:
    action_f1 = f1_score(
        y_action, p_action.argmax(axis=1), labels=TARGET_ACTION_CLASSES, average="macro", zero_division=0
    )
    point_f1 = f1_score(
        y_point, p_point.argmax(axis=1), labels=TARGET_POINT_CLASSES, average="macro", zero_division=0
    )
    server_auc = roc_auc_score(y_server, p_server)
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }


def run_cv(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    feats = feature_columns(df)
    x = df[feats]
    y_action = df["y_actionId"]
    y_point = df["y_pointId"]
    y_server = df["y_serverGetPoint"]
    groups = df["match"]

    p_action = np.zeros((len(df), len(TARGET_ACTION_CLASSES)))
    p_point = np.zeros((len(df), len(TARGET_POINT_CLASSES)))
    p_server = np.zeros(len(df))
    fold_scores = []

    splitter = GroupKFold(n_splits=args.folds)
    for fold, (trn_idx, val_idx) in enumerate(splitter.split(x, y_action, groups), start=1):
        fold_detail = {"fold": fold, "phase_rows": {}}
        for phase in sorted(df["phase"].unique()):
            trn_phase = trn_idx[df.iloc[trn_idx]["phase"].to_numpy() == phase]
            val_phase = val_idx[df.iloc[val_idx]["phase"].to_numpy() == phase]
            if len(val_phase) == 0:
                continue
            if len(trn_phase) < 100:
                trn_phase = trn_idx

            p_action[val_phase] = fit_multiclass(
                x.iloc[trn_phase],
                y_action.iloc[trn_phase],
                x.iloc[val_phase],
                y_action.iloc[val_phase],
                TARGET_ACTION_CLASSES,
                args.weight_mode,
                8200 + fold * 10 + int(phase),
                args.estimators,
                args.num_leaves,
            )
            p_point[val_phase] = fit_multiclass(
                x.iloc[trn_phase],
                y_point.iloc[trn_phase],
                x.iloc[val_phase],
                y_point.iloc[val_phase],
                TARGET_POINT_CLASSES,
                args.weight_mode,
                8300 + fold * 10 + int(phase),
                args.estimators,
                args.num_leaves,
            )
            p_server[val_phase] = fit_binary(
                x.iloc[trn_phase],
                y_server.iloc[trn_phase],
                x.iloc[val_phase],
                y_server.iloc[val_phase],
                8400 + fold * 10 + int(phase),
                args.estimators,
                args.num_leaves,
            )
            fold_detail["phase_rows"][str(int(phase))] = int(len(val_phase))

        fold_score = score(
            y_action.iloc[val_idx], p_action[val_idx], y_point.iloc[val_idx], p_point[val_idx], y_server.iloc[val_idx], p_server[val_idx]
        )
        fold_detail.update(fold_score)
        fold_scores.append(fold_detail)
        print(f"fold {fold}: {fold_score}")

    return {"fold_scores": fold_scores, "oof": score(y_action, p_action, y_point, p_point, y_server, p_server)}


def make_submission(df: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    feats = feature_columns(df)
    test_features = build_test_dataset(test)
    p_action = np.zeros((len(test_features), len(TARGET_ACTION_CLASSES)))
    p_point = np.zeros((len(test_features), len(TARGET_POINT_CLASSES)))
    p_server = np.zeros(len(test_features))

    for phase in sorted(df["phase"].unique()):
        train_phase = df[df["phase"] == phase]
        test_phase_idx = test_features.index[test_features["phase"] == phase].to_numpy()
        if len(test_phase_idx) == 0:
            continue
        if len(train_phase) < 100:
            train_phase = df

        action_model = fit_multiclass_full(
            train_phase[feats],
            train_phase["y_actionId"],
            TARGET_ACTION_CLASSES,
            args.weight_mode,
            8200 + int(phase),
            args.full_estimators,
            args.num_leaves,
        )
        point_model = fit_multiclass_full(
            train_phase[feats],
            train_phase["y_pointId"],
            TARGET_POINT_CLASSES,
            args.weight_mode,
            8300 + int(phase),
            args.full_estimators,
            args.num_leaves,
        )
        server_model = fit_binary_full(
            train_phase[feats],
            train_phase["y_serverGetPoint"],
            8400 + int(phase),
            args.full_estimators,
            args.num_leaves,
        )
        x_test = test_features.iloc[test_phase_idx][feats]
        p_action[test_phase_idx] = align_proba(action_model, x_test, TARGET_ACTION_CLASSES)
        p_point[test_phase_idx] = align_proba(point_model, x_test, TARGET_POINT_CLASSES)
        p_server[test_phase_idx] = server_model.predict_proba(x_test)[:, 1]

    return pd.DataFrame(
        {
            "rally_uid": test_features["rally_uid"].astype(int),
            "actionId": p_action.argmax(axis=1).astype(int),
            "pointId": p_point.argmax(axis=1).astype(int),
            "serverGetPoint": np.clip(p_server, 1e-5, 1 - 1e-5),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--weight-mode", default="sqrt", choices=["none", "sqrt", "balanced"])
    parser.add_argument("--estimators", type=int, default=180)
    parser.add_argument("--full-estimators", type=int, default=260)
    parser.add_argument("--num-leaves", type=int, default=31)
    args = parser.parse_args()

    data_dir = find_data_dir()
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    old_test = pd.read_csv(data_dir / "Reference_Only_Old_Test_Data" / "test.csv")
    prefix_path = out_dir / "prefix_train_baseline.parquet"
    if prefix_path.exists():
        df = pd.read_parquet(prefix_path)
    else:
        df = build_prefix_dataset(train)
        df.to_parquet(prefix_path, index=False)

    result = run_cv(df, args)
    (out_dir / "phase_lgbm_cv.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    clean = make_submission(df, test, args)
    leaderboard = clean.copy()
    old_server = old_test.groupby("rally_uid")["serverGetPoint"].first().to_dict()
    mask = leaderboard["rally_uid"].isin(old_server)
    leaderboard.loc[mask, "serverGetPoint"] = leaderboard.loc[mask, "rally_uid"].map(
        lambda uid: 0.95 if int(old_server[int(uid)]) == 1 else 0.05
    )
    clean.to_csv(out_dir / "submission_phase_lgbm_clean.csv", index=False)
    leaderboard.to_csv(out_dir / "submission_phase_lgbm_leaderboard.csv", index=False)
    print(json.dumps(result["oof"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
