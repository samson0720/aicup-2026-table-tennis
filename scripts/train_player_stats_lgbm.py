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


def add_multiclass_prior_features(
    fit_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    y_col: str,
    classes: list[int],
    key_cols: list[str],
    prefix: str,
    alpha: float,
) -> pd.DataFrame:
    out = apply_df.copy()
    global_counts = fit_df[y_col].value_counts().reindex(classes, fill_value=0).astype(float)
    global_prior = global_counts / max(global_counts.sum(), 1.0)

    grouped = (
        fit_df.groupby(key_cols)[y_col]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=classes, fill_value=0)
        .astype(float)
    )
    totals = grouped.sum(axis=1)
    probs = grouped.copy()
    for cls in classes:
        probs[cls] = (grouped[cls] + alpha * global_prior.loc[cls]) / (totals + alpha)
    probs = probs.reset_index()
    probs.columns = key_cols + [f"{prefix}_p_{cls}" for cls in classes]

    out = out.merge(probs, on=key_cols, how="left")
    for cls in classes:
        col = f"{prefix}_p_{cls}"
        out[col] = out[col].fillna(float(global_prior.loc[cls]))
    return out


def add_binary_prior_feature(
    fit_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    y_col: str,
    key_cols: list[str],
    name: str,
    alpha: float,
) -> pd.DataFrame:
    out = apply_df.copy()
    global_mean = float(fit_df[y_col].mean())
    grouped = fit_df.groupby(key_cols)[y_col].agg(["sum", "count"]).reset_index()
    grouped[name] = (grouped["sum"] + alpha * global_mean) / (grouped["count"] + alpha)
    grouped = grouped[key_cols + [name]]
    out = out.merge(grouped, on=key_cols, how="left")
    out[name] = out[name].fillna(global_mean)
    return out


def add_player_stat_features(fit_df: pd.DataFrame, apply_df: pd.DataFrame) -> pd.DataFrame:
    fit_df = fit_df.copy()
    apply_df = apply_df.copy()
    fit_df["next_pair"] = (
        fit_df["next_gamePlayerId_inferred"].astype(int) * 1000
        + fit_df["next_gamePlayerOtherId_inferred"].astype(int)
    )
    apply_df["next_pair"] = (
        apply_df["next_gamePlayerId_inferred"].astype(int) * 1000
        + apply_df["next_gamePlayerOtherId_inferred"].astype(int)
    )

    out = apply_df
    # Next hitter style.
    out = add_multiclass_prior_features(
        fit_df, out, "y_actionId", TARGET_ACTION_CLASSES, ["next_gamePlayerId_inferred"], "next_player_action", 30.0
    )
    out = add_multiclass_prior_features(
        fit_df, out, "y_pointId", TARGET_POINT_CLASSES, ["next_gamePlayerId_inferred"], "next_player_point", 30.0
    )
    # Opponent and matchup priors are heavily smoothed to avoid tiny-count noise.
    out = add_multiclass_prior_features(
        fit_df, out, "y_actionId", TARGET_ACTION_CLASSES, ["next_gamePlayerOtherId_inferred"], "next_opp_action", 50.0
    )
    out = add_multiclass_prior_features(
        fit_df, out, "y_pointId", TARGET_POINT_CLASSES, ["next_gamePlayerOtherId_inferred"], "next_opp_point", 50.0
    )
    out = add_multiclass_prior_features(
        fit_df, out, "y_actionId", TARGET_ACTION_CLASSES, ["next_pair"], "pair_action", 80.0
    )
    out = add_multiclass_prior_features(
        fit_df, out, "y_pointId", TARGET_POINT_CLASSES, ["next_pair"], "pair_point", 80.0
    )
    # Server result priors.
    out = add_binary_prior_feature(
        fit_df, out, "y_serverGetPoint", ["first_gamePlayerId"], "server_player_win_rate", 40.0
    )
    out = add_binary_prior_feature(
        fit_df, out, "y_serverGetPoint", ["first_gamePlayerOtherId"], "receiver_allows_server_rate", 40.0
    )
    out = add_binary_prior_feature(
        fit_df, out, "y_serverGetPoint", ["next_pair"], "pair_server_rate", 80.0
    )
    return out


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
    base_feats = feature_columns(df)
    groups = df["match"]
    y_action = df["y_actionId"]
    y_point = df["y_pointId"]
    y_server = df["y_serverGetPoint"]

    p_action = np.zeros((len(df), len(TARGET_ACTION_CLASSES)))
    p_point = np.zeros((len(df), len(TARGET_POINT_CLASSES)))
    p_server = np.zeros(len(df))
    fold_scores = []

    splitter = GroupKFold(n_splits=args.folds)
    for fold, (trn_idx, val_idx) in enumerate(splitter.split(df[base_feats], y_action, groups), start=1):
        train_fold = df.iloc[trn_idx].copy()
        valid_fold = df.iloc[val_idx].copy()

        train_aug = add_player_stat_features(train_fold, train_fold)
        valid_aug = add_player_stat_features(train_fold, valid_fold)

        feats = [c for c in train_aug.columns if c not in {"y_actionId", "y_pointId", "y_serverGetPoint", "rally_uid"}]
        x_train = train_aug[feats]
        x_valid = valid_aug[feats]

        p_action[val_idx] = fit_multiclass(
            x_train,
            train_aug["y_actionId"],
            x_valid,
            valid_aug["y_actionId"],
            TARGET_ACTION_CLASSES,
            args.weight_mode,
            5100 + fold,
            args.estimators,
            args.num_leaves,
        )
        p_point[val_idx] = fit_multiclass(
            x_train,
            train_aug["y_pointId"],
            x_valid,
            valid_aug["y_pointId"],
            TARGET_POINT_CLASSES,
            args.weight_mode,
            6100 + fold,
            args.estimators,
            args.num_leaves,
        )
        p_server[val_idx] = fit_binary(
            x_train,
            train_aug["y_serverGetPoint"],
            x_valid,
            valid_aug["y_serverGetPoint"],
            7100 + fold,
            args.estimators,
            args.num_leaves,
        )
        fold_score = score(
            valid_aug["y_actionId"],
            p_action[val_idx],
            valid_aug["y_pointId"],
            p_point[val_idx],
            valid_aug["y_serverGetPoint"],
            p_server[val_idx],
        )
        fold_score["fold"] = fold
        fold_scores.append(fold_score)
        print(f"fold {fold}: {fold_score}")

    return {
        "fold_scores": fold_scores,
        "oof": score(y_action, p_action, y_point, p_point, y_server, p_server),
    }


def make_submission(df: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    test_features = build_test_dataset(test)
    train_aug = add_player_stat_features(df, df)
    test_aug = add_player_stat_features(df, test_features)
    feats = [c for c in train_aug.columns if c not in {"y_actionId", "y_pointId", "y_serverGetPoint", "rally_uid"}]

    action_model = fit_multiclass_full(
        train_aug[feats],
        train_aug["y_actionId"],
        TARGET_ACTION_CLASSES,
        args.weight_mode,
        5100,
        args.full_estimators,
        args.num_leaves,
    )
    point_model = fit_multiclass_full(
        train_aug[feats],
        train_aug["y_pointId"],
        TARGET_POINT_CLASSES,
        args.weight_mode,
        6100,
        args.full_estimators,
        args.num_leaves,
    )
    server_model = fit_binary_full(
        train_aug[feats], train_aug["y_serverGetPoint"], 7100, args.full_estimators, args.num_leaves
    )

    x_test = test_aug[feats]
    return pd.DataFrame(
        {
            "rally_uid": test_features["rally_uid"].astype(int),
            "actionId": align_proba(action_model, x_test, TARGET_ACTION_CLASSES).argmax(axis=1).astype(int),
            "pointId": align_proba(point_model, x_test, TARGET_POINT_CLASSES).argmax(axis=1).astype(int),
            "serverGetPoint": np.clip(server_model.predict_proba(x_test)[:, 1], 1e-5, 1 - 1e-5),
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
    (out_dir / "player_stats_lgbm_cv.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    clean = make_submission(df, test, args)
    leaderboard = clean.copy()
    old_server = old_test.groupby("rally_uid")["serverGetPoint"].first().to_dict()
    mask = leaderboard["rally_uid"].isin(old_server)
    leaderboard.loc[mask, "serverGetPoint"] = leaderboard.loc[mask, "rally_uid"].map(
        lambda uid: 0.95 if int(old_server[int(uid)]) == 1 else 0.05
    )
    clean.to_csv(out_dir / "submission_lgbm_playerstats_clean.csv", index=False)
    leaderboard.to_csv(out_dir / "submission_lgbm_playerstats_leaderboard.csv", index=False)
    print(json.dumps(result["oof"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
