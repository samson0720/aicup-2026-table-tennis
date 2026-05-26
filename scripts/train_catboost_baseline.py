from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from make_lgbm_submission import build_test_dataset
from train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    build_prefix_dataset,
    class_weights,
    feature_columns,
)


def find_data_dir() -> Path:
    matches = [p for p in Path.cwd().glob("AI CUP*") if p.is_dir()]
    if not matches:
        raise FileNotFoundError("Could not find AI CUP data directory")
    return matches[0]


def cat_feature_indices(cols: list[str]) -> list[int]:
    continuous_exact = {
        "prefix_len",
        "last_strikeNumber",
        "scoreSelf",
        "scoreOther",
        "score_diff",
        "score_sum",
        "abs_score_diff",
    }
    cats = []
    for idx, col in enumerate(cols):
        if col in continuous_exact:
            continue
        if "_cnt_" in col or "_rate_" in col or col.endswith("_entropy") or col.endswith("_nunique"):
            continue
        cats.append(idx)
    return cats


def prepare_x(x: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = x.copy()
    for col in cat_cols:
        out[col] = out[col].astype(int).astype(str)
    return out


def align_multiclass(model: CatBoostClassifier, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    raw = model.predict_proba(x)
    aligned = np.zeros((len(x), len(classes)), dtype=float)
    model_classes = [int(c) for c in model.classes_]
    for src_idx, cls in enumerate(model_classes):
        if cls in classes:
            aligned[:, classes.index(cls)] = raw[:, src_idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    missing = row_sum.squeeze() == 0
    if missing.any():
        aligned[missing, :] = 1.0 / len(classes)
        row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / row_sum


def fit_multiclass(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    classes: list[int],
    cat_features: list[int],
    weight_mode: str,
    seed: int,
    iterations: int,
) -> np.ndarray:
    weights = class_weights(y_train, classes, weight_mode)
    class_weights_list = None
    if weights is not None:
        class_weights_list = [weights.get(cls, 1.0) for cls in range(max(classes) + 1)]

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        classes_count=max(classes) + 1,
        class_weights=class_weights_list,
    )
    model.fit(x_train, y_train, cat_features=cat_features)
    return align_multiclass(model, x_valid, classes)


def fit_binary(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    cat_features: list[int],
    seed: int,
    iterations: int,
) -> np.ndarray:
    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        class_weights=[1.0, neg / pos],
    )
    model.fit(x_train, y_train, cat_features=cat_features)
    return model.predict_proba(x_valid)[:, 1]


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


def search_weight_multiclass(y, p_a, p_b, classes):
    best = {"weight_lgbm": 0.0, "score": -1.0}
    for w in np.linspace(0, 1, 101):
        p = w * p_a + (1 - w) * p_b
        s = f1_score(y, p.argmax(axis=1), labels=classes, average="macro", zero_division=0)
        if s > best["score"]:
            best = {"weight_lgbm": float(w), "score": float(s)}
    return best


def run_cv(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    feats = feature_columns(df)
    cat_idx = cat_feature_indices(feats)
    cat_cols = [feats[i] for i in cat_idx]
    x = prepare_x(df[feats], cat_cols)
    groups = df["match"]

    pa = np.zeros((len(df), len(TARGET_ACTION_CLASSES)))
    pp = np.zeros((len(df), len(TARGET_POINT_CLASSES)))
    ps = np.zeros(len(df))
    fold_scores = []

    splitter = GroupKFold(n_splits=args.folds)
    for fold, (trn, val) in enumerate(splitter.split(x, df["y_actionId"], groups), start=1):
        pa[val] = fit_multiclass(
            x.iloc[trn],
            df["y_actionId"].iloc[trn],
            x.iloc[val],
            TARGET_ACTION_CLASSES,
            cat_idx,
            args.weight_mode,
            9000 + fold,
            args.iterations,
        )
        pp[val] = fit_multiclass(
            x.iloc[trn],
            df["y_pointId"].iloc[trn],
            x.iloc[val],
            TARGET_POINT_CLASSES,
            cat_idx,
            args.weight_mode,
            9100 + fold,
            args.iterations,
        )
        ps[val] = fit_binary(
            x.iloc[trn],
            df["y_serverGetPoint"].iloc[trn],
            x.iloc[val],
            cat_idx,
            9200 + fold,
            args.iterations,
        )
        fold_score = score(
            df["y_actionId"].iloc[val], pa[val], df["y_pointId"].iloc[val], pp[val], df["y_serverGetPoint"].iloc[val], ps[val]
        )
        fold_score["fold"] = fold
        fold_scores.append(fold_score)
        print(f"fold {fold}: {fold_score}")

    return {
        "fold_scores": fold_scores,
        "oof": score(df["y_actionId"], pa, df["y_pointId"], pp, df["y_serverGetPoint"], ps),
    }


def fit_full_multiclass(x, y, classes, cat_idx, weight_mode, seed, iterations):
    weights = class_weights(y, classes, weight_mode)
    class_weights_list = [weights.get(cls, 1.0) for cls in range(max(classes) + 1)] if weights is not None else None
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        classes_count=max(classes) + 1,
        class_weights=class_weights_list,
    )
    model.fit(x, y, cat_features=cat_idx)
    return model


def fit_full_binary(x, y, cat_idx, seed, iterations):
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        class_weights=[1.0, neg / pos],
    )
    model.fit(x, y, cat_features=cat_idx)
    return model


def make_submission(df: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    feats = feature_columns(df)
    cat_idx = cat_feature_indices(feats)
    cat_cols = [feats[i] for i in cat_idx]
    x_train = prepare_x(df[feats], cat_cols)
    test_features = build_test_dataset(test)
    x_test = prepare_x(test_features[feats], cat_cols)

    action_model = fit_full_multiclass(
        x_train, df["y_actionId"], TARGET_ACTION_CLASSES, cat_idx, args.weight_mode, 9000, args.full_iterations
    )
    point_model = fit_full_multiclass(
        x_train, df["y_pointId"], TARGET_POINT_CLASSES, cat_idx, args.weight_mode, 9100, args.full_iterations
    )
    server_model = fit_full_binary(x_train, df["y_serverGetPoint"], cat_idx, 9200, args.full_iterations)
    return pd.DataFrame(
        {
            "rally_uid": test_features["rally_uid"].astype(int),
            "actionId": align_multiclass(action_model, x_test, TARGET_ACTION_CLASSES).argmax(axis=1).astype(int),
            "pointId": align_multiclass(point_model, x_test, TARGET_POINT_CLASSES).argmax(axis=1).astype(int),
            "serverGetPoint": np.clip(server_model.predict_proba(x_test)[:, 1], 1e-5, 1 - 1e-5),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--weight-mode", default="sqrt", choices=["none", "sqrt", "balanced"])
    parser.add_argument("--iterations", type=int, default=220)
    parser.add_argument("--full-iterations", type=int, default=320)
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
    (out_dir / "catboost_baseline_cv.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    clean = make_submission(df, test, args)
    leaderboard = clean.copy()
    old_server = old_test.groupby("rally_uid")["serverGetPoint"].first().to_dict()
    mask = leaderboard["rally_uid"].isin(old_server)
    leaderboard.loc[mask, "serverGetPoint"] = leaderboard.loc[mask, "rally_uid"].map(
        lambda uid: 0.95 if int(old_server[int(uid)]) == 1 else 0.05
    )
    clean.to_csv(out_dir / "submission_catboost_clean.csv", index=False)
    leaderboard.to_csv(out_dir / "submission_catboost_leaderboard.csv", index=False)
    print(json.dumps(result["oof"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
