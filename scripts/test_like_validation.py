from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    add_prefix_features,
    class_weights,
    feature_columns,
)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_leaves: int
    n_estimators: int
    weight_mode: str = "sqrt"


def find_data_dir() -> Path:
    matches = [p for p in Path.cwd().glob("AI CUP*") if p.is_dir()]
    if not matches:
        raise FileNotFoundError("Could not find AI CUP data directory")
    return matches[0]


def empirical_prefix_lengths(test_new: pd.DataFrame) -> np.ndarray:
    lengths = test_new.groupby("rally_uid")["strikeNumber"].max().astype(int).to_numpy()
    if len(lengths) == 0:
        raise ValueError("test_new has no rally prefix lengths")
    return lengths


def sample_prefix_len(rng: np.random.Generator, empirical_lengths: np.ndarray, rally_len: int) -> int | None:
    # Need m < rally_len so that m+1 exists as the target stroke.
    feasible = empirical_lengths[empirical_lengths < rally_len]
    if len(feasible) == 0:
        return None
    return int(rng.choice(feasible))


def build_test_like_samples(train: pd.DataFrame, test_new: pd.DataFrame, seed: int) -> pd.DataFrame:
    empirical_lengths = empirical_prefix_lengths(test_new)
    rng = np.random.default_rng(seed)
    rows = []

    for _, grp in train.groupby("rally_uid", sort=False):
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        rally_len = len(grp)
        m = sample_prefix_len(rng, empirical_lengths, rally_len)
        if m is None:
            continue

        prefix = grp.iloc[:m]
        target = grp.iloc[m]
        feats = add_prefix_features(prefix, int(target["strikeNumber"]))
        feats["y_actionId"] = int(target["actionId"])
        feats["y_pointId"] = int(target["pointId"])
        feats["y_serverGetPoint"] = int(grp["serverGetPoint"].iloc[0])
        rows.append(feats)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No test-like samples were generated")
    return df


def fit_multiclass(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    classes: list[int],
    config: ModelConfig,
    seed: int,
) -> np.ndarray:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=config.n_estimators,
        learning_rate=0.035,
        num_leaves=config.num_leaves,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=seed,
        n_jobs=-1,
        class_weight=class_weights(y_train, classes, config.weight_mode),
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    raw = model.predict_proba(x_valid)
    aligned = np.zeros((len(x_valid), len(classes)), dtype=float)
    for src_idx, cls in enumerate(model.classes_):
        if int(cls) in classes:
            aligned[:, classes.index(int(cls))] = raw[:, src_idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    missing = row_sum.squeeze() == 0
    if missing.any():
        aligned[missing, :] = 1.0 / len(classes)
        row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / row_sum


def fit_binary(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: ModelConfig,
    seed: int,
) -> np.ndarray:
    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=config.n_estimators,
        learning_rate=0.035,
        num_leaves=config.num_leaves,
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
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model.predict_proba(x_valid)[:, 1]


def score_predictions(df: pd.DataFrame, p_action: np.ndarray, p_point: np.ndarray, p_server: np.ndarray) -> dict:
    action_pred = p_action.argmax(axis=1)
    point_pred = p_point.argmax(axis=1)
    action_f1 = f1_score(
        df["y_actionId"], action_pred, labels=TARGET_ACTION_CLASSES, average="macro", zero_division=0
    )
    point_f1 = f1_score(
        df["y_pointId"], point_pred, labels=TARGET_POINT_CLASSES, average="macro", zero_division=0
    )
    server_auc = roc_auc_score(df["y_serverGetPoint"], p_server)
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }


def split_indices(df: pd.DataFrame, split_mode: str, seed: int, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    x_dummy = np.zeros(len(df))
    if split_mode == "rally":
        splitter = GroupShuffleSplit(n_splits=folds, test_size=0.2, random_state=seed)
        return list(splitter.split(x_dummy, df["y_actionId"], groups=df["rally_uid"]))
    if split_mode == "match":
        splitter = GroupKFold(n_splits=folds)
        return list(splitter.split(x_dummy, df["y_actionId"], groups=df["match"]))
    raise ValueError(f"Unknown split_mode: {split_mode}")


def evaluate_config(df: pd.DataFrame, config: ModelConfig, split_mode: str, seed: int, folds: int) -> dict:
    feats = feature_columns(df)
    x = df[feats]
    p_action = np.zeros((len(df), len(TARGET_ACTION_CLASSES)), dtype=float)
    p_point = np.zeros((len(df), len(TARGET_POINT_CLASSES)), dtype=float)
    p_server = np.zeros(len(df), dtype=float)
    fold_scores = []

    for fold, (trn_idx, val_idx) in enumerate(split_indices(df, split_mode, seed, folds), start=1):
        x_train = x.iloc[trn_idx]
        x_valid = x.iloc[val_idx]
        y_action_train = df["y_actionId"].iloc[trn_idx]
        y_point_train = df["y_pointId"].iloc[trn_idx]
        y_server_train = df["y_serverGetPoint"].iloc[trn_idx]
        y_server_valid = df["y_serverGetPoint"].iloc[val_idx]

        p_action[val_idx] = fit_multiclass(
            x_train,
            y_action_train,
            x_valid,
            TARGET_ACTION_CLASSES,
            config,
            seed * 100 + fold,
        )
        p_point[val_idx] = fit_multiclass(
            x_train,
            y_point_train,
            x_valid,
            TARGET_POINT_CLASSES,
            config,
            seed * 100 + 30 + fold,
        )
        p_server[val_idx] = fit_binary(
            x_train,
            y_server_train,
            x_valid,
            y_server_valid,
            config,
            seed * 100 + 60 + fold,
        )

        fold_df = df.iloc[val_idx]
        fold_scores.append(score_predictions(fold_df, p_action[val_idx], p_point[val_idx], p_server[val_idx]))

    return {
        "config": config.name,
        "split_mode": split_mode,
        "seed": seed,
        "fold_scores": fold_scores,
        "oof": score_predictions(df, p_action, p_point, p_server),
        "rows": int(len(df)),
        "prefix_len_mean": float(df["prefix_len"].mean()),
        "prefix_len_counts": {str(k): int(v) for k, v in df["prefix_len"].value_counts().sort_index().items()},
    }


def summarize_gate(seed_results: list[dict], preferred: str, challenger: str) -> dict:
    rows = []
    for item in seed_results:
        by_name = {cfg["config"]: cfg["oof"]["overall"] for cfg in item["configs"]}
        rows.append(
            {
                "seed": item["seed"],
                preferred: float(by_name[preferred]),
                challenger: float(by_name[challenger]),
                "diff": float(by_name[preferred] - by_name[challenger]),
            }
        )
    diffs = np.array([r["diff"] for r in rows], dtype=float)
    pref_scores = np.array([r[preferred] for r in rows], dtype=float)
    chal_scores = np.array([r[challenger] for r in rows], dtype=float)
    diff_std = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    combined_std = float(max(pref_scores.std(ddof=1), chal_scores.std(ddof=1))) if len(diffs) > 1 else 0.0
    wins = int((diffs > 0).sum())
    return {
        "preferred": preferred,
        "challenger": challenger,
        "rows": rows,
        "preferred_mean": float(pref_scores.mean()),
        "challenger_mean": float(chal_scores.mean()),
        "mean_diff": float(diffs.mean()),
        "diff_std": diff_std,
        "score_std_max": combined_std,
        "preferred_wins": wins,
        "num_seeds": int(len(diffs)),
        "passes_majority": bool(wins > len(diffs) / 2),
        "passes_mean_diff_gt_diff_std": bool(diffs.mean() > diff_std),
        "passes_mean_diff_gt_score_std_max": bool(diffs.mean() > combined_std),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--folds-rally", type=int, default=3)
    parser.add_argument("--folds-match", type=int, default=3)
    parser.add_argument("--estimators", type=int, default=180)
    args = parser.parse_args()

    data_dir = find_data_dir()
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    test_new = pd.read_csv(data_dir / "test_new.csv")
    configs = [
        ModelConfig("leaves31", num_leaves=31, n_estimators=args.estimators),
        ModelConfig("leaves15", num_leaves=15, n_estimators=args.estimators),
    ]

    all_results = {"seeds": args.seeds, "split_modes": {}}
    for split_mode, folds in [("rally", args.folds_rally), ("match", args.folds_match)]:
        seed_results = []
        print(f"\n=== split_mode={split_mode} ===")
        for seed in args.seeds:
            df = build_test_like_samples(train, test_new, seed)
            cfg_results = []
            for config in configs:
                result = evaluate_config(df, config, split_mode, seed, folds)
                cfg_results.append(result)
                print(
                    f"{split_mode} seed={seed} {config.name} overall={result['oof']['overall']:.6f} "
                    f"action={result['oof']['action_macro_f1']:.6f} point={result['oof']['point_macro_f1']:.6f} "
                    f"server={result['oof']['server_auc']:.6f}"
                )
            seed_results.append(
                {
                    "seed": seed,
                    "sample_rows": int(len(df)),
                    "prefix_len_mean": float(df["prefix_len"].mean()),
                    "configs": cfg_results,
                }
            )
        all_results["split_modes"][split_mode] = {
            "seed_results": seed_results,
            "gate": summarize_gate(seed_results, "leaves31", "leaves15"),
        }

    out_path = out_dir / "test_like_validation_gate.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(json.dumps({k: v["gate"] for k, v in all_results["split_modes"].items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
