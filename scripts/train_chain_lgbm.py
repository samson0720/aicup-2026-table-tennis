"""Route B chain LightGBM trainer.

Stages:
- chain_action predicts action from v2 features.
- chain_point predicts point from v2 features + action OOF probabilities.
- chain_server predicts serverGetPoint from v2 features + action/point OOF.

The downstream stages merge prior-stage probabilities by (rally_uid, seed,
fold), so training rows receive only fold-out predictions and avoid in-bag
leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from scripts.oof_loader import write_oof
from scripts.postprocess import apply_thresholds, prior_correct, select_beta, tune_thresholds
from scripts.score_oof import overall
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    fit_binary,
    fit_multiclass,
)


N_ACTION = 19
N_POINT = 10
EXCLUDE = {
    "rally_uid",
    "match",
    "seed",
    "fold",
    "y_actionId",
    "y_pointId",
    "y_serverGetPoint",
}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in EXCLUDE]
    bad = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise TypeError(f"non-numeric feature columns: {bad[:10]}")
    return cols


def _load_v2() -> pd.DataFrame:
    df = pd.read_parquet("artifacts/prefix_train_v2.parquet")
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _merge_oof_probs(df: pd.DataFrame, model: str, target: str, prefix: str, n_classes: int) -> pd.DataFrame:
    oof = pd.read_parquet(f"artifacts/oof/{model}_{target}.parquet")
    prob_cols = [f"p_{i}" for i in range(n_classes)] if target != "server" else ["p_1"]
    renamed = oof[["rally_uid", "seed", "fold", *prob_cols]].rename(
        columns={c: f"{prefix}_{c}" for c in prob_cols}
    )
    out = df.merge(renamed, on=["rally_uid", "seed", "fold"], how="left")
    merged_cols = [f"{prefix}_{c}" for c in prob_cols]
    if out[merged_cols].isna().any().any():
        raise ValueError(f"missing OOF probabilities after merging {model}_{target}")
    return out


def _write_stage(model: str, target: str, probs_by_fold: list[tuple[pd.DataFrame, np.ndarray]]) -> None:
    rows = [x[0] for x in probs_by_fold]
    probs = [x[1] for x in probs_by_fold]
    meta = pd.concat(rows, ignore_index=True)
    write_oof(
        model,
        target,
        meta["rally_uid"].to_numpy(),
        meta["seed"].to_numpy(),
        meta["fold"].to_numpy(),
        meta["target_strikeNumber"].to_numpy(),
        np.concatenate(probs, axis=0),
    )


def train_action() -> None:
    df = _load_v2()
    feats = _feature_cols(df)
    by_fold: list[tuple[pd.DataFrame, np.ndarray]] = []
    for (seed, fold), valid in df.groupby(["seed", "fold"], sort=False):
        train = df[(df["seed"] == seed) & (df["fold"] != fold)]
        p = fit_multiclass(
            train[feats],
            train["y_actionId"],
            valid[feats],
            valid["y_actionId"],
            TARGET_ACTION_CLASSES,
            "sqrt",
            2026 + int(fold),
            240,
            31,
        )
        by_fold.append((valid[["rally_uid", "seed", "fold", "target_strikeNumber"]], p))
        print(f"action seed={seed} fold={fold} rows={len(valid)}", flush=True)
    _write_stage("chain_action", "action", by_fold)


def train_point() -> None:
    df = _merge_oof_probs(_load_v2(), "chain_action", "action", "act", N_ACTION)
    feats = _feature_cols(df)
    by_fold: list[tuple[pd.DataFrame, np.ndarray]] = []
    for (seed, fold), valid in df.groupby(["seed", "fold"], sort=False):
        train = df[(df["seed"] == seed) & (df["fold"] != fold)]
        p = fit_multiclass(
            train[feats],
            train["y_pointId"],
            valid[feats],
            valid["y_pointId"],
            TARGET_POINT_CLASSES,
            "sqrt",
            3026 + int(fold),
            240,
            31,
        )
        by_fold.append((valid[["rally_uid", "seed", "fold", "target_strikeNumber"]], p))
        print(f"point seed={seed} fold={fold} rows={len(valid)}", flush=True)
    _write_stage("chain_point", "point", by_fold)


def train_server() -> None:
    df = _load_v2()
    df = _merge_oof_probs(df, "chain_action", "action", "act", N_ACTION)
    df = _merge_oof_probs(df, "chain_point", "point", "pnt", N_POINT)
    feats = _feature_cols(df)
    by_fold: list[tuple[pd.DataFrame, np.ndarray]] = []
    for (seed, fold), valid in df.groupby(["seed", "fold"], sort=False):
        train = df[(df["seed"] == seed) & (df["fold"] != fold)]
        p = fit_binary(
            train[feats],
            train["y_serverGetPoint"],
            valid[feats],
            valid["y_serverGetPoint"],
            4026 + int(fold),
            240,
            31,
        )
        by_fold.append((valid[["rally_uid", "seed", "fold", "target_strikeNumber"]], p.reshape(-1, 1)))
        print(f"server seed={seed} fold={fold} rows={len(valid)}", flush=True)
    _write_stage("chain_server", "server", by_fold)


def score_chain() -> dict[str, float]:
    action = pd.read_parquet("artifacts/oof/chain_action_action.parquet")
    point = pd.read_parquet("artifacts/oof/chain_point_point.parquet")
    server = pd.read_parquet("artifacts/oof/chain_server_server.parquet")
    labels = _load_v2()[[
        "rally_uid",
        "seed",
        "fold",
        "y_actionId",
        "y_pointId",
        "y_serverGetPoint",
    ]]

    action = action.merge(labels, on=["rally_uid", "seed", "fold"], how="left")
    point = point.merge(labels, on=["rally_uid", "seed", "fold"], how="left")
    server = server.merge(labels, on=["rally_uid", "seed", "fold"], how="left")

    action_probs = action[[f"p_{i}" for i in range(N_ACTION)]].to_numpy()
    point_probs = point[[f"p_{i}" for i in range(N_POINT)]].to_numpy()

    action_f1_raw = float(f1_score(
        action["y_actionId"],
        action_probs.argmax(axis=1),
        labels=TARGET_ACTION_CLASSES,
        average="macro",
        zero_division=0,
    ))
    point_f1_raw = float(f1_score(
        point["y_pointId"],
        point_probs.argmax(axis=1),
        labels=TARGET_POINT_CLASSES,
        average="macro",
        zero_division=0,
    ))

    action_prior = np.bincount(action["y_actionId"], minlength=N_ACTION).astype(float)
    action_prior /= action_prior.sum()
    point_prior = np.bincount(point["y_pointId"], minlength=N_POINT).astype(float)
    point_prior /= point_prior.sum()

    # Per-target prior-correction temperature, chosen by honest cross-seed CV
    # (beta=1 is the legacy hardcoded value; pointId prefers a milder ~0.4 and
    # actionId is hurt by full correction). Thresholds are then tuned on the
    # beta-corrected probabilities so the two steps stay consistent.
    action_beta, _ = select_beta(
        action_probs, action["y_actionId"].to_numpy(), action["seed"].to_numpy(), N_ACTION
    )
    point_beta, _ = select_beta(
        point_probs, point["y_pointId"].to_numpy(), point["seed"].to_numpy(), N_POINT
    )
    action_corrected = prior_correct(action_probs, action_prior, beta=action_beta)
    point_corrected = prior_correct(point_probs, point_prior, beta=point_beta)
    action_thresholds = tune_thresholds(action_corrected, action["y_actionId"].to_numpy(), N_ACTION)
    point_thresholds = tune_thresholds(point_corrected, point["y_pointId"].to_numpy(), N_POINT)

    action_f1 = float(f1_score(
        action["y_actionId"],
        apply_thresholds(action_corrected, action_thresholds),
        labels=TARGET_ACTION_CLASSES,
        average="macro",
        zero_division=0,
    ))
    point_f1 = float(f1_score(
        point["y_pointId"],
        apply_thresholds(point_corrected, point_thresholds),
        labels=TARGET_POINT_CLASSES,
        average="macro",
        zero_division=0,
    ))
    server_auc = float(roc_auc_score(server["y_serverGetPoint"], server["p_1"]))
    out = {
        "action_macro_f1_raw": action_f1_raw,
        "point_macro_f1_raw": point_f1_raw,
        "overall_raw": overall(action_f1_raw, point_f1_raw, server_auc),
        "action_macro_f1": action_f1,
        "point_macro_f1": point_f1,
        "server_auc": server_auc,
        "overall": overall(action_f1, point_f1, server_auc),
        "action_beta": action_beta,
        "point_beta": point_beta,
    }
    Path("artifacts/route_b_chain_scores.json").write_text(json.dumps(out, indent=2))
    Path("artifacts/route_b_thr_action.json").write_text(json.dumps(action_thresholds.tolist()))
    Path("artifacts/route_b_thr_point.json").write_text(json.dumps(point_thresholds.tolist()))
    Path("artifacts/route_b_beta_action.json").write_text(json.dumps(action_beta))
    Path("artifacts/route_b_beta_point.json").write_text(json.dumps(point_beta))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["action", "point", "server", "score", "all"], default="all")
    args = parser.parse_args()

    if args.stage in {"action", "all"}:
        train_action()
    if args.stage in {"point", "all"}:
        train_point()
    if args.stage in {"server", "all"}:
        train_server()
    if args.stage in {"score", "all"}:
        print(json.dumps(score_chain(), indent=2), flush=True)


if __name__ == "__main__":
    main()
