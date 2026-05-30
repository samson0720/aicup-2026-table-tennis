"""Build Route B chain submission and test probability parquets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feature_ngrams import ngram_features
from scripts.feature_semisupervised import build_player_feature_dist
from scripts.make_lgbm_submission import align_proba, fit_binary_full, fit_multiclass_full
from scripts.oof_loader import OOF_DIR
from scripts.postprocess import apply_thresholds, prior_correct
from scripts.target_encoding import build_player_encoders
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    add_prefix_features,
)
from scripts.train_chain_lgbm import EXCLUDE, N_ACTION, N_POINT, _merge_oof_probs


def _write_test_parquet(model: str, target: str, rally_uid: np.ndarray, probs: np.ndarray) -> None:
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    out = OOF_DIR / f"{model}_{target}_test.parquet"
    if target == "server":
        df = pd.DataFrame({"rally_uid": rally_uid.astype(np.int64), "p_1": probs[:, 0].astype(np.float32)})
    else:
        cols = {"rally_uid": rally_uid.astype(np.int64)}
        for i in range(probs.shape[1]):
            cols[f"p_{i}"] = probs[:, i].astype(np.float32)
        df = pd.DataFrame(cols)
    df.to_parquet(out, index=False)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]


def _encoder_train_frame(v2: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "player": v2["next_gamePlayerId_inferred"].astype(int),
        "phase": v2["phase"].astype(int),
        "opponent": v2["next_gamePlayerOtherId_inferred"].astype(int),
        "y_action": v2["y_actionId"].astype(int),
        "y_point": v2["y_pointId"].astype(int),
        "y_server": v2["y_serverGetPoint"].astype(int),
    })


def _encoder_apply_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "player": df["next_gamePlayerId_inferred"].astype(int),
        "phase": df["phase"].astype(int),
        "opponent": df["next_gamePlayerOtherId_inferred"].astype(int),
    })


def _build_test_features(test: pd.DataFrame, semi: pd.DataFrame, encoders) -> pd.DataFrame:
    rows: list[dict] = []
    for rally_uid, group in test.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber").reset_index(drop=True)
        target_strike = int(group["strikeNumber"].iloc[-1]) + 1
        feat = add_prefix_features(group, target_strike)
        feat.update(ngram_features(group))
        feat["rally_uid"] = int(rally_uid)
        rows.append(feat)
    df = pd.DataFrame(rows)
    df = df.merge(
        semi.add_prefix("plr_dist_")
        .reset_index()
        .rename(columns={"gamePlayerId": "next_gamePlayerId_inferred"}),
        on="next_gamePlayerId_inferred",
        how="left",
    )
    te = encoders.transform(_encoder_apply_frame(df))
    te_cols = [f"te_{i}" for i in range(te.shape[1])]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(te, columns=te_cols)], axis=1).fillna(0.0)


def _align_test_columns(train_cols: list[str], test_df: pd.DataFrame) -> pd.DataFrame:
    out = test_df.copy()
    for col in train_cols:
        if col not in out.columns:
            out[col] = 0.0
    return out[train_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)


def main() -> None:
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train_raw = pd.read_csv(data_dir / "train.csv")
    test_raw = pd.read_csv(data_dir / "test_new.csv")
    v2 = pd.read_parquet("artifacts/prefix_train_v2.parquet").replace([np.inf, -np.inf], 0.0).fillna(0.0)

    semi = build_player_feature_dist(train_raw, test_raw).set_index("gamePlayerId")
    encoders = build_player_encoders(_encoder_train_frame(v2), n_action=N_ACTION, n_point=N_POINT)
    test = _build_test_features(test_raw, semi, encoders).sort_values("rally_uid").reset_index(drop=True)
    rally_uid = test["rally_uid"].to_numpy()

    base_feats = [c for c in _feature_cols(v2) if c in test.columns]
    x_train = v2[base_feats]
    x_test = _align_test_columns(base_feats, test)

    action_model = fit_multiclass_full(
        x_train, v2["y_actionId"], TARGET_ACTION_CLASSES, "sqrt", 2026, 240, 31
    )
    p_action_test = align_proba(action_model, x_test, TARGET_ACTION_CLASSES)
    _write_test_parquet("chain_action", "action", rally_uid, p_action_test)

    point_train = _merge_oof_probs(v2, "chain_action", "action", "act", N_ACTION)
    point_test = test.copy()
    for i in range(N_ACTION):
        point_test[f"act_p_{i}"] = p_action_test[:, i]
    point_feats = [c for c in _feature_cols(point_train) if c in point_test.columns]
    point_model = fit_multiclass_full(
        point_train[point_feats], point_train["y_pointId"],
        TARGET_POINT_CLASSES, "sqrt", 3026, 240, 31,
    )
    p_point_test = align_proba(point_model, _align_test_columns(point_feats, point_test), TARGET_POINT_CLASSES)
    _write_test_parquet("chain_point", "point", rally_uid, p_point_test)

    server_train = _merge_oof_probs(point_train, "chain_point", "point", "pnt", N_POINT)
    server_test = point_test.copy()
    for i in range(N_POINT):
        server_test[f"pnt_p_{i}"] = p_point_test[:, i]
    server_feats = [c for c in _feature_cols(server_train) if c in server_test.columns]
    server_model = fit_binary_full(
        server_train[server_feats], server_train["y_serverGetPoint"], 4026, 240, 31
    )
    p_server_test = server_model.predict_proba(
        _align_test_columns(server_feats, server_test)
    )[:, list(server_model.classes_).index(1)].reshape(-1, 1)
    _write_test_parquet("chain_server", "server", rally_uid, p_server_test)

    action_prior = np.bincount(v2["y_actionId"], minlength=N_ACTION).astype(float)
    action_prior /= action_prior.sum()
    point_prior = np.bincount(v2["y_pointId"], minlength=N_POINT).astype(float)
    point_prior /= point_prior.sum()
    action_thr = np.array(json.loads(Path("artifacts/route_b_thr_action.json").read_text()))
    point_thr = np.array(json.loads(Path("artifacts/route_b_thr_point.json").read_text()))
    # Per-target prior-correction temperatures chosen during scoring (score_chain).
    # Default to the legacy beta=1 if the file predates the beta selection step.
    action_beta_p = Path("artifacts/route_b_beta_action.json")
    point_beta_p = Path("artifacts/route_b_beta_point.json")
    action_beta = json.loads(action_beta_p.read_text()) if action_beta_p.exists() else 1.0
    point_beta = json.loads(point_beta_p.read_text()) if point_beta_p.exists() else 1.0

    action_pred = apply_thresholds(prior_correct(p_action_test, action_prior, beta=action_beta), action_thr)
    point_pred = apply_thresholds(prior_correct(p_point_test, point_prior, beta=point_beta), point_thr)

    sub = pd.DataFrame({
        "rally_uid": rally_uid.astype(int),
        "actionId": action_pred.astype(int),
        "pointId": point_pred.astype(int),
        "serverGetPoint": np.clip(p_server_test[:, 0], 1e-5, 1 - 1e-5),
    })
    out = Path("artifacts/submission_B_chain.csv")
    sub.to_csv(out, index=False)
    print(f"wrote {out}: {sub.shape}")


if __name__ == "__main__":
    main()
