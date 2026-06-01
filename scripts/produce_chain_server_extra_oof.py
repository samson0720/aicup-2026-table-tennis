"""Chain-server model augmented with another_data (chain_server_extra).

Strategy:
- Chain pipeline: action → point → server (server sees action+point probs as features).
- Augment training with another_data new-match rows for the server step only.
- For another_data rows, chain probs come from full-data models (slight train-time
  optimism, but acceptable approximation; these rows are never in validation).

Writes:
  artifacts/oof/chain_server_extra_server.parquet
  artifacts/oof/chain_server_extra_server_test.parquet

Usage:
  conda run -n aicup-tt python -m scripts.produce_chain_server_extra_oof
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_lgbm_submission import align_proba, fit_binary_full, fit_multiclass_full
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
from scripts.train_chain_lgbm import (
    EXCLUDE,
    N_ACTION,
    N_POINT,
    _feature_cols,
    _load_v2,
    _merge_oof_probs,
)
from scripts.train_lgbm_baseline import TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES

MODEL_NAME = "chain_server_extra"


def _align_to_model(df: pd.DataFrame, model_feats: list[str]) -> pd.DataFrame:
    """Align dataframe columns to model's training features, filling missing with 0."""
    out = df.copy()
    for col in model_feats:
        if col not in out.columns:
            out[col] = 0.0
    return out[model_feats].fillna(0.0)


def _add_chain_probs_to_extra(
    extra: pd.DataFrame,
    action_model,
    action_feats: list[str],
    point_model,
    point_feats: list[str],
) -> pd.DataFrame:
    """Compute chain_action and chain_point probs for extra rows using full-data models."""
    pa_extra = align_proba(action_model, _align_to_model(extra, action_feats), TARGET_ACTION_CLASSES)

    extra_with_act = extra.copy()
    for i in range(N_ACTION):
        extra_with_act[f"act_p_{i}"] = pa_extra[:, i]

    pp_extra = align_proba(point_model, _align_to_model(extra_with_act, point_feats), TARGET_POINT_CLASSES)

    extra_with_both = extra_with_act.copy()
    for i in range(N_POINT):
        extra_with_both[f"pnt_p_{i}"] = pp_extra[:, i]

    return extra_with_both


def main() -> None:
    train_raw = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    official_matches = set(train_raw["match"].astype(str).unique())
    extra_pairs = load_extra_pairs(official_matches)

    # Load v2 + chain action/point OOF probs (already shipped in production)
    v2 = _load_v2()
    v2_act = _merge_oof_probs(v2, "chain_action", "action", "act", N_ACTION)
    v2_act_pnt = _merge_oof_probs(v2_act, "chain_point", "point", "pnt", N_POINT)

    # Train full-data chain_action and chain_point models to get probs for extra_pairs
    print("[chain_server_extra] Training full-data action model for extra rows...", flush=True)
    base_feats = _feature_cols(v2)
    action_full = fit_multiclass_full(
        v2[base_feats], v2["y_actionId"], TARGET_ACTION_CLASSES, "sqrt", 2026, 240, 31
    )

    v2_for_point = v2.copy()
    pa_v2 = align_proba(action_full, v2[base_feats].fillna(0.0), TARGET_ACTION_CLASSES)
    for i in range(N_ACTION):
        v2_for_point[f"act_p_{i}"] = pa_v2[:, i]

    print("[chain_server_extra] Training full-data point model for extra rows...", flush=True)
    point_feats = _feature_cols(v2_for_point)
    point_full = fit_multiclass_full(
        v2_for_point[point_feats], v2["y_pointId"], TARGET_POINT_CLASSES, "sqrt", 3026, 240, 31
    )

    # Add chain probs to extra_pairs using full-data models
    print("[chain_server_extra] Computing chain probs for extra_pairs...", flush=True)
    extra_with_chain = _add_chain_probs_to_extra(extra_pairs, action_full, base_feats, point_full, point_feats)

    # Ensure extra_pairs has the required chain prob columns with same dtypes
    server_feats_ref = _feature_cols(v2_act_pnt)

    # OOF loop — server prediction only
    by_fold: list[tuple[pd.DataFrame, np.ndarray]] = []
    for (seed, fold), valid in v2_act_pnt.groupby(["seed", "fold"], sort=False):
        train_off = v2_act_pnt[(v2_act_pnt["seed"] == seed) & (v2_act_pnt["fold"] != fold)]

        # Align extra columns to training frame
        extra_aligned = extra_with_chain.copy()
        for col in server_feats_ref:
            if col not in extra_aligned.columns:
                extra_aligned[col] = 0.0
        extra_aligned = extra_aligned[[c for c in extra_aligned.columns if c in train_off.columns or c in ("y_serverGetPoint",)]]

        # Concat official train folds + extra pairs
        combined_train = pd.concat([train_off, extra_aligned], ignore_index=True)

        feats = [c for c in server_feats_ref if c in valid.columns]
        from scripts.train_lgbm_baseline import fit_binary
        p = fit_binary(
            combined_train[feats].fillna(0.0),
            combined_train["y_serverGetPoint"],
            valid[feats].fillna(0.0),
            valid["y_serverGetPoint"],
            4026 + int(fold),
            240,
            31,
        )
        by_fold.append((valid[["rally_uid", "seed", "fold", "target_strikeNumber"]], p.reshape(-1, 1)))
        print(f"chain_server_extra seed={seed} fold={fold} n_train={len(combined_train)} n_valid={len(valid)}", flush=True)

    # Write OOF
    metas = [x[0] for x in by_fold]
    probs = [x[1] for x in by_fold]
    meta = pd.concat(metas, ignore_index=True)
    p_all = np.concatenate(probs, axis=0)
    out = write_oof(MODEL_NAME, "server", meta["rally_uid"].to_numpy(), meta["seed"].to_numpy(),
                    meta["fold"].to_numpy(), meta["target_strikeNumber"].to_numpy(), p_all)
    print(f"wrote {out}: rows={len(meta)}", flush=True)

    # Build test predictions using full-data server model
    print("[chain_server_extra] Building test predictions...", flush=True)
    # Align extra for full training
    extra_for_full = extra_with_chain.copy()
    for col in server_feats_ref:
        if col not in extra_for_full.columns:
            extra_for_full[col] = 0.0

    full_train = pd.concat(
        [v2_act_pnt, extra_for_full[[c for c in extra_for_full.columns if c in v2_act_pnt.columns]]],
        ignore_index=True
    )
    feats_full = [c for c in server_feats_ref if c in full_train.columns]
    from scripts.make_lgbm_submission import fit_binary_full, build_test_dataset
    server_full = fit_binary_full(full_train[feats_full].fillna(0.0), full_train["y_serverGetPoint"], 4026, 240, 31)

    # Build test chain action/point probs then server test
    from scripts.build_route_b_submission import _build_test_features, _align_test_columns, _encoder_train_frame, _encoder_apply_frame
    from scripts.feature_ngrams import ngram_features
    from scripts.feature_semisupervised import build_player_feature_dist
    from scripts.target_encoding import build_player_encoders
    from scripts.train_lgbm_baseline import add_prefix_features

    test_raw = pd.read_csv(next(Path.cwd().glob("AI CUP*/test_new.csv")))
    semi = build_player_feature_dist(train_raw, test_raw).set_index("gamePlayerId")
    encoders = build_player_encoders(_encoder_train_frame(v2), n_action=N_ACTION, n_point=N_POINT)
    test = _build_test_features(test_raw, semi, encoders).sort_values("rally_uid").reset_index(drop=True)
    rally_uid = test["rally_uid"].to_numpy()

    x_test_base = _align_test_columns(base_feats, test)

    pa_test = align_proba(action_full, x_test_base.fillna(0.0), TARGET_ACTION_CLASSES)
    test_with_act = test.copy()
    for i in range(N_ACTION):
        test_with_act[f"act_p_{i}"] = pa_test[:, i]

    pp_test = align_proba(point_full, _align_test_columns(point_feats, test_with_act).fillna(0.0), TARGET_POINT_CLASSES)
    test_with_both = test_with_act.copy()
    for i in range(N_POINT):
        test_with_both[f"pnt_p_{i}"] = pp_test[:, i]

    x_test_server = _align_test_columns(feats_full, test_with_both).fillna(0.0)
    p_server_test = server_full.predict_proba(x_test_server)[:, list(server_full.classes_).index(1)].reshape(-1, 1)
    _write_test_parquet(MODEL_NAME, "server", rally_uid, p_server_test)
    print(f"[chain_server_extra] Done. Test rows: {len(rally_uid)}", flush=True)


if __name__ == "__main__":
    main()
