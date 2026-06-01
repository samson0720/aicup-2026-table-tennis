"""Chain-point model augmented with another_data (chain_point_extra).

Uses chain_action_extra OOF probs (fold-out) for official rows.
For another_data rows, uses chain_action_extra full-data model predictions.

Requires chain_action_extra OOF to be already computed.

Writes:
  artifacts/oof/chain_point_extra_point.parquet
  artifacts/oof/chain_point_extra_point_test.parquet

Usage:
  conda run -n aicup-tt python -m scripts.produce_chain_point_extra_oof
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_lgbm_submission import align_proba, fit_multiclass_full
from scripts.oof_loader import OOF_DIR, write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
from scripts.train_chain_lgbm import EXCLUDE, N_ACTION, N_POINT, _feature_cols, _load_v2, _merge_oof_probs
from scripts.train_lgbm_baseline import TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES, fit_multiclass

MODEL_NAME = "chain_point_extra"


def _align_extra(extra: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = extra.copy()
    for col in feats:
        if col not in out.columns:
            out[col] = 0.0
    return out[feats].fillna(0.0)


def main() -> None:
    # Verify chain_action_extra OOF exists
    ca_oof_path = OOF_DIR / "chain_action_extra_action.parquet"
    if not ca_oof_path.exists():
        raise FileNotFoundError(f"chain_action_extra OOF not found: {ca_oof_path}")

    train_raw = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    official_matches = set(train_raw["match"].astype(str).unique())
    extra_pairs = load_extra_pairs(official_matches)

    # Merge chain_action_extra OOF into v2 (as point model features)
    v2 = _load_v2()
    v2_with_act = _merge_oof_probs(v2, "chain_action_extra", "action", "act", N_ACTION)
    feats = _feature_cols(v2_with_act)

    # Build full-data chain_action_extra model to predict extra_pairs action probs
    print(f"[{MODEL_NAME}] Training full-data chain_action_extra model for extra rows...", flush=True)
    base_feats = _feature_cols(v2)
    extra_base_df = extra_pairs.copy()
    for col in base_feats:
        if col not in extra_base_df.columns:
            extra_base_df[col] = 0.0

    full_train_action = pd.concat([v2, extra_base_df], ignore_index=True)
    action_full = fit_multiclass_full(
        full_train_action[base_feats].fillna(0.0), full_train_action["y_actionId"],
        TARGET_ACTION_CLASSES, "sqrt", 2026, 240, 31
    )

    # Predict extra_pairs action probs
    pa_extra = align_proba(action_full, _align_extra(extra_pairs, base_feats), TARGET_ACTION_CLASSES)
    extra_with_act = extra_pairs.copy()
    for i in range(N_ACTION):
        extra_with_act[f"act_p_{i}"] = pa_extra[:, i]

    # Align extra to point feature set
    for col in feats:
        if col not in extra_with_act.columns:
            extra_with_act[col] = 0.0

    print(f"[{MODEL_NAME}] v2 rows={len(v2)}, extra rows={len(extra_pairs)}", flush=True)

    by_fold: list[tuple[pd.DataFrame, np.ndarray]] = []
    for (seed, fold), valid in v2_with_act.groupby(["seed", "fold"], sort=False):
        train_off = v2_with_act[(v2_with_act["seed"] == seed) & (v2_with_act["fold"] != fold)]
        combined = pd.concat([train_off, extra_with_act], ignore_index=True)

        x_tr = combined[feats].fillna(0.0)
        x_va = valid[feats].fillna(0.0)

        p = fit_multiclass(
            x_tr, combined["y_pointId"],
            x_va, valid["y_pointId"],
            TARGET_POINT_CLASSES, "sqrt",
            3026 + int(fold), 240, 31,
        )
        by_fold.append((valid[["rally_uid", "seed", "fold", "target_strikeNumber"]], p))
        print(f"{MODEL_NAME} seed={seed} fold={fold} n_train={len(combined)} n_valid={len(valid)}", flush=True)

    metas = [x[0] for x in by_fold]
    probs = [x[1] for x in by_fold]
    meta = pd.concat(metas, ignore_index=True)
    p_all = np.concatenate(probs, axis=0)
    out = write_oof(MODEL_NAME, "point", meta["rally_uid"].to_numpy(), meta["seed"].to_numpy(),
                    meta["fold"].to_numpy(), meta["target_strikeNumber"].to_numpy(), p_all)
    print(f"wrote {out}: rows={len(meta)}", flush=True)

    # Test predictions
    print(f"[{MODEL_NAME}] Building test predictions...", flush=True)
    full_train_point = pd.concat([v2_with_act, extra_with_act], ignore_index=True)
    point_full = fit_multiclass_full(
        full_train_point[feats].fillna(0.0), full_train_point["y_pointId"],
        TARGET_POINT_CLASSES, "sqrt", 3026, 240, 31
    )

    from scripts.build_route_b_submission import (
        _align_test_columns, _build_test_features, _encoder_train_frame,
    )
    from scripts.feature_semisupervised import build_player_feature_dist
    from scripts.target_encoding import build_player_encoders

    test_raw = pd.read_csv(next(Path.cwd().glob("AI CUP*/test_new.csv")))
    semi = build_player_feature_dist(train_raw, test_raw).set_index("gamePlayerId")
    encoders = build_player_encoders(_encoder_train_frame(v2), n_action=N_ACTION, n_point=N_POINT)
    test = _build_test_features(test_raw, semi, encoders).sort_values("rally_uid").reset_index(drop=True)
    rally_uid = test["rally_uid"].to_numpy()

    x_test_base = _align_test_columns(base_feats, test).fillna(0.0)
    pa_test = align_proba(action_full, x_test_base, TARGET_ACTION_CLASSES)
    test_with_act = test.copy()
    for i in range(N_ACTION):
        test_with_act[f"act_p_{i}"] = pa_test[:, i]

    x_test_point = _align_test_columns(feats, test_with_act).fillna(0.0)
    pp_test = align_proba(point_full, x_test_point, TARGET_POINT_CLASSES)
    _write_test_parquet(MODEL_NAME, "point", rally_uid, pp_test)
    print(f"[{MODEL_NAME}] Done. Test rows: {len(rally_uid)}", flush=True)


if __name__ == "__main__":
    main()
