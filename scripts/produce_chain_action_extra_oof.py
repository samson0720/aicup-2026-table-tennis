"""Chain-action model augmented with another_data (chain_action_extra).

Identical to train_chain_lgbm.train_action() but appends another_data rows
to each fold's training set. No chain-feature dependency (action is first in chain).

Writes:
  artifacts/oof/chain_action_extra_action.parquet
  artifacts/oof/chain_action_extra_action_test.parquet

Usage:
  conda run -n aicup-tt python -m scripts.produce_chain_action_extra_oof
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_lgbm_submission import align_proba, fit_multiclass_full
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
from scripts.train_chain_lgbm import EXCLUDE, N_ACTION, _feature_cols, _load_v2
from scripts.train_lgbm_baseline import TARGET_ACTION_CLASSES, fit_multiclass

MODEL_NAME = "chain_action_extra"


def _align_extra(extra: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = extra.copy()
    for col in feats:
        if col not in out.columns:
            out[col] = 0.0
    return out[feats].fillna(0.0)


def main() -> None:
    train_raw = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    official_matches = set(train_raw["match"].astype(str).unique())
    extra_pairs = load_extra_pairs(official_matches)

    v2 = _load_v2()
    feats = _feature_cols(v2)

    print(f"[{MODEL_NAME}] v2 rows={len(v2)}, extra rows={len(extra_pairs)}", flush=True)

    # Ensure extra_pairs has all required feature columns
    extra_aligned_df = extra_pairs.copy()
    for col in feats:
        if col not in extra_aligned_df.columns:
            extra_aligned_df[col] = 0.0

    by_fold: list[tuple[pd.DataFrame, np.ndarray]] = []
    for (seed, fold), valid in v2.groupby(["seed", "fold"], sort=False):
        train_off = v2[(v2["seed"] == seed) & (v2["fold"] != fold)]
        combined = pd.concat([train_off, extra_aligned_df], ignore_index=True)

        x_tr = combined[feats].fillna(0.0)
        x_va = valid[feats].fillna(0.0)

        p = fit_multiclass(
            x_tr, combined["y_actionId"],
            x_va, valid["y_actionId"],
            TARGET_ACTION_CLASSES, "sqrt",
            2026 + int(fold), 240, 31,
        )
        by_fold.append((valid[["rally_uid", "seed", "fold", "target_strikeNumber"]], p))
        print(f"{MODEL_NAME} seed={seed} fold={fold} n_train={len(combined)} n_valid={len(valid)}", flush=True)

    metas = [x[0] for x in by_fold]
    probs = [x[1] for x in by_fold]
    meta = pd.concat(metas, ignore_index=True)
    p_all = np.concatenate(probs, axis=0)
    out = write_oof(MODEL_NAME, "action", meta["rally_uid"].to_numpy(), meta["seed"].to_numpy(),
                    meta["fold"].to_numpy(), meta["target_strikeNumber"].to_numpy(), p_all)
    print(f"wrote {out}: rows={len(meta)}", flush=True)

    # Test predictions using full-data model
    print(f"[{MODEL_NAME}] Building test predictions...", flush=True)
    full_train = pd.concat([v2, extra_aligned_df], ignore_index=True)
    action_full = fit_multiclass_full(
        full_train[feats].fillna(0.0), full_train["y_actionId"],
        TARGET_ACTION_CLASSES, "sqrt", 2026, 240, 31
    )

    from scripts.build_route_b_submission import (
        _align_test_columns, _build_test_features, _encoder_train_frame,
    )
    from scripts.feature_semisupervised import build_player_feature_dist
    from scripts.target_encoding import build_player_encoders

    test_raw = pd.read_csv(next(Path.cwd().glob("AI CUP*/test_new.csv")))
    semi = build_player_feature_dist(train_raw, test_raw).set_index("gamePlayerId")
    encoders = build_player_encoders(_encoder_train_frame(v2), n_action=N_ACTION, n_point=10)
    test = _build_test_features(test_raw, semi, encoders).sort_values("rally_uid").reset_index(drop=True)
    rally_uid = test["rally_uid"].to_numpy()

    x_test = _align_test_columns(feats, test).fillna(0.0)
    pa_test = align_proba(action_full, x_test, TARGET_ACTION_CLASSES)
    _write_test_parquet(MODEL_NAME, "action", rally_uid, pa_test)
    print(f"[{MODEL_NAME}] Done. Test rows: {len(rally_uid)}", flush=True)


if __name__ == "__main__":
    main()
