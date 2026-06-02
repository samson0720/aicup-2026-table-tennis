"""Phase-specific LGBM with new spatial/player-turn features + another_data.

New features added on top of standard add_prefix_features():
  - pointId spatial geometry: zone_x (lateral 0-2), zone_y (depth 0-2)
  - Zone deltas: dx, dy between last two shots
  - Player-turn separated: my_last{1,2,3} vs opp_last{1,2,3} actionId/pointId
  - 2-shot action transitions: last2→last1

Model: phase-specific LightGBM, 31 leaves, another_data augmentation.

Writes:
  artifacts/oof/phase_lgbm_newfeats_extra_{action,point,server}.parquet (+test)

Usage:
  conda run -n aicup-tt python -m scripts.produce_phase_lgbm_newfeats_extra_oof
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import (
    align_proba, build_test_dataset, fit_binary_full, fit_multiclass_full,
)
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_extra_lgbm_oof import load_extra_pairs
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES,
    fit_binary, fit_multiclass,
    feature_columns,
)

MODEL_NAME = "phase_lgbm_newfeats_extra"
NUM_LEAVES = 31
N_ESTIMATORS = 180
MIN_PHASE_ROWS = 100


def _pzone(pid: int) -> tuple[int, int]:
    """pointId 1-9 → (zone_x 0-2, zone_y 0-2). 0 stays (0,0)."""
    if pid <= 0 or pid > 9:
        return (-1, -1)
    z = pid - 1
    return (z % 3, z // 3)


def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Augment a DataFrame of prefix features with new spatial/turn features."""
    out = df.copy()

    # --- pointId spatial geometry for last 1-3 shots ---
    for back in range(1, 4):
        col = f"last{back}_pointId"
        if col in out.columns:
            zx = out[col].apply(lambda v: _pzone(int(v))[0])
            zy = out[col].apply(lambda v: _pzone(int(v))[1])
            out[f"last{back}_zone_x"] = zx
            out[f"last{back}_zone_y"] = zy

    # --- Zone deltas (directional movement) ---
    for back in range(1, 3):
        x1 = f"last{back}_zone_x"
        x2 = f"last{back+1}_zone_x"
        y1 = f"last{back}_zone_y"
        y2 = f"last{back+1}_zone_y"
        if all(c in out.columns for c in [x1, x2, y1, y2]):
            out[f"dzone_x_{back}"] = out[x1] - out[x2]
            out[f"dzone_y_{back}"] = out[y1] - out[y2]

    # --- Player-turn separated features ---
    # In a rally the player at target_strike hits at even/odd positions.
    # last1 = opponent, last2 = me, last3 = opponent, last4 = me ...
    for my_back, src_back in [(1, 2), (2, 4), (3, 6)]:
        for col_suffix in ("actionId", "pointId", "spinId"):
            src_col = f"last{src_back}_{col_suffix}"
            if src_col in out.columns:
                out[f"my_last{my_back}_{col_suffix}"] = out[src_col]
        for col_suffix in ("actionId", "pointId", "spinId"):
            opp_src = f"last{src_back-1}_{col_suffix}"
            if opp_src in out.columns:
                out[f"opp_last{my_back}_{col_suffix}"] = out[opp_src]

    # --- 2-shot action transition ---
    for a_cols in [("last2_actionId", "last1_actionId")]:
        c1, c2 = a_cols
        if c1 in out.columns and c2 in out.columns:
            out["trans_action_2_1"] = out[c1] * 100 + out[c2]

    # --- My vs opponent action entropy ---
    if "my_last1_actionId" in out.columns and "my_last2_actionId" in out.columns:
        # Simple diversity: did my last two shots use the same action?
        out["my_action_repeat"] = (out["my_last1_actionId"] == out["my_last2_actionId"]).astype(int)
    if "opp_last1_actionId" in out.columns and "opp_last2_actionId" in out.columns:
        out["opp_action_repeat"] = (out["opp_last1_actionId"] == out["opp_last2_actionId"]).astype(int)

    return out.replace([np.inf, -np.inf], -1).fillna(-1)


def main() -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    official_matches = set(train["match"].astype(str).unique())
    extra_pairs = load_extra_pairs(official_matches)

    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}

    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty:
            continue

        # Add another_data + new features
        df_train = pd.concat([df_train, extra_pairs], ignore_index=True)
        df_train = add_new_features(df_train)
        df_valid = add_new_features(df_valid)

        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        x_tr, x_va = df_train[feats], df_valid[feats]

        # Phase-specific training
        train_phase = df_train["phase"].to_numpy()
        valid_phase = df_valid["phase"].to_numpy()

        p_action = np.zeros((len(df_valid), len(TARGET_ACTION_CLASSES)))
        p_point = np.zeros((len(df_valid), len(TARGET_POINT_CLASSES)))
        p_server = np.zeros(len(df_valid))

        for phase in sorted(set(int(p) for p in valid_phase)):
            val_mask = valid_phase == phase
            trn_mask = train_phase == phase
            if int(trn_mask.sum()) < MIN_PHASE_ROWS:
                trn_mask = np.ones(len(train_phase), dtype=bool)
            xtr_p = x_tr.iloc[trn_mask] if hasattr(x_tr, "iloc") else x_tr[trn_mask]
            xva_p = x_va.iloc[val_mask] if hasattr(x_va, "iloc") else x_va[val_mask]
            ytr = df_train.iloc[trn_mask] if hasattr(df_train, "iloc") else df_train[trn_mask]

            p_action[val_mask] = fit_multiclass(
                xtr_p, ytr["y_actionId"], xva_p, df_valid.loc[val_mask, "y_actionId"],
                TARGET_ACTION_CLASSES, "sqrt", 2026 + int(phase) * 10, N_ESTIMATORS, NUM_LEAVES,
            )
            p_point[val_mask] = fit_multiclass(
                xtr_p, ytr["y_pointId"], xva_p, df_valid.loc[val_mask, "y_pointId"],
                TARGET_POINT_CLASSES, "sqrt", 3026 + int(phase) * 10, N_ESTIMATORS, NUM_LEAVES,
            )
            p_server[val_mask] = fit_binary(
                xtr_p, ytr["y_serverGetPoint"], xva_p, df_valid.loc[val_mask, "y_serverGetPoint"],
                4026 + int(phase) * 10, N_ESTIMATORS, NUM_LEAVES,
            )

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()

        for tgt, p in [("action", p_action), ("point", p_point), ("server", p_server.reshape(-1, 1))]:
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)

        print(f"{MODEL_NAME} seed={seed} fold={fold} n_train={len(df_train)} n_valid={len(df_valid)}", flush=True)

    for tgt in ("action", "point", "server"):
        r = np.concatenate(bag[tgt]["r"]); s = np.concatenate(bag[tgt]["s"])
        f = np.concatenate(bag[tgt]["f"]); c = np.concatenate(bag[tgt]["c"])
        p = np.concatenate(bag[tgt]["p"], axis=0)
        out = write_oof(MODEL_NAME, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)

    # Build test predictions
    print(f"\n[{MODEL_NAME}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_splits = splits
    full_df_train = build_one_sample_per_rally(train, full_splits)
    full_df_train = pd.concat([full_df_train, extra_pairs], ignore_index=True)
    full_df_train = add_new_features(full_df_train)

    test_raw = build_test_dataset(dd)
    test_raw = add_new_features(test_raw)

    rally_test = test_raw["rally_uid"].to_numpy()
    feats_full = [c for c in feature_columns(full_df_train) if c in test_raw.columns]
    x_full = full_df_train[feats_full].fillna(-1)
    x_test = test_raw[feats_full].fillna(-1)

    train_phase = full_df_train["phase"].to_numpy()
    test_phase = test_raw["phase"].to_numpy()

    p_action = np.zeros((len(test_raw), len(TARGET_ACTION_CLASSES)))
    p_point = np.zeros((len(test_raw), len(TARGET_POINT_CLASSES)))
    p_server = np.zeros(len(test_raw))

    for phase in sorted(set(int(p) for p in test_phase)):
        tst_mask = test_phase == phase
        trn_mask = train_phase == phase
        if int(trn_mask.sum()) < MIN_PHASE_ROWS:
            trn_mask = np.ones(len(train_phase), dtype=bool)
        xtr_p = x_full.iloc[trn_mask]
        xte_p = x_test.iloc[tst_mask]
        ytr = full_df_train.iloc[trn_mask]

        am = fit_multiclass_full(xtr_p, ytr["y_actionId"], TARGET_ACTION_CLASSES, "sqrt", 8200 + phase, N_ESTIMATORS, NUM_LEAVES)
        pm = fit_multiclass_full(xtr_p, ytr["y_pointId"], TARGET_POINT_CLASSES, "sqrt", 8300 + phase, N_ESTIMATORS, NUM_LEAVES)
        sm = fit_binary_full(xtr_p, ytr["y_serverGetPoint"], 8400 + phase, N_ESTIMATORS, NUM_LEAVES)

        p_action[tst_mask] = align_proba(am, xte_p, TARGET_ACTION_CLASSES)
        p_point[tst_mask] = align_proba(pm, xte_p, TARGET_POINT_CLASSES)
        p_server[tst_mask] = sm.predict_proba(xte_p)[:, list(sm.classes_).index(1)]

    _write_test_parquet(MODEL_NAME, "action", rally_test, p_action)
    _write_test_parquet(MODEL_NAME, "point", rally_test, p_point)
    _write_test_parquet(MODEL_NAME, "server", rally_test, p_server.reshape(-1, 1))
    print(f"[{MODEL_NAME}] Done.", flush=True)


if __name__ == "__main__":
    main()
