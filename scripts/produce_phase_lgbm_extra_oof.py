"""Phase-specific LGBM + another_data augmentation OOF + test parquets.

Combines the phase-splitting approach of train_phase_lgbm with the
another_data augmentation from produce_extra_lgbm_oof. Writes:
  artifacts/oof/phase_lgbm_extra_{action,point,server}.parquet
  artifacts/oof/phase_lgbm_extra_{action,point,server}_test.parquet

Usage:
  conda run -n aicup-tt python -m scripts.produce_phase_lgbm_extra_oof
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import (
    align_proba,
    build_test_dataset,
    fit_binary_full,
    fit_multiclass_full,
)
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    build_prefix_dataset,
    feature_columns,
    fit_binary,
    fit_multiclass,
)

EXTRA_DATA_PATH = Path("another_data/train.csv")
MODEL_NAME = "phase_lgbm_extra"
N_ESTIMATORS = 180
NUM_LEAVES = 31
MIN_PHASE_ROWS = 100


def load_extra_pairs(official_matches: set[str]) -> pd.DataFrame:
    df = pd.read_csv(EXTRA_DATA_PATH)
    df = df.rename(columns={"strickNumber": "strikeNumber", "strickId": "strikeId"})
    df = df[df["let"] == 0].copy()
    df = df[~df["match"].astype(str).isin(official_matches)].copy()
    df = df.drop(columns=[c for c in ("serveId", "serveNumber", "let") if c in df.columns])
    sgp = df.groupby("rally_uid")["serverGetPoint"].first()
    df["serverGetPoint"] = df["rally_uid"].map(sgp)
    pairs = build_prefix_dataset(df)
    print(f"[extra] {len(pairs)} training pairs from {df['match'].nunique()} new matches", flush=True)
    return pairs


def _phase_predict(df_train: pd.DataFrame, df_valid: pd.DataFrame, feats: list[str]) -> tuple:
    """Train per-phase models on df_train, predict on df_valid. Returns (p_action, p_point, p_server)."""
    x_train = df_train[feats].reset_index(drop=True)
    x_valid = df_valid[feats].reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    p_action = np.zeros((len(df_valid), len(TARGET_ACTION_CLASSES)), dtype=np.float64)
    p_point = np.zeros((len(df_valid), len(TARGET_POINT_CLASSES)), dtype=np.float64)
    p_server = np.zeros(len(df_valid), dtype=np.float64)

    train_phase = df_train["phase"].to_numpy()
    valid_phase = df_valid["phase"].to_numpy()
    for phase in sorted(set(int(p) for p in valid_phase)):
        val_mask = valid_phase == phase
        if not val_mask.any():
            continue
        trn_mask = train_phase == phase
        if int(trn_mask.sum()) < MIN_PHASE_ROWS:
            trn_mask = np.ones_like(train_phase, dtype=bool)

        xt = x_train.iloc[trn_mask]
        xv = x_valid.iloc[val_mask]
        ya_t = df_train.loc[trn_mask, "y_actionId"]
        yp_t = df_train.loc[trn_mask, "y_pointId"]
        ys_t = df_train.loc[trn_mask, "y_serverGetPoint"]

        p_action[val_mask] = fit_multiclass(
            xt, ya_t, xv, df_valid.loc[val_mask, "y_actionId"],
            TARGET_ACTION_CLASSES, "sqrt", 8200 + int(phase) * 10, N_ESTIMATORS, NUM_LEAVES,
        )
        p_point[val_mask] = fit_multiclass(
            xt, yp_t, xv, df_valid.loc[val_mask, "y_pointId"],
            TARGET_POINT_CLASSES, "sqrt", 8300 + int(phase) * 10, N_ESTIMATORS, NUM_LEAVES,
        )
        p_server[val_mask] = fit_binary(
            xt, ys_t, xv, df_valid.loc[val_mask, "y_serverGetPoint"],
            8400 + int(phase) * 10, N_ESTIMATORS, NUM_LEAVES,
        )
    return p_action, p_point, p_server


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

        df_train = pd.concat([df_train, extra_pairs], ignore_index=True)
        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]

        pa, pp, ps = _phase_predict(df_train, df_valid, feats)

        rally = df_valid["rally_uid"].to_numpy()
        sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold)
        cut = df_valid["target_strikeNumber"].to_numpy()
        for tgt, p in [("action", pa), ("point", pp), ("server", ps.reshape(-1, 1))]:
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"{MODEL_NAME} seed={seed} fold={fold} n_train={len(df_train)} n_valid={len(rally)}", flush=True)

    for tgt in ("action", "point", "server"):
        r = np.concatenate(bag[tgt]["r"]); s = np.concatenate(bag[tgt]["s"])
        f = np.concatenate(bag[tgt]["f"]); c = np.concatenate(bag[tgt]["c"])
        p = np.concatenate(bag[tgt]["p"], axis=0)
        out = write_oof(MODEL_NAME, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)

    # --- Test predictions (full train + extra, phase-specific) ---
    print(f"\n[{MODEL_NAME}] Building test predictions...", flush=True)
    dd = next(Path.cwd().glob("AI CUP*"))
    full_df_train = build_one_sample_per_rally(train, splits)
    full_df_train = pd.concat([full_df_train, extra_pairs], ignore_index=True)
    full_df_train = full_df_train.reset_index(drop=True)

    test_features = build_test_dataset(pd.read_csv(dd / "test_new.csv")).sort_values("rally_uid").reset_index(drop=True)
    feats = [c for c in feature_columns(full_df_train) if c in test_features.columns]
    x_train_full = full_df_train[feats].reset_index(drop=True)
    x_test = test_features[feats].reset_index(drop=True)

    n_test = len(test_features)
    p_action = np.zeros((n_test, len(TARGET_ACTION_CLASSES)), dtype=np.float64)
    p_point = np.zeros((n_test, len(TARGET_POINT_CLASSES)), dtype=np.float64)
    p_server = np.zeros(n_test, dtype=np.float64)

    train_phase = full_df_train["phase"].to_numpy()
    test_phase = test_features["phase"].to_numpy()
    for phase in sorted(set(int(p) for p in test_phase)):
        test_mask = test_phase == phase
        if not test_mask.any():
            continue
        train_mask = train_phase == phase
        if int(train_mask.sum()) < MIN_PHASE_ROWS:
            train_mask = np.ones_like(train_phase, dtype=bool)

        xt = x_train_full.iloc[train_mask]
        ya = full_df_train.loc[train_mask, "y_actionId"]
        yp = full_df_train.loc[train_mask, "y_pointId"]
        ys = full_df_train.loc[train_mask, "y_serverGetPoint"]
        xv = x_test.iloc[test_mask]

        am = fit_multiclass_full(xt, ya, TARGET_ACTION_CLASSES, "sqrt",
                                 seed=8200 + int(phase), n_estimators=N_ESTIMATORS, num_leaves=NUM_LEAVES)
        pm = fit_multiclass_full(xt, yp, TARGET_POINT_CLASSES, "sqrt",
                                 seed=8300 + int(phase), n_estimators=N_ESTIMATORS, num_leaves=NUM_LEAVES)
        sm = fit_binary_full(xt, ys, seed=8400 + int(phase), n_estimators=N_ESTIMATORS, num_leaves=NUM_LEAVES)

        p_action[test_mask] = align_proba(am, xv, TARGET_ACTION_CLASSES)
        p_point[test_mask] = align_proba(pm, xv, TARGET_POINT_CLASSES)
        p_server[test_mask] = sm.predict_proba(xv)[:, list(sm.classes_).index(1)]
        print(f"phase={phase}: n_train={int(train_mask.sum())}, n_test={int(test_mask.sum())}", flush=True)

    rally_test = test_features["rally_uid"].to_numpy()
    _write_test_parquet(MODEL_NAME, "action", rally_test, p_action)
    _write_test_parquet(MODEL_NAME, "point", rally_test, p_point)
    _write_test_parquet(MODEL_NAME, "server", rally_test, p_server.reshape(-1, 1))
    print(f"[{MODEL_NAME}] test predictions written.", flush=True)


if __name__ == "__main__":
    main()
