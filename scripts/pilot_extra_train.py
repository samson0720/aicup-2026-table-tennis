"""Pilot: does adding extra training data from another_data/train.csv help?

Strategy:
  - Load another_data/train.csv (course project dataset, 57 new match IDs)
  - Filter to only new matches not in official train
  - Filter let=0 rows only (skip hidden strokes with -1 labels)
  - Build (prefix→next_stroke) training pairs from new matches
  - CV test: old train only  vs  old train + extra pairs
  - Validation always on official cv_splits.parquet (same folds as baseline)

Usage:
  conda run -n aicup-tt python -m scripts.pilot_extra_train
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    build_prefix_dataset,
    feature_columns,
    fit_binary,
    fit_multiclass,
)

EXTRA_DATA_PATH = Path("another_data/train.csv")
SPLITS_PATH = Path("artifacts/cv_splits.parquet")
OFFICIAL_TRAIN_GLOB = "AI CUP*/train.csv"

SEEDS_TO_RUN = [11, 22, 33, 44, 55]


def load_extra_data(official_matches: set[str]) -> pd.DataFrame:
    """Load and normalise the course-project data; keep only new matches."""
    df = pd.read_csv(EXTRA_DATA_PATH)

    # Normalise typo-style column names to match official schema
    df = df.rename(columns={
        "strickNumber": "strikeNumber",
        "strickId": "strikeId",
    })

    # Keep only fully-labelled strokes from new matches
    df = df[df["let"] == 0].copy()
    df = df[~df["match"].astype(str).isin(official_matches)].copy()

    # Drop extra columns not in official schema
    extra_cols = {"serveId", "serveNumber", "let"}
    df = df.drop(columns=[c for c in extra_cols if c in df.columns])

    # serverGetPoint is per-rally; broadcast the first row's value to all rows
    # (official format: every row in a rally has the same serverGetPoint)
    sgp = df.groupby("rally_uid")["serverGetPoint"].first()
    df["serverGetPoint"] = df["rally_uid"].map(sgp)

    print(f"[extra] {len(df)} rows, {df['rally_uid'].nunique()} rallies, "
          f"{df['match'].nunique()} new matches, "
          f"{df['gamePlayerId'].nunique()} players")
    return df.reset_index(drop=True)


def build_extra_train_rows(extra: pd.DataFrame) -> pd.DataFrame:
    """Build all (prefix→next_stroke) training pairs from extra data."""
    rows = build_prefix_dataset(extra)
    print(f"[extra] built {len(rows)} training pairs from extra data")
    return rows


def run_cv(
    official_train: pd.DataFrame,
    splits: pd.DataFrame,
    extra_rows: pd.DataFrame | None,
    label: str,
    seed: int,
) -> list[dict]:
    results = []
    folds_run = 0
    for s, fold, train_view, valid_view in iter_cv_folds(official_train, splits):
        if s != seed:
            continue
        s_train = splits[(splits["seed"] == s) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == s) & (splits["fold"] == fold)]

        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty:
            continue

        if extra_rows is not None and len(extra_rows) > 0:
            df_train = pd.concat([df_train, extra_rows], ignore_index=True)

        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        x_train = df_train[feats]
        x_valid = df_valid[feats]

        pa = fit_multiclass(
            x_train, df_train["y_actionId"],
            x_valid, df_valid["y_actionId"],
            TARGET_ACTION_CLASSES, "sqrt", 2026 + fold, 180, 15,
        )
        pp = fit_multiclass(
            x_train, df_train["y_pointId"],
            x_valid, df_valid["y_pointId"],
            TARGET_POINT_CLASSES, "sqrt", 3026 + fold, 180, 15,
        )
        ps = fit_binary(
            x_train, df_train["y_serverGetPoint"],
            x_valid, df_valid["y_serverGetPoint"],
            4026 + fold, 180, 15,
        )

        action_f1 = f1_score(df_valid["y_actionId"], pa.argmax(1),
                             labels=TARGET_ACTION_CLASSES, average="macro", zero_division=0)
        point_f1 = f1_score(df_valid["y_pointId"], pp.argmax(1),
                            labels=TARGET_POINT_CLASSES, average="macro", zero_division=0)
        server_auc = roc_auc_score(df_valid["y_serverGetPoint"], ps)
        overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc

        r = {
            "label": label, "seed": s, "fold": fold,
            "n_train": len(df_train), "n_valid": len(df_valid),
            "action_f1": round(action_f1, 5),
            "point_f1": round(point_f1, 5),
            "server_auc": round(server_auc, 5),
            "overall": round(overall, 5),
        }
        results.append(r)
        folds_run += 1
        print(f"  [{label}] seed={s} fold={fold}: "
              f"overall={overall:.5f}  action={action_f1:.5f}  "
              f"point={point_f1:.5f}  server={server_auc:.5f}  "
              f"(n_train={len(df_train)})")
    return results


def main() -> None:
    splits = pd.read_parquet(SPLITS_PATH)
    official_train = pd.read_csv(next(Path.cwd().glob(OFFICIAL_TRAIN_GLOB)))
    official_matches = set(official_train["match"].astype(str).unique())

    extra_raw = load_extra_data(official_matches)
    extra_rows = build_extra_train_rows(extra_raw)

    all_results: list[dict] = []
    for seed in SEEDS_TO_RUN:
        print(f"\n=== seed={seed}: baseline (old train only) ===")
        all_results += run_cv(official_train, splits, None, "baseline", seed)
        print(f"\n=== seed={seed}: +extra ({len(extra_rows)} rows) ===")
        all_results += run_cv(official_train, splits, extra_rows, "+extra", seed)

    df = pd.DataFrame(all_results)
    metric_cols = ["action_f1", "point_f1", "server_auc", "overall"]

    print("\n=== SUMMARY ===")
    summary = df.groupby("label")[metric_cols].mean().round(5)
    print(summary.to_string())

    baseline_overall = df[df["label"] == "baseline"]["overall"].mean()
    extra_overall = df[df["label"] == "+extra"]["overall"].mean()
    delta = extra_overall - baseline_overall
    print(f"\nΔoverall = {delta:+.5f}  "
          f"({'POSITIVE - worth pursuing' if delta > 0 else 'NEGATIVE - not useful'})")

    out = {
        "results": all_results,
        "summary": summary.to_dict(),
        "delta_overall": round(delta, 5),
        "extra_rows_added": len(extra_rows),
        "new_matches": int(extra_raw["match"].nunique()),
    }
    Path("artifacts/extra_train_pilot.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("\nSaved → artifacts/extra_train_pilot.json")


if __name__ == "__main__":
    main()
