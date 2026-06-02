"""Pilot: CatBoost +extra training data (GPU).

Tests whether adding another_data new-match rows improves CatBoost CV.
Runs 1 seed (fast sanity) by default; set SEEDS_TO_RUN for full 5-seed.

Usage:
  conda run -n aicup-tt python -m scripts.pilot_extra_train_cat
"""
from __future__ import annotations

import json
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
)
from scripts.train_catboost_baseline import (
    cat_feature_indices,
    fit_binary,
    fit_multiclass,
    prepare_x,
)

EXTRA_DATA_PATH = Path("another_data/train.csv")
SPLITS_PATH = Path("artifacts/cv_splits.parquet")

# 1 seed for quick signal; expand to [11,22,33,44,55] if positive
SEEDS_TO_RUN = [11]
CAT_ITERATIONS = 500   # production uses 1000; 500 is fast enough for pilot
CAT_DEPTH = 6
TASK_TYPE = "GPU"
DEVICES = "0"


def load_extra_data(official_matches: set[str]) -> pd.DataFrame:
    df = pd.read_csv(EXTRA_DATA_PATH)
    df = df.rename(columns={"strickNumber": "strikeNumber", "strickId": "strikeId"})
    df = df[df["let"] == 0].copy()
    df = df[~df["match"].astype(str).isin(official_matches)].copy()
    df = df.drop(columns=[c for c in ("serveId", "serveNumber", "let") if c in df.columns])
    sgp = df.groupby("rally_uid")["serverGetPoint"].first()
    df["serverGetPoint"] = df["rally_uid"].map(sgp)
    print(f"[extra] {len(df)} rows, {df['rally_uid'].nunique()} rallies, "
          f"{df['match'].nunique()} new matches")
    return df.reset_index(drop=True)


def run_cv(official_train, splits, extra_rows, label, seed):
    results = []
    for s, fold, train_view, valid_view in iter_cv_folds(official_train, splits):
        if s != seed:
            continue
        s_train = splits[(splits["seed"] == s) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == s) & (splits["fold"] == fold)]

        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty:
            continue

        if extra_rows is not None:
            df_train = pd.concat([df_train, extra_rows], ignore_index=True)

        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        cat_idx = cat_feature_indices(feats)
        cat_cols = [feats[i] for i in cat_idx]
        x_train = prepare_x(df_train[feats], cat_cols)
        x_valid = prepare_x(df_valid[feats], cat_cols)

        pa = fit_multiclass(
            x_train, df_train["y_actionId"], x_valid,
            TARGET_ACTION_CLASSES, cat_idx, "sqrt",
            9000 + fold, CAT_ITERATIONS, CAT_DEPTH, TASK_TYPE, DEVICES,
        )
        pp = fit_multiclass(
            x_train, df_train["y_pointId"], x_valid,
            TARGET_POINT_CLASSES, cat_idx, "sqrt",
            9100 + fold, CAT_ITERATIONS, CAT_DEPTH, TASK_TYPE, DEVICES,
        )
        ps = fit_binary(
            x_train, df_train["y_serverGetPoint"], x_valid,
            cat_idx, 9200 + fold, CAT_ITERATIONS, CAT_DEPTH, TASK_TYPE, DEVICES,
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
        print(f"  [{label}] seed={s} fold={fold}: "
              f"overall={overall:.5f}  action={action_f1:.5f}  "
              f"point={point_f1:.5f}  server={server_auc:.5f}  "
              f"(n_train={len(df_train)})", flush=True)
    return results


def main():
    splits = pd.read_parquet(SPLITS_PATH)
    official_train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    official_matches = set(official_train["match"].astype(str).unique())

    extra_raw = load_extra_data(official_matches)
    extra_rows = build_prefix_dataset(extra_raw)
    print(f"[extra] built {len(extra_rows)} training pairs", flush=True)

    all_results = []
    for seed in SEEDS_TO_RUN:
        print(f"\n=== seed={seed}: CatBoost baseline ===", flush=True)
        all_results += run_cv(official_train, splits, None, "cat_baseline", seed)
        print(f"\n=== seed={seed}: CatBoost +extra ===", flush=True)
        all_results += run_cv(official_train, splits, extra_rows, "cat+extra", seed)

    df = pd.DataFrame(all_results)
    metric_cols = ["action_f1", "point_f1", "server_auc", "overall"]

    print("\n=== SUMMARY ===")
    summary = df.groupby("label")[metric_cols].mean().round(5)
    print(summary.to_string())

    baseline = df[df["label"] == "cat_baseline"]["overall"].mean()
    extra = df[df["label"] == "cat+extra"]["overall"].mean()
    delta = extra - baseline
    print(f"\nΔoverall = {delta:+.5f}  "
          f"({'POSITIVE' if delta > 0 else 'NEGATIVE'})")

    out = {
        "results": all_results,
        "summary": summary.to_dict(),
        "delta_overall": round(delta, 5),
        "extra_rows_added": len(extra_rows),
        "cat_iterations": CAT_ITERATIONS,
        "task_type": TASK_TYPE,
    }
    Path("artifacts/extra_train_cat_pilot.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("Saved → artifacts/extra_train_cat_pilot.json")


if __name__ == "__main__":
    main()
