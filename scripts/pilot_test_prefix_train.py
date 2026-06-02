"""Pilot: use test_new.csv visible prefix strokes as training augmentation.

test_new.csv has 5,668 labeled rows (actionId/pointId visible for ALL prefix
strokes). For rallies with >=2 strokes, build (prefix→next) training pairs.
These come from the SAME matches/players as the test targets — unlike the
course-data pilot (different matches), this is same-distribution augmentation.

serverGetPoint is unknown for test → set to 0 (dummy); server target is excluded
from comparison (action+point only).

Usage:
  conda run -n aicup-tt python -m scripts.pilot_test_prefix_train
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
    TARGET_ACTION_CLASSES, TARGET_POINT_CLASSES,
    build_prefix_dataset, feature_columns,
    fit_binary, fit_multiclass,
)

SEEDS_TO_RUN = [11, 22, 33, 44, 55]


def load_test_pairs() -> pd.DataFrame:
    test = pd.read_csv(next(Path.cwd().glob("AI CUP*/test_new.csv")))
    multi = test.groupby("rally_uid").filter(lambda g: len(g) >= 2).copy()
    multi["serverGetPoint"] = 0  # dummy — server target excluded from comparison
    pairs = build_prefix_dataset(multi)
    print(f"[test_pairs] {len(pairs)} pairs from {multi['rally_uid'].nunique()} test rallies")
    return pairs


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
        x_train, x_valid = df_train[feats], df_valid[feats]

        pa = fit_multiclass(x_train, df_train["y_actionId"], x_valid, df_valid["y_actionId"],
                            TARGET_ACTION_CLASSES, "sqrt", 2026 + fold, 180, 15)
        pp = fit_multiclass(x_train, df_train["y_pointId"], x_valid, df_valid["y_pointId"],
                            TARGET_POINT_CLASSES, "sqrt", 3026 + fold, 180, 15)
        ps = fit_binary(x_train, df_train["y_serverGetPoint"], x_valid, df_valid["y_serverGetPoint"],
                        4026 + fold, 180, 15)

        action_f1 = f1_score(df_valid["y_actionId"], pa.argmax(1),
                             labels=TARGET_ACTION_CLASSES, average="macro", zero_division=0)
        point_f1  = f1_score(df_valid["y_pointId"],  pp.argmax(1),
                             labels=TARGET_POINT_CLASSES,  average="macro", zero_division=0)
        server_auc = roc_auc_score(df_valid["y_serverGetPoint"], ps)
        overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc

        results.append({
            "label": label, "seed": s, "fold": fold,
            "n_train": len(df_train),
            "action_f1": round(action_f1, 5), "point_f1": round(point_f1, 5),
            "server_auc": round(server_auc, 5), "overall": round(overall, 5),
        })
        print(f"  [{label}] seed={s} fold={fold}: overall={overall:.5f}  "
              f"action={action_f1:.5f}  point={point_f1:.5f}  (n_train={len(df_train)})", flush=True)
    return results


def main():
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    official_train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    test_pairs = load_test_pairs()

    all_results = []
    for seed in SEEDS_TO_RUN:
        print(f"\n=== seed={seed}: baseline ===", flush=True)
        all_results += run_cv(official_train, splits, None, "baseline", seed)
        print(f"\n=== seed={seed}: +test_prefix ===", flush=True)
        all_results += run_cv(official_train, splits, test_pairs, "+test_prefix", seed)

    df = pd.DataFrame(all_results)
    metric_cols = ["action_f1", "point_f1", "server_auc", "overall"]
    summary = df.groupby("label")[metric_cols].mean().round(5)
    print("\n=== SUMMARY ===")
    print(summary.to_string())

    baseline = df[df["label"] == "baseline"]["overall"].mean()
    extra    = df[df["label"] == "+test_prefix"]["overall"].mean()
    delta    = extra - baseline
    print(f"\nΔoverall = {delta:+.5f}  ({'POSITIVE' if delta > 0 else 'NEGATIVE'})")

    Path("artifacts/test_prefix_pilot.json").write_text(
        json.dumps({"results": all_results,
                    "summary": summary.to_dict(),
                    "delta_overall": round(delta, 5),
                    "test_pairs_added": len(test_pairs)}, indent=2),
        encoding="utf-8"
    )
    print("Saved → artifacts/test_prefix_pilot.json")


if __name__ == "__main__":
    main()
