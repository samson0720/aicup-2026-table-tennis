"""Re-run the existing LightGBM baseline through the new private-safe CV.

Outputs artifacts/cv_gap_diagnostic.json with per-fold and aggregate scores so
we can compare to artifacts/lgbm_baseline_cv.json (old GroupKFold metric).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from scripts.cv_splits import iter_cv_folds
from scripts.train_lgbm_baseline import (
    TARGET_ACTION_CLASSES,
    TARGET_POINT_CLASSES,
    add_prefix_features,
    feature_columns,
    fit_binary,
    fit_multiclass,
)


def build_one_sample_per_rally(
    view: pd.DataFrame, splits_sub: pd.DataFrame, with_displacement: bool = False
) -> pd.DataFrame:
    """For each rally in `view`, build a single feature row at the seed's cut point."""
    cut_by_rally = dict(zip(splits_sub["rally_uid"], splits_sub["cut_strikeNumber"]))
    rows: list[dict] = []
    for rally_uid, grp in view.groupby("rally_uid", sort=False):
        cut = cut_by_rally.get(int(rally_uid))
        if cut is None:
            continue
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        prefix = grp[grp["strikeNumber"] < cut]
        target = grp[grp["strikeNumber"] == cut]
        if len(prefix) == 0 or len(target) == 0:
            continue
        feats = add_prefix_features(
            prefix, int(target.iloc[0]["strikeNumber"]), with_displacement=with_displacement
        )
        feats["y_actionId"] = int(target.iloc[0]["actionId"])
        feats["y_pointId"] = int(target.iloc[0]["pointId"])
        feats["y_serverGetPoint"] = int(grp.iloc[0]["serverGetPoint"])
        rows.append(feats)
    return pd.DataFrame(rows)


def main() -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))

    per_fold: list[dict] = []
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]
        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty:
            continue

        feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
        x_train, x_valid = df_train[feats], df_valid[feats]

        pa = fit_multiclass(
            x_train, df_train["y_actionId"], x_valid, df_valid["y_actionId"],
            TARGET_ACTION_CLASSES, "sqrt", 2026 + fold, 180, 15,
        )
        pp = fit_multiclass(
            x_train, df_train["y_pointId"], x_valid, df_valid["y_pointId"],
            TARGET_POINT_CLASSES, "sqrt", 3026 + fold, 180, 15,
        )
        ps = fit_binary(
            x_train, df_train["y_serverGetPoint"], x_valid, df_valid["y_serverGetPoint"],
            4026 + fold, 180, 15,
        )

        action_f1 = f1_score(df_valid["y_actionId"], pa.argmax(1),
                             labels=TARGET_ACTION_CLASSES, average="macro", zero_division=0)
        point_f1 = f1_score(df_valid["y_pointId"], pp.argmax(1),
                            labels=TARGET_POINT_CLASSES, average="macro", zero_division=0)
        server_auc = roc_auc_score(df_valid["y_serverGetPoint"], ps)
        overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc

        per_fold.append({
            "seed": seed, "fold": fold,
            "action_macro_f1": float(action_f1),
            "point_macro_f1": float(point_f1),
            "server_auc": float(server_auc),
            "overall": float(overall),
            "n_train": int(len(df_train)), "n_valid": int(len(df_valid)),
        })
        print(f"seed={seed} fold={fold}: overall={overall:.5f}")

    df = pd.DataFrame(per_fold)
    metric_cols = ["action_macro_f1", "point_macro_f1", "server_auc", "overall"]
    by_seed = df.groupby("seed")[metric_cols].mean().reset_index()
    summary = {
        "per_fold": per_fold,
        "by_seed": by_seed.to_dict("records"),
        "mean": df[metric_cols].mean().to_dict(),
        # across-seed std: std of the 5 seed-level mean numbers. This is the
        # noise floor defined by the spec Section 1.2 ("any improvement smaller
        # than the across-seed standard deviation is rejected as noise").
        "std_across_seed": by_seed[metric_cols].std().to_dict(),
        # across-fold std: std of all 25 (seed, fold) cells together. Larger
        # than across-seed std because it includes within-seed fold-to-fold
        # variance. Useful for spotting unstable folds, NOT the noise floor.
        "std_across_fold": df[metric_cols].std().to_dict(),
        "old_cv_reference": {
            "source": "artifacts/lgbm_baseline_cv.json",
            "note": "compare overall and per-target macro-F1 against this file's results.sqrt.oof",
        },
    }
    Path("artifacts/cv_gap_diagnostic.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("mean:",            json.dumps(summary["mean"],            indent=2, ensure_ascii=False))
    print("std_across_seed:", json.dumps(summary["std_across_seed"], indent=2, ensure_ascii=False))
    print("std_across_fold:", json.dumps(summary["std_across_fold"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
