"""Track A gate: does ensembling leak-feature point models (cat_sgp + lgbm_sgp)
beat cat_sgp alone on honest point macro-F1 (prior-correct + nested thresholds)?

Honest nested scoring mirrors build_final_perrow._nested_f1: tune thresholds on
train folds, score held-out folds. Compares cat_sgp-alone vs the mean-prob ensemble.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from scripts.oof_loader import read_oof
from scripts.postprocess import apply_thresholds, prior_correct, tune_thresholds

N = 10
KEYS = ["rally_uid", "seed", "fold", "cut_strikeNumber"]
PCOLS = [f"p_{i}" for i in range(N)]


def _labels():
    tr = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    # one label per (rally) cut — reuse score_oof.attach_labels semantics via merge on rally_uid+cut
    from scripts.score_oof import attach_labels
    base = read_oof("cat_sgp", "point")
    lab = attach_labels(base, tr)
    return lab


def _nested_point_f1(probs, y, groups):
    prior = np.bincount(y, minlength=N).astype(float); prior /= prior.sum()
    yhat = np.zeros(len(y), dtype=int)
    for tr, va in GroupKFold(n_splits=5).split(probs, y, groups):
        thr = tune_thresholds(prior_correct(probs[tr], prior), y[tr], N)
        yhat[va] = apply_thresholds(prior_correct(probs[va], prior), thr)
    return float(f1_score(y, yhat, labels=list(range(N)), average="macro", zero_division=0))


def main() -> None:
    cat = read_oof("cat_sgp", "point").sort_values(KEYS).reset_index(drop=True)
    lgb = read_oof("lgbm_sgp", "point").sort_values(KEYS).reset_index(drop=True)
    assert (cat[KEYS].values == lgb[KEYS].values).all(), "key misalignment"

    lab = _labels().sort_values(KEYS).reset_index(drop=True)
    assert (lab[KEYS].values == cat[KEYS].values).all(), "label key misalignment"
    y = lab["pointId"].astype(int).to_numpy()
    groups = cat["rally_uid"].to_numpy()

    Pcat = cat[PCOLS].to_numpy()
    Plgb = lgb[PCOLS].to_numpy()
    Pens = (Pcat + Plgb) / 2.0

    f_cat = _nested_point_f1(Pcat, y, groups)
    f_lgb = _nested_point_f1(Plgb, y, groups)
    f_ens = _nested_point_f1(Pens, y, groups)
    print(f"cat_sgp alone   point F1 = {f_cat:.4f}")
    print(f"lgbm_sgp alone  point F1 = {f_lgb:.4f}")
    print(f"ENSEMBLE (mean) point F1 = {f_ens:.4f}")
    print(f"ensemble lift vs cat_sgp = {f_ens - f_cat:+.4f}  (point floor 0.00506)")


if __name__ == "__main__":
    main()
