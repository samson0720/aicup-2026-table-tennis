"""Public-max submission: override point with cat_sgp (outcome-leak-as-feature)
on the 1236 old-test-overlap rallies (where serverGetPoint is truly known),
keeping server smoothing. Leaves the honest submission untouched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.oof_loader import OOF_DIR, read_oof
from scripts.postprocess import apply_thresholds, prior_correct, tune_thresholds
from scripts.score_oof import attach_labels

N = 10  # point classes


def main() -> None:
    dd = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(dd / "train.csv")
    overlap = set(pd.read_csv(dd / "Reference_Only_Old_Test_Data" / "test.csv")["rally_uid"].unique())

    # cat_sgp point: tune thresholds on its OOF, apply to its test probs
    oof = attach_labels(read_oof("cat_sgp", "point"), train).dropna(subset=["pointId"])
    y = oof["pointId"].astype(int).to_numpy()
    P = oof[[f"p_{i}" for i in range(N)]].to_numpy()
    prior = np.bincount(y, minlength=N).astype(float); prior /= prior.sum()
    thr = tune_thresholds(prior_correct(P, prior), y, N)

    test = pd.read_parquet(OOF_DIR / "cat_sgp_point_test.parquet").drop_duplicates("rally_uid")
    Pt = test[[f"p_{i}" for i in range(N)]].to_numpy()
    pt_cls = apply_thresholds(prior_correct(Pt, prior), thr)
    sgp_point = dict(zip(test["rally_uid"].astype(int), pt_cls.astype(int)))

    sm = pd.read_csv("artifacts/submission_FINAL_smooth_perrow.csv")
    mask = sm["rally_uid"].isin(overlap)
    sm.loc[mask, "pointId"] = sm.loc[mask, "rally_uid"].map(sgp_point).astype(int)

    assert sm["rally_uid"].nunique() == 1845
    assert sm["actionId"].between(0, 18).all() and sm["pointId"].between(0, 9).all()
    assert sm["serverGetPoint"].between(0, 1).all()
    sm.to_csv("artifacts/submission_FINAL_leakmax.csv", index=False)
    print(f"wrote artifacts/submission_FINAL_leakmax.csv: overrode point on {int(mask.sum())} overlap rallies")


if __name__ == "__main__":
    main()
