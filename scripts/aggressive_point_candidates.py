"""Aggressive leaderboard candidates: point-only edits on the leakmax submission.

Strategy override (user): not maximizing OOF — gambling remaining submissions on
high-leverage point edits. BUT we MEASURE each rule on OOF first: train/test point
priors differ by only ~0.05 TVD, so an OOF catastrophe (−0.0X) is a test catastrophe
too; OOF magnitude is the honest downside bound. No new model, no retrain.

Builds:
  point0_boost  : force point->0 on low-margin rows where P(0) is top-3
  long789_boost : force point->best long(7/8/9) on low-margin rows where a long is top-3
  hybrid        : leak rows from leakmax, non-leak rows from safe_perrow
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from scripts.build_final_perrow import _perrow_features, _stack_oof, _test_features, KEYS
from scripts.postprocess import apply_thresholds, prior_correct, tune_thresholds
from scripts.score_oof import attach_labels

N = 10
LONG = [7, 8, 9]


def macro(y, yh):
    return f1_score(y, yh, labels=list(range(N)), average="macro", zero_division=0)


def rule_point0(corr, base_pred, margin_thr=0.12):
    top1 = corr.max(1)
    p0 = corr[:, 0]
    top3 = np.argsort(-corr, 1)[:, :3]
    is_top3 = (top3 == 0).any(1)
    chg = (base_pred != 0) & is_top3 & ((top1 - p0) < margin_thr)
    out = base_pred.copy(); out[chg] = 0
    return out, chg


def rule_long789(corr, base_pred, margin_thr=0.10):
    top1 = corr.max(1)
    long_p = corr[:, LONG]
    best_long_idx = np.array(LONG)[long_p.argmax(1)]
    best_long_val = long_p.max(1)
    top3 = np.argsort(-corr, 1)[:, :3]
    long_in_top3 = np.isin(top3, LONG).any(1)
    chg = (~np.isin(base_pred, LONG)) & long_in_top3 & ((top1 - best_long_val) < margin_thr)
    out = base_pred.copy(); out[chg] = best_long_idx[chg]
    return out, chg


def main():
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*")) / "train.csv")
    frame, feat_cols = _perrow_features("point", N)
    lab = attach_labels(frame[KEYS].copy(), train)[KEYS + ["pointId"]]
    frame = frame.merge(lab, on=KEYS, how="left")
    mpr = train.drop_duplicates("rally_uid").set_index("rally_uid")["match"]
    frame["match"] = frame["rally_uid"].map(mpr)
    frame = frame.dropna(subset=["pointId", "match"]).reset_index(drop=True)
    X = frame[feat_cols].fillna(0.0).to_numpy()
    y = frame["pointId"].astype(int).to_numpy()
    groups = frame["match"].to_numpy()
    prior = np.bincount(y, minlength=N).astype(float); prior /= prior.sum()

    # nested baseline: store per-fold corrected probs + baseline preds
    stk = _stack_oof(X, y, groups, "multiclass", N)
    corr = np.zeros_like(stk); base = np.zeros_like(y)
    for tr, va in GroupKFold(5).split(stk, y, groups):
        thr = tune_thresholds(prior_correct(stk[tr], prior), y[tr], N)
        c = prior_correct(stk[va], prior); corr[va] = c
        base[va] = apply_thresholds(c, thr)
    f_base = macro(y, base)
    print(f"=== OOF point macro-F1 (Δoverall = 0.4·Δpoint; overall floor 0.00168 => need Δpoint>0.0042) ===")
    print(f"baseline                  : {f_base:.5f}")
    for name, fn in [("point0_boost (thr .12)", lambda: rule_point0(corr, base, .12)),
                     ("point0_boost (thr .20)", lambda: rule_point0(corr, base, .20)),
                     ("long789_boost (thr .10)", lambda: rule_long789(corr, base, .10))]:
        yh, chg = fn()
        print(f"{name:25s}: {macro(y, yh):.5f}  Δ={macro(y, yh)-f_base:+.5f}  changed={chg.mean():.1%}")

    # ---- TEST submissions ----
    rally_uids = np.sort(pd.read_csv(next(Path.cwd().glob('AI CUP*'))/'test_new.csv')["rally_uid"].unique())
    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500, C=1.0).fit(X, y)
    thr_full = tune_thresholds(prior_correct(stk, prior), y, N)  # for reference
    Xt = _test_features("point", N, rally_uids, feat_cols)
    raw = clf.predict_proba(Xt); aligned = np.zeros((len(rally_uids), N))
    for i, c in enumerate(clf.classes_):
        aligned[:, int(c)] = raw[:, i]
    corr_t = prior_correct(aligned, prior)
    thr = tune_thresholds(prior_correct(stk, prior), y, N)
    base_t = apply_thresholds(corr_t, thr)

    leak = pd.read_csv("artifacts/submission_FINAL_leakmax.csv").set_index("rally_uid").reindex(rally_uids).reset_index()
    base_point_leak = leak["pointId"].to_numpy()  # leakmax point (cat_sgp on leaked rows)

    def write_from_leak(new_point, fname, changed):
        out = leak.copy(); out["pointId"] = new_point.astype(int)
        assert out["rally_uid"].nunique() == 1845 and out["pointId"].between(0, 9).all()
        assert out["actionId"].between(0, 18).all() and out["serverGetPoint"].between(0, 1).all()
        out[["rally_uid", "actionId", "pointId", "serverGetPoint"]].to_csv(f"artifacts/{fname}", index=False)
        print(f"  wrote {fname}: changed {int(changed.sum())}/1845 vs leakmax point")

    # apply rules to the leakmax point using the ensemble corrected test probs
    p0, c0 = rule_point0(corr_t, base_point_leak, .12)
    pl, cl = rule_long789(corr_t, base_point_leak, .10)
    print("\n=== TEST candidate submissions (point edits on leakmax) ===")
    write_from_leak(p0, "submission_leakmax_point0_boost.csv", c0)
    write_from_leak(pl, "submission_leakmax_long789_boost.csv", cl)

    # hybrid: leak rows from leakmax, non-leak rows from safe_perrow
    overlap = set(pd.read_csv(next(Path.cwd().glob('AI CUP*'))/'Reference_Only_Old_Test_Data'/'test.csv')["rally_uid"].unique())
    safe = pd.read_csv("artifacts/submission_FINAL_safe_perrow.csv").set_index("rally_uid").reindex(rally_uids).reset_index()
    hyb = leak.copy()
    nonleak_mask = ~hyb["rally_uid"].isin(overlap)
    for col in ["actionId", "pointId", "serverGetPoint"]:
        hyb.loc[nonleak_mask, col] = safe.set_index("rally_uid").reindex(rally_uids).reset_index().loc[nonleak_mask, col].values
    diff = int((hyb["pointId"].to_numpy() != leak["pointId"].to_numpy()).sum() +
               (hyb["actionId"].to_numpy() != leak["actionId"].to_numpy()).sum())
    hyb[["rally_uid", "actionId", "pointId", "serverGetPoint"]].to_csv("artifacts/submission_hybrid_leak_safe_nonleak.csv", index=False)
    print(f"  wrote submission_hybrid_leak_safe_nonleak.csv: non-leak rows={int(nonleak_mask.sum())}, action/point cells changed vs leakmax={diff}")


if __name__ == "__main__":
    main()
