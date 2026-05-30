"""Step 3 (teammate's per-target beta): does prior-temperature help OUR real pipeline?

Their select_beta picked beta on ARGMAX macro-F1 (no thresholds). Our production is
prior_correct(beta=1) + NESTED THRESHOLD TUNING, and the thresholds already move
per-class boundaries — so beta's argmax gain may vanish under threshold tuning (the P1
false-positive failure mode). This sweeps beta for action(19) & point(10) THROUGH the
real nested-threshold ruler on our 8-base stack. Δoverall = 0.4·(Δaction+Δpoint).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from scripts.build_final_perrow import _perrow_features, _stack_oof, KEYS
from scripts.postprocess import apply_thresholds, prior_correct, tune_thresholds
from scripts.score_oof import attach_labels


def nested_f1(stk, y, groups, n_cls, beta):
    prior = np.bincount(y, minlength=n_cls).astype(float); prior /= prior.sum()
    yhat = np.zeros_like(y)
    for tr, va in GroupKFold(5).split(stk, y, groups):
        thr = tune_thresholds(prior_correct(stk[tr], prior, beta), y[tr], n_cls)
        yhat[va] = apply_thresholds(prior_correct(stk[va], prior, beta), thr)
    return float(f1_score(y, yhat, labels=list(range(n_cls)), average="macro", zero_division=0))


def argmax_f1(stk, y, groups, n_cls, beta):
    """Their metric: per-fold prior, argmax (NO thresholds)."""
    yhat = np.zeros_like(y)
    for tr, va in GroupKFold(5).split(stk, y, groups):
        prior = np.bincount(y[tr], minlength=n_cls).astype(float); prior /= prior.sum()
        yhat[va] = prior_correct(stk[va], prior, beta).argmax(1)
    return float(f1_score(y, yhat, labels=list(range(n_cls)), average="macro", zero_division=0))


def load(target, n_cls, train):
    frame, feat_cols = _perrow_features(target, n_cls)
    lab = attach_labels(frame[KEYS].copy(), train)[KEYS + [{"action": "actionId", "point": "pointId"}[target]]]
    frame = frame.merge(lab, on=KEYS, how="left")
    mpr = train.drop_duplicates("rally_uid").set_index("rally_uid")["match"]
    frame["match"] = frame["rally_uid"].map(mpr)
    ycol = {"action": "actionId", "point": "pointId"}[target]
    frame = frame.dropna(subset=[ycol, "match"]).reset_index(drop=True)
    X = frame[feat_cols].fillna(0.0).to_numpy()
    y = frame[ycol].astype(int).to_numpy()
    groups = frame["match"].to_numpy()
    return _stack_oof(X, y, groups, "multiclass", n_cls), y, groups


def main():
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*")) / "train.csv")
    betas = [round(b, 2) for b in np.arange(0.0, 1.51, 0.1)]
    summary = {}
    for target, n_cls in [("action", 19), ("point", 10)]:
        stk, y, groups = load(target, n_cls, train)
        print(f"\n=== {target} (n={len(y)}) ===")
        print(f"{'beta':>5} | {'argmax_f1 (their ruler)':>22} | {'nested_thr_f1 (OUR ruler)':>26}")
        rows = []
        for b in betas:
            af = argmax_f1(stk, y, groups, n_cls, b)
            nf = nested_f1(stk, y, groups, n_cls, b)
            rows.append((b, af, nf))
            print(f"{b:5.1f} | {af:22.5f} | {nf:26.5f}")
        nf1 = dict((b, nf) for b, _, nf in rows)[1.0]
        best_b, best_nf = max(((b, nf) for b, _, nf in rows), key=lambda t: t[1])
        summary[target] = (nf1, best_b, best_nf)
        print(f"  OUR ruler: beta=1.0 -> {nf1:.5f} | best beta={best_b} -> {best_nf:.5f} | lift={best_nf-nf1:+.5f}")
    da = summary["action"][2] - summary["action"][0]
    dp = summary["point"][2] - summary["point"][0]
    print(f"\n=== OUR-ruler overall lift if per-target beta adopted = 0.4*({da:+.5f}+{dp:+.5f}) = {0.4*(da+dp):+.5f} (floor 0.00168) ===")


if __name__ == "__main__":
    main()
