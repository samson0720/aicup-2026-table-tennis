"""Step 2: restructure the POINT decision (no new base model).

Reuses the production per-row POINT stack (8 bases -> LR meta OOF probs) and compares
decision RULES via the same match-grouped nested CV used in build_final_perrow:
  A baseline   : prior_correct + joint 10-class threshold tuning (= production rule)
  B hierarchical: decouple terminal(0) from spatial(1-9) — tune a terminal threshold,
                  tune 1-9 thresholds in the renormalized 1-9 subspace
  C +geometry  : blend lateral{fh,mid,bh}/depth{short,half,long} marginals (verified map)

Only a point-rule change => Δoverall = 0.4·Δpoint, so the overall floor 0.00168 needs
Δpoint > ~0.0042. Reports point macro-F1 per rule.
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

N = 10
LAT = {k: (k - 1) % 3 for k in range(1, 10)}   # 0 fh{1,4,7} 1 mid{2,5,8} 2 bh{3,6,9}
DEP = {k: (k - 1) // 3 for k in range(1, 10)}   # 0 short 1 half 2 long
EPS = 1e-9


def macro(y, yhat):
    return f1_score(y, yhat, labels=list(range(N)), average="macro", zero_division=0)


# ---- Rule A: production baseline -------------------------------------------
def fitA(stk, y, prior):
    return tune_thresholds(prior_correct(stk, prior), y, N)

def predA(stk, prior, thr):
    return apply_thresholds(prior_correct(stk, prior), thr)


# ---- Rule B: hierarchical terminal/spatial ---------------------------------
def _spatial(corrected):
    s = corrected[:, 1:N]
    return s / np.clip(s.sum(1, keepdims=True), EPS, None)

def fitB(stk, y, prior):
    corrected = prior_correct(stk, prior)
    sp = _spatial(corrected)
    m19 = y >= 1
    thr9 = tune_thresholds(sp[m19], (y[m19] - 1), N - 1)  # tune on true 1-9 rows
    # pick terminal threshold tau maximizing 10-class macro-F1
    best_tau, best = 0.5, -1.0
    sp_pred = apply_thresholds(sp, thr9) + 1
    for tau in np.linspace(0.05, 0.8, 16):
        pred = np.where(corrected[:, 0] >= tau, 0, sp_pred)
        f = macro(y, pred)
        if f > best:
            best, best_tau = f, tau
    return thr9, best_tau

def predB(stk, prior, params):
    thr9, tau = params
    corrected = prior_correct(stk, prior)
    sp_pred = apply_thresholds(_spatial(corrected), thr9) + 1
    return np.where(corrected[:, 0] >= tau, 0, sp_pred)


# ---- Rule C: hierarchical + lateral/depth geometry blend -------------------
def fitC(stk, y, prior):
    corrected = prior_correct(stk, prior)
    sp = _spatial(corrected)
    m19 = y >= 1
    best = (-1.0, None)
    for beta in (0.0, 0.15, 0.3, 0.5):
        for gamma in (0.0, 0.15, 0.3, 0.5):
            scored = _geom_score(sp, beta, gamma)
            thr9 = tune_thresholds(scored[m19], (y[m19] - 1), N - 1)
            sp_pred = apply_thresholds(scored, thr9) + 1
            for tau in np.linspace(0.05, 0.8, 16):
                pred = np.where(corrected[:, 0] >= tau, 0, sp_pred)
                f = macro(y, pred)
                if f > best[0]:
                    best = (f, (beta, gamma, thr9, tau))
    return best[1]

def _geom_score(sp, beta, gamma):
    # sp: (n,9) spatial probs over classes 1..9 (index 0 -> class 1)
    lat = np.zeros((len(sp), 3)); dep = np.zeros((len(sp), 3))
    for j in range(9):
        lat[:, LAT[j + 1]] += sp[:, j]
        dep[:, DEP[j + 1]] += sp[:, j]
    score = np.log(sp + EPS).copy()
    for j in range(9):
        score[:, j] += beta * np.log(lat[:, LAT[j + 1]] + EPS) + gamma * np.log(dep[:, DEP[j + 1]] + EPS)
    return score

def predC(stk, prior, params):
    beta, gamma, thr9, tau = params
    corrected = prior_correct(stk, prior)
    sp_pred = apply_thresholds(_geom_score(_spatial(corrected), beta, gamma), thr9) + 1
    return np.where(corrected[:, 0] >= tau, 0, sp_pred)


def nested(stk, y, groups, fit, pred):
    prior = np.bincount(y, minlength=N).astype(float); prior /= prior.sum()
    yhat = np.zeros_like(y)
    for tr, va in GroupKFold(5).split(stk, y, groups):
        p = fit(stk[tr], y[tr], prior)
        yhat[va] = pred(stk[va], prior, p)
    return macro(y, yhat)


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
    stk = _stack_oof(X, y, groups, "multiclass", N)
    print(f"rows={len(y)} terminal(class0) share={(y==0).mean():.3f}")
    for name, fit, pred in [("A baseline", fitA, predA), ("B hierarchical", fitB, predB),
                            ("C +geometry", fitC, predC)]:
        f = nested(stk, y, groups, fit, pred)
        print(f"{name:16s} point macro-F1 = {f:.5f}")


if __name__ == "__main__":
    main()
