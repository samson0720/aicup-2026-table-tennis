"""Diagnostic (A/B verification): quantify the per-target prior-temperature (beta)
lift on the existing chain (Route B) OOF, using the real `select_beta` helper.

Read-only on artifacts. Reports, per target (action, point):
  - legacy beta=1 vs selected beta
  - the beta select_beta picks (honest cross-seed CV)
  - honest held-out macro-F1 (seeds 11/22/33 estimate prior, 44/55 evaluated),
    which is the trustworthy lift
  - the score_chain-style reported macro-F1 (selected beta + in-sample threshold
    tuning), to predict what route_b_chain_scores.json will move to.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:
    from scripts.score_oof import attach_labels, overall
    from scripts.postprocess import prior_correct, apply_thresholds, tune_thresholds, select_beta
except ImportError:
    from score_oof import attach_labels, overall
    from postprocess import prior_correct, apply_thresholds, tune_thresholds, select_beta

TARGETS = {"action": ("actionId", 19), "point": ("pointId", 10)}
EVAL_SEEDS = {44, 55}


def macro(y, yhat, n):
    return float(f1_score(y, yhat, labels=list(range(n)), average="macro", zero_division=0))


def load(model, target):
    label_col, n = TARGETS[target]
    p = Path("artifacts/oof") / f"{model}_{target}_{target}.parquet"
    if not p.exists():
        p = Path("artifacts/oof") / f"{model}_{target}.parquet"
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    df = attach_labels(pd.read_parquet(p), train)
    df = df[df[label_col].notna()].copy()
    probs = df[[f"p_{i}" for i in range(n)]].to_numpy()
    y = df[label_col].astype(int).to_numpy()
    seed = df["seed"].to_numpy()
    return probs, y, seed, n


def held_out_f1(probs, y, seed, n, beta):
    """Prior from train seeds, scored on EVAL_SEEDS — honest."""
    tr = ~np.isin(seed, list(EVAL_SEEDS))
    va = np.isin(seed, list(EVAL_SEEDS))
    prior = np.bincount(y[tr], minlength=n).astype(float); prior /= prior.sum()
    return macro(y[va], prior_correct(probs[va], prior, beta=beta).argmax(1), n)


def reported_f1(probs, y, n, beta):
    """score_chain-style: global prior + in-sample tuned thresholds (optimistic)."""
    prior = np.bincount(y, minlength=n).astype(float); prior /= prior.sum()
    corrected = prior_correct(probs, prior, beta=beta)
    thr = tune_thresholds(corrected, y, n)
    return macro(y, apply_thresholds(corrected, thr), n)


def main():
    report = {}
    for tgt in ("action", "point"):
        probs, y, seed, n = load("chain", tgt)
        sel_beta, sel_cv = select_beta(probs, y, seed, n)
        r = {
            "selected_beta": sel_beta,
            "select_beta_cv_f1": round(sel_cv, 4),
            "held_out_f1_beta1": round(held_out_f1(probs, y, seed, n, 1.0), 4),
            "held_out_f1_selected": round(held_out_f1(probs, y, seed, n, sel_beta), 4),
            "reported_f1_beta1": round(reported_f1(probs, y, n, 1.0), 4),
            "reported_f1_selected": round(reported_f1(probs, y, n, sel_beta), 4),
        }
        r["honest_lift"] = round(r["held_out_f1_selected"] - r["held_out_f1_beta1"], 4)
        report[tgt] = r
        print(f"\n=== chain / {tgt} ===")
        for k, v in r.items():
            print(f"  {k:>22}: {v}")

    # Overall impact estimate: both F1 targets carry 0.4 weight each.
    est = 0.4 * report["action"]["honest_lift"] + 0.4 * report["point"]["honest_lift"]
    report["est_overall_lift_from_f1"] = round(est, 4)
    print(f"\nEstimated overall lift from action+point beta (server unchanged): {est:+.4f}")
    print("Noise floor (across-seed std) = 0.00168")
    Path("artifacts/diag_prior_headroom.json").write_text(json.dumps(report, indent=2))
    print("wrote artifacts/diag_prior_headroom.json")


if __name__ == "__main__":
    main()
