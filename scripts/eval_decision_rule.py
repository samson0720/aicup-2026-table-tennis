"""Compare macro-F1 decision rules on the per-row stacked OOF (P1 gate).

Honest ruler = nested-CV macro-F1 (the same scheme that produced 0.32568).
Kill switch = seed-55 holdout macro-F1 (rules that only win in-sample regress
here, the hill-climb lesson).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from scripts.build_final_perrow import (
    BASES, KEYS, _perrow_features, _stack_oof, _data_dir,
)
from scripts.decision_rule import RULES
from scripts.score_oof import attach_labels, overall

# Targets P1 can affect (server has no threshold rule).
TARGETS = [("action", 19, "actionId"), ("point", 10, "pointId")]
SERVER_AUC_BASELINE = 0.6567  # production server AUC; P1 leaves it untouched.


def _empirical_prior(y, n_cls):
    p = np.bincount(y, minlength=n_cls).astype(float)
    return p / p.sum()


def nested_cv_f1(rule_factory, probs, y, prior, groups, n_cls, n_folds=5):
    yhat = np.zeros(len(y), dtype=int)
    for tr, va in GroupKFold(n_splits=n_folds).split(probs, y, groups):
        rule = rule_factory().fit(probs[tr], y[tr], n_cls, _empirical_prior(y[tr], n_cls))
        yhat[va] = rule.predict(probs[va], n_cls, prior)
    return float(f1_score(y, yhat, labels=list(range(n_cls)),
                          average="macro", zero_division=0))


def seed_holdout_f1(rule_factory, probs, y, prior, seeds, n_cls, holdout=55):
    seeds = np.asarray(seeds)
    if not (seeds == holdout).any():
        raise ValueError(f"holdout seed {holdout} not present in seeds")
    tr = seeds != holdout
    va = seeds == holdout
    rule = rule_factory().fit(probs[tr], y[tr], n_cls, _empirical_prior(y[tr], n_cls))
    yhat = rule.predict(probs[va], n_cls, prior)
    return float(f1_score(y[va], yhat, labels=list(range(n_cls)),
                          average="macro", zero_division=0))


def _build_stack(target, n_cls):
    """Per-row stacked OOF probabilities + aligned labels/groups/seeds."""
    train = pd.read_csv(_data_dir() / "train.csv")
    match_per_rally = train.drop_duplicates("rally_uid").set_index("rally_uid")["match"]
    frame, feat_cols = _perrow_features(target, n_cls)
    y_col = {"action": "actionId", "point": "pointId"}[target]
    lab = attach_labels(frame[KEYS].copy(), train)[KEYS + [y_col]]
    frame = frame.merge(lab, on=KEYS, how="left")
    frame["match"] = frame["rally_uid"].map(match_per_rally)
    frame = frame.dropna(subset=[y_col, "match"]).reset_index(drop=True)
    X = frame[feat_cols].fillna(0.0).to_numpy()
    y = frame[y_col].astype(int).to_numpy()
    groups = frame["match"].to_numpy()
    seeds = frame["seed"].to_numpy()
    stk = _stack_oof(X, y, groups, "multiclass", n_cls)
    return stk, y, groups, seeds


def main(holdout=55):
    result = {}
    per_target = {}
    for target, n_cls, _ in TARGETS:
        stk, y, groups, seeds = _build_stack(target, n_cls)
        prior = _empirical_prior(y, n_cls)
        per_target[target] = {}
        for name, factory in RULES.items():
            per_target[target][name] = {
                "nested_cv": nested_cv_f1(factory, stk, y, prior, groups, n_cls),
                "seed_holdout": seed_holdout_f1(factory, stk, y, prior, seeds, n_cls, holdout),
            }
    # overall (nested-CV) per rule = 0.4*action + 0.4*point + 0.2*server_baseline
    result["per_target"] = per_target
    result["overall_nested"] = {
        name: overall(per_target["action"][name]["nested_cv"],
                      per_target["point"][name]["nested_cv"],
                      SERVER_AUC_BASELINE)
        for name in RULES
    }
    base = result["overall_nested"]["additive_baseline"]
    result["lift_vs_baseline"] = {n: result["overall_nested"][n] - base for n in RULES}
    result["noise_floor"] = 0.00168
    Path("artifacts/decision_rule_scores.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # verdict
    winner, best_lift = "additive_baseline", 0.0
    for n in RULES:
        if n == "additive_baseline":
            continue
        lift = result["lift_vs_baseline"][n]
        a_hold = per_target["action"][n]["seed_holdout"] >= per_target["action"]["additive_baseline"]["seed_holdout"] - 1e-9
        p_hold = per_target["point"][n]["seed_holdout"] >= per_target["point"]["additive_baseline"]["seed_holdout"] - 1e-9
        if lift > 0.00168 and a_hold and p_hold and lift > best_lift:
            winner, best_lift = n, lift
    print(f"\nGATE VERDICT: winner={winner} lift={best_lift:+.5f} "
          f"({'SHIP' if winner != 'additive_baseline' else 'REJECT — keep production rule'})")


if __name__ == "__main__":
    import sys
    main(holdout=int(sys.argv[1]) if len(sys.argv) > 1 else 55)
