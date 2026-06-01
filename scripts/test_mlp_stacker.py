"""Test MLP (neural net) meta-learner vs current LR stacker.

Replaces LogisticRegression with a small 2-hidden-layer MLP in the stacking step.
Non-linear interactions between 18 base model probabilities may improve over LR.

Usage:
  conda run -n aicup-tt python -m scripts.test_mlp_stacker
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

from scripts.build_final_perrow import _perrow_features, KEYS, SPEC
from scripts.decision_rule import AdditiveThreshold
from scripts.postprocess import prior_correct
from scripts.score_oof import attach_labels, overall


def _stack_mlp(X, y, groups, kind, n_cls, n_folds=5):
    out = np.zeros((len(y), n_cls if kind == "multiclass" else 1), dtype=np.float64)
    for tr, va in GroupKFold(n_splits=n_folds).split(X, y, groups):
        if kind == "multiclass":
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                max_iter=500,
                alpha=0.01,
                learning_rate_init=0.001,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[va])
            for i, c in enumerate(clf.classes_):
                out[va, int(c)] = p[:, i]
        else:
            clf = MLPClassifier(
                hidden_layer_sizes=(128,),
                activation="relu",
                max_iter=500,
                alpha=0.01,
                learning_rate_init=0.001,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )
            clf.fit(X[tr], y[tr])
            out[va, 0] = clf.predict_proba(X[va])[:, 1]
    return out


def _nested_f1_mlp(stk, y, groups, n_cls, n_folds=5):
    from sklearn.metrics import f1_score
    from scripts.postprocess import apply_thresholds, tune_thresholds
    beta = 0.6 if n_cls == 19 else 0.5
    prior = np.bincount(y, minlength=n_cls).astype(float); prior /= prior.sum()
    yhat = np.zeros(len(y), dtype=int)
    for tr, va in GroupKFold(n_folds).split(stk, y, groups):
        rule = AdditiveThreshold(beta=beta).fit(stk[tr], y[tr], n_cls, prior)
        yhat[va] = rule.predict(stk[va], n_cls, prior)
    return float(f1_score(y, yhat, labels=list(range(n_cls)), average="macro", zero_division=0))


def main():
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*")) / "train.csv")
    match_per_rally = train.drop_duplicates("rally_uid").set_index("rally_uid")["match"]
    scores = {}

    for kind, target, n_cls, y_col in SPEC:
        frame, feat_cols = _perrow_features(target, n_cls)
        lab = attach_labels(frame[KEYS].copy(), train)[KEYS + [y_col]]
        frame = frame.merge(lab, on=KEYS, how="left")
        frame["match"] = frame["rally_uid"].map(match_per_rally)
        frame = frame.dropna(subset=[y_col, "match"]).reset_index(drop=True)

        X = frame[feat_cols].fillna(0.0).to_numpy()
        y = frame[y_col].astype(int).to_numpy()
        groups = frame["match"].to_numpy()

        print(f"\n=== {target} (n={len(y)}) ===", flush=True)
        stk = _stack_mlp(X, y, groups, kind, n_cls if kind == "multiclass" else 1)
        if target == "server":
            scores["server_auc"] = float(roc_auc_score(y, stk[:, 0]))
            print(f"server_auc = {scores['server_auc']:.5f}", flush=True)
        else:
            f1 = _nested_f1_mlp(stk, y, groups, n_cls)
            scores[f"{target}_macro_f1"] = f1
            print(f"{target}_macro_f1 = {f1:.5f}", flush=True)

    scores["overall"] = overall(scores["action_macro_f1"], scores["point_macro_f1"], scores["server_auc"])
    print(f"\n=== MLP stacker overall: {scores['overall']:.6f} ===")
    print(f"vs production LR:       0.371610")
    print(f"lift:                   {scores['overall'] - 0.371610:+.6f}")
    Path("artifacts/mlp_stacker_scores.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
