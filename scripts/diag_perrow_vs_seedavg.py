"""Decisive test: is the local lift an unrealizable seed-averaging artifact?

At submission time each test rally has ONE prefix (one cut), so there is NO
averaging across cut points. But the local OOF averages each rally's prediction
over 5 seeds, and each seed cuts the SAME rally at a DIFFERENT strikeNumber.
Averaging over seeds = averaging over cut points -> an ensemble you CANNOT
reproduce at test time.

This script stacks the 4 dense Route A bases TWO ways and scores both:
  (A) seed-averaged, one row per rally   -> what final_scores.json used (optimistic)
  (B) per-row (rally, seed, cut)          -> what test time actually looks like

If (A) >> (B), the reported local score is inflated by seed-averaging and the
per-row number is the realistic (public/private-like) estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.oof_loader import read_oof, average_over_seeds
from scripts.score_oof import attach_labels, overall
from scripts.postprocess import prior_correct, tune_thresholds, apply_thresholds

BASES = ["lgbm15", "lgbm31", "markov", "phase_lgbm"]
KEYS = ["rally_uid", "seed", "fold", "cut_strikeNumber"]


def _train():
    return pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))


def _perrow_frame(target: str, n_cls: int) -> tuple[np.ndarray, list[str]]:
    """Merge all base per-row OOFs into one feature matrix keyed by KEYS."""
    base = None
    feat_cols: list[str] = []
    cols = [f"p_{i}" for i in range(n_cls)] if target != "server" else ["p_1"]
    for m in BASES:
        df = read_oof(m, target)[KEYS + cols].rename(columns={c: f"{m}__{c}" for c in cols})
        feat_cols += [f"{m}__{c}" for c in cols]
        base = df if base is None else base.merge(df, on=KEYS, how="inner")
    return base, feat_cols


def _stack_oof(X, y, groups, kind, n_cls, n_folds=5):
    out = np.zeros((len(y), n_cls if kind == "multiclass" else 1), dtype=np.float64)
    kf = GroupKFold(n_splits=n_folds)
    for tr, va in kf.split(X, y, groups):
        if kind == "multiclass":
            clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=300, C=1.0)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[va])
            for i, c in enumerate(clf.classes_):
                out[va, int(c)] = p[:, i]
        else:
            clf = LogisticRegression(max_iter=300, C=1.0)
            clf.fit(X[tr], y[tr])
            out[va, 0] = clf.predict_proba(X[va])[:, 1]
    return out


def _nested_f1(corrected, y, groups, n_cls, n_folds=5):
    yhat = np.zeros(len(y), dtype=int)
    for tr, va in GroupKFold(n_splits=n_folds).split(corrected, y, groups):
        thr = tune_thresholds(corrected[tr], y[tr], n_cls)
        yhat[va] = apply_thresholds(corrected[va], thr)
    return float(f1_score(y, yhat, labels=list(range(n_cls)), average="macro", zero_division=0))


def main() -> None:
    train = _train()
    match_per_rally = train.drop_duplicates("rally_uid").set_index("rally_uid")["match"]

    results = {"per_row": {}, "seed_avg": {}}

    for target, n_cls, y_col in [("action", 19, "actionId"), ("point", 10, "pointId"), ("server", 1, "serverGetPoint")]:
        frame, feat_cols = _perrow_frame(target, n_cls)
        labels = attach_labels(frame[KEYS].copy(), train)
        frame = frame.merge(labels[KEYS + [y_col]], on=KEYS, how="left")
        frame["match"] = frame["rally_uid"].map(match_per_rally)
        frame = frame.dropna(subset=[y_col, "match"]).reset_index(drop=True)

        X = frame[feat_cols].fillna(0.0).to_numpy()
        y = frame[y_col].astype(int).to_numpy() if target != "server" else frame[y_col].astype(int).to_numpy()
        groups = frame["match"].to_numpy()
        kind = "binary" if target == "server" else "multiclass"

        # ---- (B) per-row stack: realistic, test-like population ----
        stk = _stack_oof(X, y, groups, kind, n_cls if kind == "multiclass" else 1)
        if target == "server":
            results["per_row"][target] = float(roc_auc_score(y, stk[:, 0]))
        else:
            prior = np.bincount(y, minlength=n_cls).astype(float); prior /= prior.sum()
            corrected = prior_correct(stk, prior)
            results["per_row"][target] = _nested_f1(corrected, y, groups, n_cls)

        # ---- (A) seed-averaged stack: average the per-row OOF stack over (rally) ----
        frame["_pred"] = list(stk)
        agg = frame.groupby("rally_uid").agg(
            pred=("_pred", lambda s: np.mean(np.stack(s.values), axis=0)),
            y=(y_col, "first"),
            match=("match", "first"),
        ).reset_index()
        sa = np.stack(agg["pred"].values)
        ysa = agg["y"].astype(int).to_numpy()
        gsa = agg["match"].to_numpy()
        if target == "server":
            results["seed_avg"][target] = float(roc_auc_score(ysa, sa[:, 0]))
        else:
            prior = np.bincount(ysa, minlength=n_cls).astype(float); prior /= prior.sum()
            corrected = prior_correct(sa, prior)
            results["seed_avg"][target] = _nested_f1(corrected, ysa, gsa, n_cls)

    for pop in ("per_row", "seed_avg"):
        r = results[pop]
        r["overall"] = overall(r["action"], r["point"], r["server"])

    print("\n=== ensemble stack scored two ways (honest nested thresholds) ===")
    df = pd.DataFrame(results).T[["action", "point", "server", "overall"]]
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(df)
    pr, sa = results["per_row"]["overall"], results["seed_avg"]["overall"]
    print(f"\nseed-averaged overall (what final_scores.json reports) = {sa:.4f}")
    print(f"per-row / test-realistic overall                       = {pr:.4f}")
    print(f"seed-averaging inflation                               = {sa - pr:+.4f}")
    print(f"base lgbm15 all-cuts overall (PROGRESS, realistic)     = 0.3027")
    print(f"public clean ensemble (reported by user)               ~ 0.32")


if __name__ == "__main__":
    main()
