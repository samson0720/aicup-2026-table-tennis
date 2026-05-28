"""Diagnostic: how much of the ensemble's local lift is in-sample threshold tuning?

Scores best single bases AND the final ensemble stack on the SAME population
(one row per rally, seed-averaged) under three regimes:
  raw   = argmax of raw probs (no postproc)
  insmp = prior_correct + thresholds tuned AND scored on the same labels
  nested= prior_correct + thresholds tuned per match-fold, scored on held-out

If `insmp` >> `nested` for action/point, the reported local score is inflated
by in-sample threshold tuning and will not transfer to public/private.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.oof_loader import read_oof, average_over_seeds
from scripts.score_oof import attach_labels, overall
from scripts.postprocess import prior_correct, tune_thresholds, apply_thresholds


def _attached() -> pd.DataFrame:
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    sample = read_oof("lgbm15", "action")
    match = train.drop_duplicates("rally_uid")[["rally_uid", "match"]]
    return (
        attach_labels(sample, train)[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
        .drop_duplicates("rally_uid")
        .merge(match, on="rally_uid", how="left")
        .reset_index(drop=True)
    )


def _probs_for(model: str, target: str, n_cls: int, order: np.ndarray) -> np.ndarray:
    if model == "ensemble":
        df = pd.read_parquet(f"artifacts/final_stack_{target}.parquet")
    else:
        df = average_over_seeds(read_oof(model, target), target)
    cols = [f"p_{i}" for i in range(n_cls)] if target != "server" else ["p_1"]
    df = df.set_index("rally_uid").reindex(order)
    return df[cols].to_numpy()


def _raw_f1(probs, y, n_cls):
    return float(f1_score(y, probs.argmax(1), labels=list(range(n_cls)),
                          average="macro", zero_division=0))


def _insample_f1(corrected, y, n_cls):
    thr = tune_thresholds(corrected, y, n_cls)
    return float(f1_score(y, apply_thresholds(corrected, thr),
                          labels=list(range(n_cls)), average="macro", zero_division=0))


def _nested_f1(corrected, y, groups, n_cls, n_folds=5):
    yhat = np.zeros(len(y), dtype=int)
    kf = GroupKFold(n_splits=n_folds)
    for tr, va in kf.split(corrected, y, groups):
        thr = tune_thresholds(corrected[tr], y[tr], n_cls)
        yhat[va] = apply_thresholds(corrected[va], thr)
    return float(f1_score(y, yhat, labels=list(range(n_cls)),
                          average="macro", zero_division=0))


def main() -> None:
    att = _attached()
    order = att["rally_uid"].to_numpy()
    groups = att["match"].to_numpy()
    y_action = att["actionId"].to_numpy()
    y_point = att["pointId"].to_numpy()
    y_server = att["serverGetPoint"].to_numpy()

    prior_a = np.bincount(y_action, minlength=19).astype(float); prior_a /= prior_a.sum()
    prior_p = np.bincount(y_point, minlength=10).astype(float); prior_p /= prior_p.sum()

    models = ["lgbm15", "lgbm31", "phase_lgbm", "ensemble"]
    rows = []
    for m in models:
        pa = _probs_for(m, "action", 19, order)
        pp = _probs_for(m, "point", 10, order)
        ps = _probs_for(m, "server", 1, order).ravel()

        ca = prior_correct(pa, prior_a)
        cp = prior_correct(pp, prior_p)

        a_raw = _raw_f1(pa, y_action, 19)
        a_ins = _insample_f1(ca, y_action, 19)
        a_nst = _nested_f1(ca, y_action, groups, 19)

        p_raw = _raw_f1(pp, y_point, 10)
        p_ins = _insample_f1(cp, y_point, 10)
        p_nst = _nested_f1(cp, y_point, groups, 10)

        s_auc = float(roc_auc_score(y_server, ps))

        rows.append({
            "model": m,
            "act_raw": a_raw, "act_insmp": a_ins, "act_nested": a_nst,
            "pt_raw": p_raw, "pt_insmp": p_ins, "pt_nested": p_nst,
            "srv_auc": s_auc,
            "overall_insmp": overall(a_ins, p_ins, s_auc),
            "overall_nested": overall(a_nst, p_nst, s_auc),
        })

    df = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.4f}")
    print("\n=== action macro-F1 ===")
    print(df[["act_raw", "act_insmp", "act_nested"]])
    print("\n=== point macro-F1 ===")
    print(df[["pt_raw", "pt_insmp", "pt_nested"]])
    print("\n=== server AUC (one row per rally) ===")
    print(df[["srv_auc"]])
    print("\n=== overall: in-sample threshold vs honest nested threshold ===")
    print(df[["overall_insmp", "overall_nested"]])

    base_best = df.loc[["lgbm15", "lgbm31", "phase_lgbm"], "overall_nested"].max()
    ens = df.loc["ensemble", "overall_nested"]
    print(f"\nHONEST ensemble overall (nested) = {ens:.4f}")
    print(f"HONEST best single base (nested) = {base_best:.4f}")
    print(f"HONEST ensemble lift over best base = {ens - base_best:+.4f}")
    print(f"(reported in-sample ensemble overall was 0.3497; noise floor 0.00168)")


if __name__ == "__main__":
    main()
