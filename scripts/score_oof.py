"""Score OOF parquets uniformly."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def attach_labels(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Join the cut-target label onto an OOF parquet.

    For each OOF row, look up the train row identified by
    (rally_uid, cut_strikeNumber == strikeNumber) and bring over
    actionId, pointId, serverGetPoint as the labels at the cut.
    """
    # Per (rally_uid, cut_strikeNumber), find the train row whose strikeNumber == cut.
    key = train[["rally_uid", "strikeNumber", "actionId", "pointId", "serverGetPoint"]]
    return df.merge(
        key.rename(columns={"strikeNumber": "cut_strikeNumber"}),
        on=["rally_uid", "cut_strikeNumber"], how="left",
    )


def score_action(df_with_labels: pd.DataFrame) -> float:
    cols = [f"p_{i}" for i in range(19)]
    yhat = df_with_labels[cols].to_numpy().argmax(1)
    return float(f1_score(df_with_labels["actionId"], yhat,
                          labels=list(range(19)), average="macro", zero_division=0))


def score_point(df_with_labels: pd.DataFrame) -> float:
    cols = [f"p_{i}" for i in range(10)]
    yhat = df_with_labels[cols].to_numpy().argmax(1)
    return float(f1_score(df_with_labels["pointId"], yhat,
                          labels=list(range(10)), average="macro", zero_division=0))


def score_server(df_with_labels: pd.DataFrame) -> float:
    return float(roc_auc_score(df_with_labels["serverGetPoint"], df_with_labels["p_1"]))


def overall(action_f1: float, point_f1: float, server_auc: float) -> float:
    return 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc


def score_model(model: str) -> dict:
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    out = {}
    for tgt, fn in (("action", score_action), ("point", score_point), ("server", score_server)):
        df = pd.read_parquet(f"artifacts/oof/{model}_{tgt}.parquet")
        df = attach_labels(df, train)
        out[tgt] = fn(df)
    out["overall"] = overall(out["action"], out["point"], out["server"])
    return out


if __name__ == "__main__":
    import sys
    models = sys.argv[1:] or ["lgbm15", "lgbm31", "markov", "phase_lgbm"]
    table = {m: score_model(m) for m in models}
    print(json.dumps(table, indent=2, ensure_ascii=False))
    Path("artifacts/base_oof_scores.json").write_text(json.dumps(table, indent=2, ensure_ascii=False))
