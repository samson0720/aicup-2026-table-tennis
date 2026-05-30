"""Diagnostic: which signals predict the NEXT pointId?

Read-only. For every (rally, cut) we form the target = pointId at the cut and a
set of candidate predictors derived strictly from the prefix (strokes before the
cut). We report, per candidate, the conditional-entropy reduction vs the base
entropy of next pointId (a.k.a. mutual information), and the best-single-rule
macro-F1 (predict the most-likely next point given the candidate value).

This tells us whether sequence/transition features (currently unused for point)
have headroom BEFORE we spend time engineering and retraining.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

N_POINT = 10


def _entropy(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def conditional_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """H(y | x) in bits."""
    df = pd.DataFrame({"x": x, "y": y})
    total = len(df)
    h = 0.0
    for _, g in df.groupby("x"):
        w = len(g) / total
        c = np.bincount(g["y"].to_numpy(), minlength=N_POINT)
        h += w * _entropy(c)
    return h


def best_rule_f1(x: np.ndarray, y: np.ndarray) -> float:
    """macro-F1 of 'predict argmax next-point given x' (in-sample upper bound)."""
    df = pd.DataFrame({"x": x, "y": y})
    mode = df.groupby("x")["y"].agg(lambda s: s.value_counts().idxmax())
    yhat = df["x"].map(mode).to_numpy()
    return float(f1_score(y, yhat, labels=list(range(N_POINT)), average="macro", zero_division=0))


def main() -> None:
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    # Use seed 11 only (one cut per rally) so each rally contributes one example.
    s = splits[splits["seed"] == 11][["rally_uid", "cut_strikeNumber"]].drop_duplicates("rally_uid")
    cut_by = dict(zip(s["rally_uid"], s["cut_strikeNumber"]))

    rows = []
    g = train.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False)
    for ruid, grp in g:
        cut = cut_by.get(int(ruid))
        if cut is None:
            continue
        pre = grp[grp["strikeNumber"] < cut]
        tgt = grp[grp["strikeNumber"] == cut]
        if pre.empty or tgt.empty:
            continue
        pt = pre["pointId"].astype(int).tolist()
        pos = pre["positionId"].astype(int).tolist()
        act = pre["actionId"].astype(int).tolist()
        rows.append({
            "next_point": int(tgt["pointId"].iloc[0]),
            "last_point": pt[-1],
            "last2_point": pt[-2] if len(pt) >= 2 else -1,
            "last_position": pos[-1],
            "last_action": act[-1],
            "bigram_point": pt[-2] * 10 + pt[-1] if len(pt) >= 2 else -1,
            "point_pos": pt[-1] * 10 + pos[-1],
            "mode_point": int(pd.Series(pt).value_counts().idxmax()),
            "prefix_len": len(pt),
        })
    d = pd.DataFrame(rows)
    y = d["next_point"].to_numpy()
    base_h = _entropy(np.bincount(y, minlength=N_POINT))
    print(f"n={len(d)}  base H(next_point)={base_h:.3f} bits\n")
    print(f"{'feature':>16} {'H(y|x)':>8} {'MI(bits)':>9} {'MI%':>6} {'rule_f1':>8} {'nvals':>6}")
    cands = ["last_point", "last2_point", "last_position", "last_action",
             "bigram_point", "point_pos", "mode_point", "prefix_len"]
    for c in cands:
        x = d[c].to_numpy()
        hyx = conditional_entropy(x, y)
        mi = base_h - hyx
        print(f"{c:>16} {hyx:>8.3f} {mi:>9.3f} {100*mi/base_h:>5.1f}% "
              f"{best_rule_f1(x, y):>8.3f} {len(np.unique(x)):>6}")
    print("\n(MI% = fraction of next-point uncertainty explained; rule_f1 is an "
          "in-sample upper bound for that single feature.)")


if __name__ == "__main__":
    main()
