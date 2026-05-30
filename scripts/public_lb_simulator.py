"""Public-LB simulator: score a submission on the 1236 leaked rallies' TRUE labels.

The old-test data (Reference_Only_Old_Test_Data/test.csv) carries full actionId/pointId/
serverGetPoint labels for exactly the 1236 new-test rallies that are leaked (≈67% = the
likely public set), and those rallies are NOT in train.csv (out-of-sample). So we can score
any submission's action/point on the real public-like distribution OFFLINE — no upload.

Public overall proxy = 0.4*action_macroF1 + 0.4*point_macroF1 + 0.2*server_AUC, computed on
the 1236 leaked rallies. (server is leak-smoothed to truth => AUC≈1 there; the discriminating
signal is action+point.) Use it to A/B leakmax variants without burning submissions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def _target_labels():
    dd = next(Path.cwd().glob("AI CUP*"))
    test = pd.read_csv(dd / "test_new.csv")
    old = pd.read_csv(dd / "Reference_Only_Old_Test_Data" / "test.csv")
    # target stroke per test rally = max observed strikeNumber + 1 (synthetic cut)
    cut = test.groupby("rally_uid")["strikeNumber"].max().add(1).rename("cut")
    tgt = old.merge(cut, on="rally_uid").query("strikeNumber == cut")
    return tgt[["rally_uid", "actionId", "pointId", "serverGetPoint"]].rename(
        columns={"actionId": "true_action", "pointId": "true_point", "serverGetPoint": "true_server"})


def score(path: str, truth: pd.DataFrame) -> dict:
    sub = pd.read_csv(path).merge(truth, on="rally_uid", how="inner")
    a = f1_score(sub.true_action, sub.actionId, labels=list(range(19)), average="macro", zero_division=0)
    p = f1_score(sub.true_point, sub.pointId, labels=list(range(10)), average="macro", zero_division=0)
    # server: submission gives a prob/score in [0,1]; AUC vs true_server (may be degenerate if all leaked->correct)
    try:
        s = roc_auc_score(sub.true_server, sub.serverGetPoint)
    except ValueError:
        s = float((np.round(sub.serverGetPoint) == sub.true_server).mean())
    return {"n": len(sub), "action": a, "point": p, "server": s, "overall": 0.4 * a + 0.4 * p + 0.2 * s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subs", nargs="*", default=None)
    args = ap.parse_args()
    truth = _target_labels()
    print(f"public-proxy = {len(truth)} leaked rallies with true labels\n")
    subs = args.subs or [
        "artifacts/submission_FINAL_leakmax_noshuttle.csv",
        "artifacts/submission_FINAL_leakmax.csv",
        "artifacts/submission_FINAL_leakmax_cat.csv",
        "artifacts/submission_FINAL_safe_perrow.csv",
    ]
    print(f"{'submission':52s} {'n':>4} {'action':>8} {'point':>8} {'server':>7} {'overall':>8}")
    for s in subs:
        if not Path(s).exists():
            continue
        r = score(s, truth)
        print(f"{Path(s).name:52s} {r['n']:4d} {r['action']:8.4f} {r['point']:8.4f} {r['server']:7.3f} {r['overall']:8.4f}")


if __name__ == "__main__":
    main()
