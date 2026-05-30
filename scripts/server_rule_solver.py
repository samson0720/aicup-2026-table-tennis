"""serverGetPoint rule solver from the score progression (Step 1 — VERIFIED).

serverGetPoint = did the server (strike-1 player) win the rally. Within a (match,
numberGame), the per-serve score (scoreSelf=server pts, scoreOther=receiver pts) gives
score_sum = scoreSelf+scoreOther which increments by exactly 1 per rally (= play order).
The winner of rally R is whoever's absolute score increments between R and R+1, so
serverGetPoint_R = (winner == server_R). DERIVABLE ONLY IF rally R+1 (score_sum+1) is
present in the same dataset.

Result (verified): TRAIN coverage 89.6%, **accuracy 1.0000** (exact). But TEST coverage
is ~1% — the organizers sampled test rallies NON-CONSECUTIVELY (only 1.0% have their
successor present), deliberately blocking this reconstruction. So it adds nothing beyond
the existing 1236-rally serverGetPoint leak. Kept as verified tooling / for any future use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def solve_server(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-rally (rally_uid, solver_server, confidence). NaN where not derivable."""
    sv = df[df.strikeNumber == 1].copy()
    sv["ssum"] = sv.scoreSelf + sv.scoreOther
    rows = []
    for (_m, _g), grp in sv.groupby(["match", "numberGame"]):
        by = {int(r.ssum): (int(r.gamePlayerId), int(r.gamePlayerOtherId),
                            int(r.scoreSelf), int(r.scoreOther), int(r.rally_uid))
              for r in grp.itertuples()}
        for r in grp.itertuples():
            nxt = by.get(int(r.ssum) + 1)
            sgp, conf = np.nan, "none"
            if nxt is not None:
                cur = {int(r.gamePlayerId): int(r.scoreSelf), int(r.gamePlayerOtherId): int(r.scoreOther)}
                ns, nr, nss, nso, _ = nxt
                nextsc = {ns: nss, nr: nso}
                winner = next((pl for pl in cur if pl in nextsc and nextsc[pl] == cur[pl] + 1), None)
                if winner is not None:
                    sgp, conf = int(winner == int(r.gamePlayerId)), "score_delta"
            rows.append((int(r.rally_uid), sgp, conf))
    return pd.DataFrame(rows, columns=["rally_uid", "solver_server", "confidence"])


def main() -> None:
    dd = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(dd / "train.csv")
    o = solve_server(train).merge(
        train.groupby("rally_uid").serverGetPoint.first().rename("true"), on="rally_uid")
    cov = o.solver_server.notna()
    print(f"TRAIN coverage={cov.mean():.1%} accuracy={(o[cov].solver_server == o[cov].true).mean():.4f}")
    test = pd.read_csv(dd / "test_new.csv")
    ot = solve_server(test)
    print(f"TEST coverage={ot.solver_server.notna().mean():.1%} ({int(ot.solver_server.notna().sum())}/{len(ot)})")
    ot.to_csv("artifacts/test_solver_server.csv", index=False)
    print("wrote artifacts/test_solver_server.csv")


if __name__ == "__main__":
    main()
