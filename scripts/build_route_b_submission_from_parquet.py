"""Build the Route B submission from already-materialized OOF/test parquets.

No LightGBM, no retraining, no libomp: every base probability LightGBM produced
is already on disk under artifacts/oof/. This reproduces exactly the selection
that build_route_b_submission.py applies after training:
  - select per-target prior-correction temperature beta on the chain OOF
    (honest cross-seed CV), then tune thresholds on beta-corrected OOF probs;
  - apply beta + thresholds to the materialized chain *_test parquets;
  - server probability is the stored chain_server p_1.

Output: artifacts/submission_B_chain_betatuned.csv (does NOT overwrite the
committed submission_B_chain.csv so the two can be compared).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.score_oof import attach_labels
from scripts.postprocess import apply_thresholds, prior_correct, select_beta, tune_thresholds

N_ACTION, N_POINT = 19, 10


def _select(target: str, n: int, label_col: str):
    oof = pd.read_parquet(f"artifacts/oof/chain_{target}_{target}.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    oof = attach_labels(oof, train)
    oof = oof[oof[label_col].notna()].copy()
    probs = oof[[f"p_{i}" for i in range(n)]].to_numpy()
    y = oof[label_col].astype(int).to_numpy()
    seed = oof["seed"].to_numpy()
    beta, _ = select_beta(probs, y, seed, n)
    prior = np.bincount(y, minlength=n).astype(float); prior /= prior.sum()
    corrected = prior_correct(probs, prior, beta=beta)
    thr = tune_thresholds(corrected, y, n)
    return beta, prior, thr


def _predict(target: str, n: int, beta: float, prior: np.ndarray, thr: np.ndarray):
    test = pd.read_parquet(f"artifacts/oof/chain_{target}_{target}_test.parquet").sort_values("rally_uid")
    rally = test["rally_uid"].to_numpy()
    p = test[[f"p_{i}" for i in range(n)]].to_numpy()
    pred = apply_thresholds(prior_correct(p, prior, beta=beta), thr)
    return rally, pred


def main() -> None:
    a_beta, a_prior, a_thr = _select("action", N_ACTION, "actionId")
    p_beta, p_prior, p_thr = _select("point", N_POINT, "pointId")
    print(f"selected beta: action={a_beta}  point={p_beta}")

    rally_a, action_pred = _predict("action", N_ACTION, a_beta, a_prior, a_thr)
    rally_p, point_pred = _predict("point", N_POINT, p_beta, p_prior, p_thr)
    assert np.array_equal(rally_a, rally_p)

    server = pd.read_parquet("artifacts/oof/chain_server_server_test.parquet").sort_values("rally_uid")
    assert np.array_equal(server["rally_uid"].to_numpy(), rally_a)

    sub = pd.DataFrame({
        "rally_uid": rally_a.astype(int),
        "actionId": action_pred.astype(int),
        "pointId": point_pred.astype(int),
        "serverGetPoint": np.clip(server["p_1"].to_numpy(), 1e-5, 1 - 1e-5),
    })
    out = Path("artifacts/submission_B_chain_betatuned.csv")
    sub.to_csv(out, index=False)
    print(f"wrote {out}: {sub.shape}, unique rallies={sub['rally_uid'].nunique()}")
    print("action class dist:", sub["actionId"].value_counts().sort_index().to_dict())
    print("point  class dist:", sub["pointId"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
