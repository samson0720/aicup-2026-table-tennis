"""markovp_robust: route the player-conditional base by player familiarity.

Diagnostic (PROGRESS v5 Idea 3 / [[aicup-train-test-shift]]): markovp COLLAPSES on
players unseen in fold-train (action 0.2020->0.1109, point 0.1095->0.0574 — worse than
the player-AGNOSTIC markov there), because its player tables don't transfer. Test is
~44% unseen players. Fix (leakage-free; we know the hitter's id at inference):

    markovp_robust = markovp  where the target hitter was seen in train,
                     markov    (player-agnostic) where unseen.

OOF "seen" = target hitter in that (seed,fold)'s fold-train players; test "seen" =
hitter in the full-train player set. Writes markovp_robust_{action,point} OOF + test,
a drop-in swap for markovp in build_final_perrow BASES.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import read_oof, write_oof
from scripts.predict_test_base import _write_test_parquet

KEYS = ["rally_uid", "seed", "fold", "cut_strikeNumber"]
TARGETS = ("action", "point")
PCOLS = {"action": [f"p_{i}" for i in range(19)], "point": [f"p_{i}" for i in range(10)]}


def _data_dir() -> Path:
    return next(Path.cwd().glob("AI CUP*"))


def _hitter_map(train: pd.DataFrame) -> pd.DataFrame:
    return train[["rally_uid", "strikeNumber", "gamePlayerId"]].rename(
        columns={"strikeNumber": "cut_strikeNumber", "gamePlayerId": "hitter"})


def run_oof(train, splits) -> None:
    hit = _hitter_map(train)
    # fold-train player sets, per (seed,fold)
    fold_players: dict[tuple[int, int], set] = {}
    for seed in splits.seed.unique():
        for fold in splits.fold.unique():
            tr = splits[(splits.seed == seed) & (splits.fold != fold)].rally_uid.unique()
            fold_players[(seed, fold)] = set(train[train.rally_uid.isin(tr)].gamePlayerId.unique())

    for t in TARGETS:
        mp = read_oof("markovp", t).sort_values(KEYS).reset_index(drop=True)
        mk = read_oof("markov", t).sort_values(KEYS).reset_index(drop=True)
        assert mp[KEYS].equals(mk[KEYS]), f"{t}: markovp/markov OOF key mismatch"
        d = mp.merge(hit, on=["rally_uid", "cut_strikeNumber"], how="left")
        seen = np.array([h in fold_players[(s, f)] for h, s, f in
                         zip(d["hitter"], d["seed"], d["fold"])], dtype=bool)
        P = np.where(seen[:, None], mp[PCOLS[t]].to_numpy(), mk[PCOLS[t]].to_numpy())
        out = write_oof("markovp_robust", t, mp.rally_uid.to_numpy(), mp.seed.to_numpy(),
                        mp.fold.to_numpy(), mp.cut_strikeNumber.to_numpy(), P)
        print(f"wrote {out}: rows={len(P)} seen={int(seen.sum())} unseen={int((~seen).sum())}", flush=True)


def run_test(train) -> None:
    test = pd.read_csv(_data_dir() / "test_new.csv")
    te = build_test_dataset(test).sort_values("rally_uid").reset_index(drop=True)
    full_players = set(train.gamePlayerId.unique())
    seen = te["next_gamePlayerId_inferred"].isin(full_players).to_numpy()
    print(f"test: seen={int(seen.sum())} unseen={int((~seen).sum())} of {len(te)}", flush=True)
    for t in TARGETS:
        mp = pd.read_parquet(f"artifacts/oof/markovp_{t}_test.parquet").drop_duplicates("rally_uid").sort_values("rally_uid").reset_index(drop=True)
        mk = pd.read_parquet(f"artifacts/oof/markov_{t}_test.parquet").drop_duplicates("rally_uid").sort_values("rally_uid").reset_index(drop=True)
        assert (mp.rally_uid.values == te.rally_uid.values).all() and (mk.rally_uid.values == te.rally_uid.values).all()
        P = np.where(seen[:, None], mp[PCOLS[t]].to_numpy(), mk[PCOLS[t]].to_numpy())
        _write_test_parquet("markovp_robust", t, te.rally_uid.to_numpy(), P)
        print(f"wrote markovp_robust_{t}_test: {P.shape}", flush=True)


def main() -> None:
    train = pd.read_csv(_data_dir() / "train.csv")
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    run_oof(train, splits)
    run_test(train)


if __name__ == "__main__":
    main()
