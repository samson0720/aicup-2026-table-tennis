"""Private-safe CV splits for AI Cup 2026 table-tennis."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


PHASE_BUCKETS = ("phase0", "phase1", "phase2plus")


def _phase_bucket(target_strike: int) -> str:
    if target_strike == 2:
        return "phase0"
    if target_strike == 3:
        return "phase1"
    return "phase2plus"


def build_cv_splits(
    train: pd.DataFrame,
    seeds: Sequence[int] = (11, 22, 33, 44, 55),
    n_folds: int = 5,
) -> pd.DataFrame:
    """Return one row per (rally_uid, seed) with assigned fold and cut point."""
    rallies = (
        train.groupby("rally_uid", sort=False)
        .agg(match=("match", "first"), rally_len=("strikeNumber", "max"))
        .reset_index()
    )
    rallies = rallies[rallies["rally_len"] >= 2].copy()
    matches = rallies["match"].unique()

    rows: list[dict] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)

        # Phase-stratified assignment: sample provisional cuts per rally, count
        # phase buckets per match, then greedily assign each match to the fold
        # with the smallest load in that match's dominant phase. Keeps per-fold
        # phase shares close to global.
        cut_provisional = rng.integers(2, rallies["rally_len"].to_numpy() + 1)
        bucket_provisional = np.array([_phase_bucket(int(c)) for c in cut_provisional])
        match_phase = (
            pd.DataFrame({"match": rallies["match"].to_numpy(), "bucket": bucket_provisional})
            .groupby(["match", "bucket"]).size().unstack(fill_value=0)
            .reindex(columns=list(PHASE_BUCKETS), fill_value=0)
        )
        fold_load = {f: {b: 0 for b in PHASE_BUCKETS} for f in range(n_folds)}
        fold_of_match: dict[int, int] = {}
        order = matches.copy()
        rng.shuffle(order)
        for m in order:
            counts = match_phase.loc[int(m)] if int(m) in match_phase.index else None
            if counts is not None and counts.sum() > 0:
                dominant = str(counts.idxmax())
            else:
                dominant = PHASE_BUCKETS[2]
            target_fold = min(range(n_folds), key=lambda f: fold_load[f][dominant])
            fold_of_match[int(m)] = target_fold
            if counts is not None:
                for b in PHASE_BUCKETS:
                    fold_load[target_fold][b] += int(counts.get(b, 0))

        # Now sample real cuts for the output rows. Re-seed deterministically
        # so cuts depend on `seed` but not on the order-shuffle randomness.
        cut_rng = np.random.default_rng(seed + 10_000)
        for _, r in rallies.iterrows():
            cut = int(cut_rng.integers(2, int(r["rally_len"]) + 1))
            rows.append(
                {
                    "rally_uid": int(r["rally_uid"]),
                    "match": int(r["match"]),
                    "seed": int(seed),
                    "fold": fold_of_match[int(r["match"])],
                    "cut_strikeNumber": cut,
                    "phase_bucket": _phase_bucket(cut),
                }
            )
    return pd.DataFrame(rows)


def iter_cv_folds(
    train: pd.DataFrame,
    splits: pd.DataFrame,
) -> Iterator[tuple[int, int, pd.DataFrame, pd.DataFrame]]:
    """Yield (seed, fold, train_view, valid_view) per (seed, fold).

    For each (seed, fold), valid_view contains all rows whose rally_uid is in
    that fold; train_view contains rows from every other fold's rallies. The
    only guarantee here is that no rally_uid appears in both train and valid
    for the same (seed, fold). Downstream scripts decide how to slice prefixes,
    fit encoders, etc.
    """
    train_by_rally = train.set_index("rally_uid", drop=False).sort_index()
    for seed in sorted(splits["seed"].unique()):
        seed_splits = splits[splits["seed"] == seed]
        for fold in sorted(seed_splits["fold"].unique()):
            valid_meta = seed_splits[seed_splits["fold"] == fold]
            train_meta = seed_splits[seed_splits["fold"] != fold]
            valid_rallies = valid_meta["rally_uid"].to_numpy()
            train_rallies = train_meta["rally_uid"].to_numpy()
            valid_view = train_by_rally.loc[valid_rallies].reset_index(drop=True)
            train_view = train_by_rally.loc[train_rallies].reset_index(drop=True)
            yield int(seed), int(fold), train_view, valid_view


def _find_data_dir() -> Path:
    for p in Path.cwd().glob("AI CUP*"):
        if p.is_dir():
            return p
    raise FileNotFoundError("AI CUP data directory not found from cwd")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("artifacts/cv_splits.parquet"))
    args = parser.parse_args()

    data_dir = _find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    splits = build_cv_splits(train, seeds=tuple(args.seeds), n_folds=args.folds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(args.out, index=False)
    print(f"wrote {args.out}: {splits.shape}")


if __name__ == "__main__":
    main()
