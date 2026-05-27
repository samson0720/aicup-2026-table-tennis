from pathlib import Path

import pandas as pd
import pytest

from scripts.cv_splits import build_cv_splits, iter_cv_folds

DATA_DIR = Path(__file__).resolve().parents[1] / "AI CUP競賽資料集"


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv")


@pytest.fixture(scope="module")
def splits(train_df: pd.DataFrame) -> pd.DataFrame:
    return build_cv_splits(train_df, seeds=(11, 22, 33, 44, 55), n_folds=5)


def test_schema_columns(splits: pd.DataFrame) -> None:
    expected = {"rally_uid", "match", "seed", "fold", "cut_strikeNumber", "phase_bucket"}
    assert expected.issubset(splits.columns), f"missing: {expected - set(splits.columns)}"


def test_row_count(splits: pd.DataFrame) -> None:
    # 14,995 rallies × 5 seeds = 74,975 rows.
    assert len(splits) == 14_995 * 5


def test_match_groups_share_a_fold(splits: pd.DataFrame) -> None:
    bad = (
        splits.groupby(["seed", "match"])["fold"].nunique()
        .reset_index(name="n_folds")
    )
    offenders = bad[bad["n_folds"] != 1]
    assert offenders.empty, offenders.head().to_dict("records")


def test_fold_range(splits: pd.DataFrame) -> None:
    assert splits["fold"].between(0, 4).all()


def test_phase_balance_within_tolerance(splits: pd.DataFrame) -> None:
    # Per seed, each fold's phase share within 5pp of seed-wide share.
    for seed, sub in splits.groupby("seed"):
        global_share = sub["phase_bucket"].value_counts(normalize=True)
        per_fold = (
            sub.groupby("fold")["phase_bucket"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
        )
        diff = (per_fold - global_share).abs()
        worst = float(diff.values.max())
        assert worst <= 0.05, f"seed {seed} worst phase-share drift {worst:.3f}\n{diff}"


def test_cut_within_rally_length(splits: pd.DataFrame, train_df: pd.DataFrame) -> None:
    rally_len = train_df.groupby("rally_uid")["strikeNumber"].max().to_dict()
    bad = splits[splits["cut_strikeNumber"] > splits["rally_uid"].map(rally_len)]
    assert bad.empty, bad.head().to_dict("records")
    assert (splits["cut_strikeNumber"] >= 2).all()


def test_cut_varies_across_seeds(splits: pd.DataFrame, train_df: pd.DataFrame) -> None:
    rally_len = train_df.groupby("rally_uid")["strikeNumber"].max()
    long_rallies = rally_len[rally_len >= 4].index
    sub = splits[splits["rally_uid"].isin(long_rallies)]
    distinct = sub.groupby("rally_uid")["cut_strikeNumber"].nunique()
    coincide = (distinct < 2).mean()
    assert coincide < 0.01, f"{coincide:.3%} of long rallies have identical cuts across seeds"


def test_iter_cv_folds_no_rally_leak(splits: pd.DataFrame, train_df: pd.DataFrame) -> None:
    seen_folds = set()
    for seed, fold, train_view, valid_view in iter_cv_folds(train_df, splits):
        assert isinstance(train_view, pd.DataFrame)
        assert isinstance(valid_view, pd.DataFrame)
        train_rallies = set(train_view["rally_uid"].unique())
        valid_rallies = set(valid_view["rally_uid"].unique())
        assert train_rallies.isdisjoint(valid_rallies), (
            f"seed={seed} fold={fold} leaks {len(train_rallies & valid_rallies)} rallies"
        )
        seen_folds.add((seed, fold))
    assert len(seen_folds) == 5 * 5  # 5 seeds × 5 folds
