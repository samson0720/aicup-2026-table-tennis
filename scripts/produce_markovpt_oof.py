"""Transductive player-transition base from observable rally prefixes.

Unlike markovp, this base fits counts from every visible prefix transition.  At
OOF time it may also adapt from fold-valid transitions strictly before the
hidden cut, mirroring test time where every row in test_new.csv is observable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.oof_loader import write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.produce_markovp_oof import fit_tables, predict


def observable_transition_samples(
    strokes: pd.DataFrame,
    cut_by_rally: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Build prefix->next-stroke rows whose targets are already observable.

    When ``cut_by_rally`` is provided, each rally is truncated before its
    hidden target: only transitions with ``target_strikeNumber < cut`` remain.
    """
    rows: list[dict] = []
    for rally_uid, group in strokes.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber").reset_index(drop=True)
        hidden_cut = cut_by_rally.get(int(rally_uid)) if cut_by_rally is not None else None
        for idx in range(1, len(group)):
            target = group.iloc[idx]
            target_strike = int(target["strikeNumber"])
            if hidden_cut is not None and target_strike >= hidden_cut:
                break
            last = group.iloc[idx - 1]
            rows.append({
                "rally_uid": int(rally_uid),
                "target_strikeNumber": target_strike,
                "last1_actionId": int(last["actionId"]),
                "last1_pointId": int(last["pointId"]),
                "next_gamePlayerId_inferred": int(last["gamePlayerOtherId"]),
                "y_actionId": int(target["actionId"]),
                "y_pointId": int(target["pointId"]),
            })
    return pd.DataFrame(rows)


def _cut_map(splits: pd.DataFrame) -> dict[int, int]:
    return dict(zip(splits["rally_uid"].astype(int), splits["cut_strikeNumber"].astype(int)))


def visible_before_cuts(transitions: pd.DataFrame, cut_by_rally: dict[int, int]) -> pd.DataFrame:
    """Select observable transitions from a pre-built transition frame."""
    cuts = transitions["rally_uid"].map(cut_by_rally)
    return transitions[cuts.notna() & transitions["target_strikeNumber"].lt(cuts)].copy()


def prediction_rows(strokes: pd.DataFrame, cut_by_rally: dict[int, int] | None = None) -> pd.DataFrame:
    """Build the minimal feature rows needed to predict one hidden cut per rally."""
    rows: list[dict] = []
    for rally_uid, group in strokes.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber")
        cut = (
            cut_by_rally.get(int(rally_uid))
            if cut_by_rally is not None
            else int(group["strikeNumber"].max()) + 1
        )
        if cut is None:
            continue
        prefix = group[group["strikeNumber"] < cut]
        if prefix.empty:
            continue
        last = prefix.iloc[-1]
        rows.append({
            "rally_uid": int(rally_uid),
            "target_strikeNumber": int(cut),
            "last1_actionId": int(last["actionId"]),
            "last1_pointId": int(last["pointId"]),
            "next_gamePlayerId_inferred": int(last["gamePlayerOtherId"]),
        })
    return pd.DataFrame(rows)


def run_oof() -> None:
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    all_obs = observable_transition_samples(train)
    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point")}
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits.seed == seed) & (splits.fold != fold)]
        s_valid = splits[(splits.seed == seed) & (splits.fold == fold)]
        valid = prediction_rows(valid_view, _cut_map(s_valid))
        if valid.empty:
            continue
        train_rallies = set(train_view["rally_uid"].unique())
        train_obs = all_obs[all_obs["rally_uid"].isin(train_rallies)]
        valid_obs = visible_before_cuts(all_obs, _cut_map(s_valid))
        adaptive = pd.concat([train_obs, valid_obs], ignore_index=True)
        rally = valid["rally_uid"].to_numpy()
        for target in ("action", "point"):
            probs = predict(valid, target, fit_tables(adaptive, target))
            bag[target]["r"].append(rally)
            bag[target]["s"].append(np.full(len(rally), seed))
            bag[target]["f"].append(np.full(len(rally), fold))
            bag[target]["c"].append(valid["target_strikeNumber"].to_numpy())
            bag[target]["p"].append(probs)
        print(
            f"markovpt seed={seed} fold={fold} valid={len(valid)} "
            f"train_obs={len(train_obs)} valid_obs={len(valid_obs)}",
            flush=True,
        )
    for target in ("action", "point"):
        out = write_oof(
            "markovpt",
            target,
            np.concatenate(bag[target]["r"]),
            np.concatenate(bag[target]["s"]),
            np.concatenate(bag[target]["f"]),
            np.concatenate(bag[target]["c"]),
            np.concatenate(bag[target]["p"], axis=0),
        )
        print(f"wrote {out}", flush=True)


def run_test() -> None:
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    train_obs = observable_transition_samples(train)
    test_obs = observable_transition_samples(test)
    adaptive = pd.concat([train_obs, test_obs], ignore_index=True)
    test_features = prediction_rows(test).sort_values("rally_uid").reset_index(drop=True)
    rally = test_features["rally_uid"].to_numpy()
    for target in ("action", "point"):
        probs = predict(test_features, target, fit_tables(adaptive, target))
        _write_test_parquet("markovpt", target, rally, probs)
        print(f"wrote markovpt_{target}_test: {probs.shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-test", action="store_true")
    args = parser.parse_args()
    run_test() if args.predict_test else run_oof()


if __name__ == "__main__":
    main()
