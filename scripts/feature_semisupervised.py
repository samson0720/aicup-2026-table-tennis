"""Player-level feature distributions from train + test_new prefix rows.

Only observable stroke features are used. No target labels from test are read.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURE_COLS = ("strikeId", "handId", "strengthId", "spinId", "positionId")
MAX_VAL = {"strikeId": 3, "handId": 3, "strengthId": 3, "spinId": 5, "positionId": 3}


def build_player_feature_dist(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    pool = pd.concat(
        [train[["gamePlayerId", *FEATURE_COLS]], test[["gamePlayerId", *FEATURE_COLS]]],
        ignore_index=True,
    )
    rows: list[dict[str, float | int]] = []
    for player, group in pool.groupby("gamePlayerId", sort=True):
        row: dict[str, float | int] = {"gamePlayerId": int(player), "n_strokes": int(len(group))}
        for col in FEATURE_COLS:
            rates = group[col].value_counts(normalize=True)
            for value in range(MAX_VAL[col] + 1):
                row[f"plr_{col}_rate_{value}"] = float(rates.get(value, 0.0))
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def main() -> None:
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    out = Path("artifacts/player_feature_dist.parquet")
    df = build_player_feature_dist(train, test)
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {df.shape}")


if __name__ == "__main__":
    main()
