"""Torch dataset for rally-prefix sequence modeling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CATEGORICAL_COLS = (
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
    "gamePlayerId",
    "gamePlayerOtherId",
    "sex",
)
SCORE_FLOAT_COLS = ("scoreSelf", "scoreOther")
MAX_LEN = 24
POSITION_COL_IDX = CATEGORICAL_COLS.index("positionId")


class RallyPrefixDataset(Dataset):
    """One item per CV split row for a single seed.

    Tokens contain only strokes with strikeNumber < cut_strikeNumber. The target
    stroke at cut_strikeNumber is used only for labels.
    """

    def __init__(self, train: pd.DataFrame, splits: pd.DataFrame, seed: int, fold: int | None = None):
        sub = splits[splits["seed"] == seed].copy()
        if fold is not None:
            sub = sub[sub["fold"] == fold]
        self._cuts = dict(zip(sub["rally_uid"].astype(int), sub["cut_strikeNumber"].astype(int)))
        self._folds = dict(zip(sub["rally_uid"].astype(int), sub["fold"].astype(int)))
        train = train[train["rally_uid"].isin(self._cuts)].copy()
        self._by_rally = {
            int(rally_uid): group.sort_values("strikeNumber").reset_index(drop=True)
            for rally_uid, group in train.groupby("rally_uid", sort=False)
        }
        self._rallies = [r for r in self._cuts if r in self._by_rally]

    def __len__(self) -> int:
        return len(self._rallies)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        rally_uid = self._rallies[idx]
        group = self._by_rally[rally_uid]
        cut = int(self._cuts[rally_uid])
        prefix = group[group["strikeNumber"] < cut].tail(MAX_LEN).reset_index(drop=True)
        target = group[group["strikeNumber"] == cut]
        if target.empty:
            raise IndexError(f"missing target stroke for rally_uid={rally_uid}, cut={cut}")
        target_row = target.iloc[0]

        tokens = np.zeros((MAX_LEN, len(CATEGORICAL_COLS)), dtype=np.int64)
        floats = np.zeros((MAX_LEN, len(SCORE_FLOAT_COLS)), dtype=np.float32)
        mask = np.zeros(MAX_LEN, dtype=bool)
        for i, (_, row) in enumerate(prefix.iterrows()):
            for j, col in enumerate(CATEGORICAL_COLS):
                tokens[i, j] = int(row[col])
            for j, col in enumerate(SCORE_FLOAT_COLS):
                floats[i, j] = float(row[col])
            mask[i] = True

        return {
            "tokens": torch.from_numpy(tokens),
            "floats": torch.from_numpy(floats),
            "mask": torch.from_numpy(mask),
            "y_action": torch.tensor(int(target_row["actionId"]), dtype=torch.long),
            "y_point": torch.tensor(int(target_row["pointId"]), dtype=torch.long),
            "y_server": torch.tensor(float(group.iloc[0]["serverGetPoint"]), dtype=torch.float32),
            "rally_uid": int(rally_uid),
            "fold": int(self._folds[rally_uid]),
            "target_strike": cut,
        }


class MirrorPositionActionDataset(Dataset):
    """Append left/right-mirrored copies for action-only supervision.

    positionId values 2 and 3 are the left/right variants. pointId is a
    landing-location label, but the dataset does not publish its mirror map, so
    mirrored copies intentionally contribute only to the action loss.
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return 2 * len(self.base)

    def __getitem__(self, idx: int) -> dict:
        is_mirrored = idx >= len(self.base)
        item = dict(self.base[idx % len(self.base)])
        item["point_loss_weight"] = torch.tensor(
            0.0 if is_mirrored else 1.0, dtype=torch.float32
        )
        if not is_mirrored:
            return item

        tokens = item["tokens"].clone()
        positions = tokens[:, POSITION_COL_IDX]
        tokens[:, POSITION_COL_IDX] = torch.where(
            positions == 2,
            torch.tensor(3, dtype=positions.dtype),
            torch.where(
                positions == 3,
                torch.tensor(2, dtype=positions.dtype),
                positions,
            ),
        )
        item["tokens"] = tokens
        return item


def collate_batch(items: list[dict]) -> dict[str, torch.Tensor]:
    keys = ("tokens", "floats", "mask", "y_action", "y_point", "y_server")
    out = {key: torch.stack([item[key] for item in items]) for key in keys}
    if "point_loss_weight" in items[0]:
        out["point_loss_weight"] = torch.stack([item["point_loss_weight"] for item in items])
    out["rally_uid"] = torch.tensor([int(item["rally_uid"]) for item in items], dtype=torch.long)
    out["fold"] = torch.tensor([int(item["fold"]) for item in items], dtype=torch.long)
    out["target_strike"] = torch.tensor([int(item["target_strike"]) for item in items], dtype=torch.long)
    return out
