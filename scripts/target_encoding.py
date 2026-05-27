"""OOF-safe smoothed target encoding helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class SmoothedEncoder:
    """Smoothed class-frequency encoder with global-prior fallback."""

    keys: Sequence[str]
    n_classes: int
    alpha: float = 20.0
    global_prior: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._stats: pd.DataFrame | None = None
        self._global: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, y_col: str) -> "SmoothedEncoder":
        if len(train) == 0:
            self._stats = pd.DataFrame()
            self._global = (
                self.global_prior.astype(float).copy()
                if self.global_prior is not None
                else np.full(self.n_classes, 1.0 / self.n_classes)
            )
            return self

        if self.global_prior is None:
            counts = np.bincount(train[y_col].astype(int), minlength=self.n_classes)
            total = max(float(counts.sum()), 1.0)
            self._global = counts.astype(float) / total
        else:
            self._global = self.global_prior.astype(float).copy()

        grouped = train.groupby(list(self.keys), dropna=False)[y_col]
        stats = grouped.value_counts().unstack(fill_value=0)
        stats = stats.reindex(columns=range(self.n_classes), fill_value=0).astype(float)
        n = stats.sum(axis=1)
        for cls in range(self.n_classes):
            stats[cls] = (stats[cls] + self.alpha * self._global[cls]) / (n + self.alpha)
        self._stats = stats[list(range(self.n_classes))]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._stats is None or self._global is None:
            raise RuntimeError("call fit() before transform()")
        if self._stats.empty:
            return np.tile(self._global, (len(df), 1)).astype(np.float32)

        if len(self.keys) == 1:
            raw_keys = df[self.keys[0]].tolist()
        else:
            raw_keys = list(df[list(self.keys)].itertuples(index=False, name=None))

        lookup = {k: row.to_numpy(dtype=float) for k, row in self._stats.iterrows()}
        out = np.zeros((len(df), self.n_classes), dtype=np.float32)
        for i, key in enumerate(raw_keys):
            row = lookup.get(key)
            out[i] = row if row is not None else self._global
        return out


@dataclass
class PlayerEncoderBundle:
    encoders: dict[str, SmoothedEncoder]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return np.concatenate([enc.transform(df) for enc in self.encoders.values()], axis=1)


def build_player_encoders(
    train: pd.DataFrame,
    n_action: int = 19,
    n_point: int = 10,
    alpha: float = 20.0,
) -> PlayerEncoderBundle:
    """Fit encoders for player, player-phase, and player-opponent keys."""

    key_specs = [
        ("player", ["player"]),
        ("player_phase", ["player", "phase"]),
        ("player_opponent", ["player", "opponent"]),
    ]
    encoders: dict[str, SmoothedEncoder] = {}
    for name, keys in key_specs:
        encoders[f"{name}__action"] = SmoothedEncoder(keys, n_action, alpha).fit(train, "y_action")
        encoders[f"{name}__point"] = SmoothedEncoder(keys, n_point, alpha).fit(train, "y_point")
        encoders[f"{name}__server"] = SmoothedEncoder(keys, 2, alpha).fit(train, "y_server")
    return PlayerEncoderBundle(encoders)
