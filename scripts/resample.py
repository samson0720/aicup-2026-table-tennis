"""Random rare-class oversampling for macro-F1 (duplicates real rows only)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def oversample_rare(x: pd.DataFrame, y: pd.Series, seed: int, min_frac: float = 0.3):
    """Duplicate rows of minority classes until each class has at least
    ``min_frac * max_class_count`` rows. Majority classes are untouched.
    Returns (x_resampled, y_resampled) with reset indices."""
    y = y.reset_index(drop=True)
    x = x.reset_index(drop=True)
    counts = y.value_counts()
    floor = int(np.ceil(min_frac * int(counts.max())))
    rng = np.random.default_rng(seed)
    extra_idx: list[np.ndarray] = []
    for cls, cnt in counts.items():
        if cnt >= floor:
            continue
        pool = np.flatnonzero((y == cls).to_numpy())
        extra_idx.append(rng.choice(pool, size=floor - cnt, replace=True))
    if not extra_idx:
        return x, y
    add = np.concatenate(extra_idx)
    keep = np.arange(len(y))
    order = np.concatenate([keep, add])
    return x.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
