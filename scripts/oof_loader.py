"""Uniform OOF probability storage and loading.

Schema for every artifacts/oof/<model>_<target>.parquet file:
- rally_uid : int64
- seed      : int32 (one of (11, 22, 33, 44, 55))
- fold      : int32 (one of 0..4)
- cut_strikeNumber : int32 (mirrored from cv_splits for traceability)
- p_<class_id> : float32 for each class id of the target

Targets:
- action : 19 classes -> p_0 .. p_18
- point  : 10 classes -> p_0 .. p_9
- server : 1 class    -> p_1 (probability that serverGetPoint == 1)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_CLASS_COUNTS = {"action": 19, "point": 10, "server": 1}

OOF_DIR = Path("artifacts/oof")


def oof_path(model: str, target: str) -> Path:
    return OOF_DIR / f"{model}_{target}.parquet"


def write_oof(
    model: str,
    target: str,
    rally_uid: np.ndarray,
    seed: np.ndarray,
    fold: np.ndarray,
    cut: np.ndarray,
    probs: np.ndarray,
) -> Path:
    n_class = TARGET_CLASS_COUNTS[target]
    assert probs.ndim == 2 and probs.shape[1] == n_class, f"probs shape {probs.shape}"
    df = pd.DataFrame({
        "rally_uid": rally_uid.astype(np.int64),
        "seed": seed.astype(np.int32),
        "fold": fold.astype(np.int32),
        "cut_strikeNumber": cut.astype(np.int32),
    })
    if target == "server":
        df["p_1"] = probs[:, 0].astype(np.float32)
    else:
        for c in range(n_class):
            df[f"p_{c}"] = probs[:, c].astype(np.float32)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    out = oof_path(model, target)
    df.to_parquet(out, index=False)
    return out


def read_oof(model: str, target: str) -> pd.DataFrame:
    return pd.read_parquet(oof_path(model, target))


def average_over_seeds(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Reduce a 5-seed OOF parquet to one row per rally_uid by mean-of-probabilities."""
    prob_cols = [c for c in df.columns if c.startswith("p_")]
    grouped = df.groupby("rally_uid", as_index=False)[prob_cols].mean()
    return grouped
