"""Post-processing utilities for OOF probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def prior_correct(probs: np.ndarray, prior: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Divide each class probability by its train prior, then renormalize.

    Equivalent to shifting predictions toward a uniform-prior posterior.
    Common trick to recover macro-F1 when argmax under the original prior
    collapses to majority classes.
    """
    assert probs.shape[1] == prior.shape[0]
    adjusted = probs / np.clip(prior, eps, None)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Argmax of (probs - thresholds[None, :])."""
    return (probs - thresholds[None, :]).argmax(axis=1)


def tune_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    grid: tuple[float, ...] = (-0.10, -0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06, 0.10),
    max_passes: int = 2,
) -> np.ndarray:
    """Per-class additive threshold grid search to maximize macro-F1.

    Greedy coordinate ascent: for each class c, fix others and pick the best `v`
    from `grid`. Repeat up to `max_passes` times. Stops early when no class moves.
    Defaults reproduce the original (±0.10 grid, 2 passes).
    """
    from sklearn.metrics import f1_score

    thr = np.zeros(n_classes)

    def score(t: np.ndarray) -> float:
        yhat = apply_thresholds(probs, t)
        return f1_score(
            y, yhat, labels=list(range(n_classes)),
            average="macro", zero_division=0,
        )

    best_global = score(thr)
    for _ in range(max_passes):
        moved = False
        for c in range(n_classes):
            best_local = best_global
            best_v = thr[c]
            for v in grid:
                trial = thr.copy()
                trial[c] = v
                s = score(trial)
                if s > best_local + 1e-9:
                    best_local = s
                    best_v = v
            if best_v != thr[c]:
                thr[c] = best_v
                best_global = best_local
                moved = True
        if not moved:
            break
    return thr


def phase_blend_server(
    p_model: np.ndarray,
    p_prior: np.ndarray,
    phase: np.ndarray,
    weights: dict[int, float],
) -> np.ndarray:
    """Blend model probability and prior probability per phase bucket.

    weights[phase] is the share assigned to p_prior; (1 - weights[phase]) goes
    to p_model. Missing phase keys default to 0 (pure model).
    """
    out = p_model.astype(float).copy()
    for ph in np.unique(phase):
        w = float(weights.get(int(ph), 0.0))
        mask = phase == ph
        out[mask] = w * p_prior[mask] + (1.0 - w) * p_model[mask]
    return out


def build_server_pair_prior(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    alpha: float = 20.0,
) -> pd.Series:
    """For each rally in `valid`, return the (server, receiver) historical
    serve-win rate computed from `train` only. Unseen pairs fall back to the
    global rate. Smoothing alpha avoids over-trusting small samples.

    Indexed by rally_uid of `valid`.
    """
    # First stroke of each rally identifies the server (gamePlayerId of strike 1).
    train_first = (
        train.sort_values(["rally_uid", "strikeNumber"])
        .drop_duplicates("rally_uid", keep="first")
    )
    pair_stats = (
        train_first.groupby(["gamePlayerId", "gamePlayerOtherId"])
        .agg(n=("serverGetPoint", "size"), w=("serverGetPoint", "sum"))
        .reset_index()
    )
    global_rate = float(train_first["serverGetPoint"].mean())
    pair_stats["rate"] = (
        (pair_stats["w"] + alpha * global_rate) / (pair_stats["n"] + alpha)
    )
    lookup = pair_stats.set_index(["gamePlayerId", "gamePlayerOtherId"])["rate"]

    valid_first = (
        valid.sort_values(["rally_uid", "strikeNumber"])
        .drop_duplicates("rally_uid", keep="first")
    )
    keys = list(zip(valid_first["gamePlayerId"], valid_first["gamePlayerOtherId"]))
    rates = np.array([float(lookup.get(k, global_rate)) for k in keys])
    return pd.Series(rates, index=valid_first["rally_uid"].to_numpy(), name="server_pair_prior")
