"""Post-processing utilities for OOF probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def prior_correct(
    probs: np.ndarray, prior: np.ndarray, beta: float = 1.0, eps: float = 1e-12
) -> np.ndarray:
    """Divide each class probability by ``prior ** beta``, then renormalize.

    ``beta`` is the prior-correction *temperature*. ``beta=1`` (the default,
    backward-compatible) divides by the raw prior, shifting predictions toward a
    uniform-prior posterior — the classic trick to recover macro-F1 when argmax
    under the original prior collapses to majority classes. ``beta=0`` is a no-op
    (raw argmax). Intermediate values partially debias: empirically pointId wants
    ``beta~=0.4`` while actionId is hurt by full correction and prefers ``beta~=0``
    (see scripts/diag_prior_headroom.py). Pick it per target with `select_beta`.
    """
    assert probs.shape[1] == prior.shape[0]
    adjusted = probs / np.clip(prior ** beta, eps, None)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def select_beta(
    probs: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    betas: tuple[float, ...] = tuple(round(b, 2) for b in np.arange(0.0, 1.51, 0.1)),
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Pick the prior-correction temperature ``beta`` by grouped leave-one-out CV.

    For each candidate ``beta`` and each held-out group ``g`` (e.g. a CV seed),
    the prior is estimated from the *other* groups and argmax macro-F1 is scored
    on ``g``. The beta with the best mean held-out macro-F1 wins, so the choice
    cannot peek at the rows it is evaluated on. Returns ``(best_beta, best_mean_f1)``.
    Falls back to a single split if there is only one group.
    """
    from sklearn.metrics import f1_score

    probs = np.asarray(probs)
    y = np.asarray(y)
    groups = np.asarray(groups)
    labels = list(range(n_classes))
    uniq = np.unique(groups)

    def folds():
        if len(uniq) < 2:
            yield np.ones(len(y), bool), np.ones(len(y), bool)
        else:
            for g in uniq:
                yield groups != g, groups == g

    splits = list(folds())
    best_beta, best_score = 1.0, -1.0
    for b in betas:
        fold_scores = []
        for tr, va in splits:
            prior = np.bincount(y[tr], minlength=n_classes).astype(float)
            prior = prior / prior.sum()
            adj = probs[va] / np.clip(prior ** b, eps, None)
            fold_scores.append(f1_score(
                y[va], adj.argmax(1), labels=labels, average="macro", zero_division=0,
            ))
        m = float(np.mean(fold_scores))
        if m > best_score + 1e-12:
            best_score, best_beta = m, float(b)
    return best_beta, best_score


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Argmax of (probs - thresholds[None, :])."""
    return (probs - thresholds[None, :]).argmax(axis=1)


def tune_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    grid: tuple[float, ...] = (-0.10, -0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06, 0.10),
) -> np.ndarray:
    """Per-class additive threshold grid search to maximize macro-F1.

    Greedy two-pass: for each class c, fix others at current thresholds and pick the
    best `v` from `grid`. Repeat once. Stops when no class moves.
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
    for _ in range(2):
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
