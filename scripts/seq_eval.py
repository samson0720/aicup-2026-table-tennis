"""Honest per-row scoring and LR scheduling for the sequence pilot.

Scoring mirrors the per-row ensemble ruler: prior-correct probabilities,
action/point nested threshold tuning for final verdicts, and direct server AUC.
Predictions are never averaged over seeds.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.postprocess import apply_thresholds, prior_correct, tune_thresholds


def warmup_cosine_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    """Return a LambdaLR multiplier with linear warmup and cosine decay."""

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def _prior(y: np.ndarray, n_cls: int) -> np.ndarray:
    counts = np.clip(np.bincount(y.astype(int), minlength=n_cls).astype(float), 1.0, None)
    return counts / counts.sum()


def _auc(y_server: np.ndarray, p_server: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_server, p_server))
    except ValueError:
        return 0.5


def monitor_score(
    action_probs: np.ndarray,
    point_probs: np.ndarray,
    server_probs: np.ndarray,
    y_action: np.ndarray,
    y_point: np.ndarray,
    y_server: np.ndarray,
) -> dict[str, float]:
    """Cheap early-stop monitor: prior-corrected argmax F1 plus server AUC."""
    n_action = action_probs.shape[1]
    n_point = point_probs.shape[1]
    action_pred = prior_correct(action_probs, _prior(y_action, n_action)).argmax(1)
    point_pred = prior_correct(point_probs, _prior(y_point, n_point)).argmax(1)
    action_f1 = f1_score(y_action, action_pred, average="macro", zero_division=0)
    point_f1 = f1_score(y_point, point_pred, average="macro", zero_division=0)
    server_auc = _auc(y_server, server_probs)
    return {
        "action_f1": float(action_f1),
        "point_f1": float(point_f1),
        "server_auc": server_auc,
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }


def _nested_f1(
    probs: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_cls: int,
    n_folds: int = 5,
) -> float:
    corrected = prior_correct(probs, _prior(y, n_cls))
    pred = np.zeros(len(y), dtype=int)
    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < 2:
        thr = tune_thresholds(corrected, y, n_cls)
        pred = apply_thresholds(corrected, thr)
    else:
        splitter = GroupKFold(n_splits=min(n_folds, n_unique_groups))
        for train_idx, valid_idx in splitter.split(corrected, y, groups):
            thr = tune_thresholds(corrected[train_idx], y[train_idx], n_cls)
            pred[valid_idx] = apply_thresholds(corrected[valid_idx], thr)
    return float(f1_score(y, pred, labels=list(range(n_cls)), average="macro", zero_division=0))


def honest_scores(
    action_probs: np.ndarray,
    point_probs: np.ndarray,
    server_probs: np.ndarray,
    y_action: np.ndarray,
    y_point: np.ndarray,
    y_server: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """Final verdict scores with prior correction and nested-threshold F1."""
    action_f1 = _nested_f1(action_probs, y_action, groups, action_probs.shape[1])
    point_f1 = _nested_f1(point_probs, y_point, groups, point_probs.shape[1])
    server_auc = _auc(y_server, server_probs)
    return {
        "action_f1": action_f1,
        "point_f1": point_f1,
        "server_auc": server_auc,
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }
