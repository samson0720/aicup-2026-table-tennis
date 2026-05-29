"""Structured (action, point) joint base (v4 L4). New signal vs the chain:
the marginalized joint P(point) = sum_a P(point|a) P_hat(a). OOF-safe: the
conditional P(point|a) is fit on each fold's train; P_hat(a) is read from an
existing action OOF base (cat). Evaluated only through the full downstream
pipeline (build_final_perrow), never argmax-only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.oof_loader import read_oof, write_oof
from scripts.predict_test_base import _write_test_parquet
from scripts.train_lgbm_baseline import build_prefix_dataset

N_ACTION, N_POINT = 19, 10
ALPHA = 4.0


def fit_point_given_action(df, n_action=N_ACTION, n_point=N_POINT, alpha=ALPHA):
    a = df["y_actionId"].to_numpy().astype(int)
    p = df["y_pointId"].to_numpy().astype(int)
    glob = np.bincount(p, minlength=n_point).astype(float)
    glob = glob / glob.sum() if glob.sum() else np.full(n_point, 1.0 / n_point)
    cond = np.zeros((n_action, n_point))
    for cls in range(n_action):
        c = np.bincount(p[a == cls], minlength=n_point).astype(float)
        cond[cls] = (c + alpha * glob) / (c.sum() + alpha)
    return cond


def marginalize_point(phat_action, cond):
    out = phat_action @ cond
    return out / out.sum(axis=1, keepdims=True)
