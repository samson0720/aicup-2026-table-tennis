"""Per-target stacking meta-learner with inner match-aware CV."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def stacked_oof(
    base_oofs: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    target_kind: Literal["multiclass", "binary"],
    n_classes: int,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Return one OOF row per rally produced by an L2-regularized LR meta-learner.

    `base_oofs` is keyed by model name; each value is a DataFrame with
    `rally_uid` + probability columns `p_0`, `p_1`, ... already averaged
    across the original seeds.

    `labels` must contain `rally_uid`, `match`, `y`. Match groups are used
    by an inner ``GroupKFold`` so the meta-learner never sees rallies from
    the same match in train and validation simultaneously.

    For ``target_kind="multiclass"``, the returned frame has columns
    ``rally_uid`` + ``p_0`` ... ``p_{n_classes-1}``. For
    ``target_kind="binary"`` (``n_classes`` should be ``1``), it has
    ``rally_uid`` + ``p_1`` (probability of the positive class).
    """
    base = labels[["rally_uid", "match", "y"]].copy()
    for name, df in base_oofs.items():
        pc = [c for c in df.columns if c.startswith("p_")]
        rename = {c: f"{name}__{c}" for c in pc}
        renamed = df.rename(columns=rename)[["rally_uid", *rename.values()]]
        base = base.merge(renamed, on="rally_uid", how="left")

    feature_cols = [c for c in base.columns if "__p_" in c]
    X = base[feature_cols].to_numpy()
    y = base["y"].to_numpy()
    groups = base["match"].to_numpy()

    if target_kind == "multiclass":
        oof = np.zeros((len(base), n_classes), dtype=np.float32)
    else:
        oof = np.zeros((len(base), 1), dtype=np.float32)

    kf = GroupKFold(n_splits=n_folds)
    for fold, (tr, va) in enumerate(kf.split(X, y, groups)):
        if target_kind == "multiclass":
            clf = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=200,
                C=1.0,
            )
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[va])
            aligned = np.zeros((len(va), n_classes), dtype=np.float32)
            for i, cls in enumerate(clf.classes_):
                aligned[:, int(cls)] = p[:, i]
            oof[va] = aligned
        else:
            clf = LogisticRegression(max_iter=200, C=1.0)
            clf.fit(X[tr], y[tr])
            oof[va, 0] = clf.predict_proba(X[va])[:, 1]

    if target_kind == "multiclass":
        cols = [f"p_{i}" for i in range(oof.shape[1])]
    else:
        cols = ["p_1"]
    return pd.concat(
        [
            base[["rally_uid"]].reset_index(drop=True),
            pd.DataFrame(oof, columns=cols),
        ],
        axis=1,
    )
