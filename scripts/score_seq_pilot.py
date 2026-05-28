"""Honest pilot comparison on the seed-11 folds-0-2 slice.

Compares the sequence model standalone against lgbm15, then compares a per-row
stack of the existing bases with and without the sequence model. No predictions
are averaged over seeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from scripts.oof_loader import read_oof
from scripts.score_oof import attach_labels
from scripts.seq_eval import honest_scores

SEQ = sys.argv[1] if len(sys.argv) > 1 else "seq_pilot"
SLICE_SEED = 11
SLICE_FOLDS = (0, 1, 2)
EXISTING = ["lgbm15", "lgbm31", "markov", "phase_lgbm"]
CHAIN = {"action": "chain_action", "point": "chain_point", "server": "chain_server"}
KEYS = ["rally_uid", "seed", "fold", "cut_strikeNumber"]
NOISE = 0.00168


def _slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["seed"] == SLICE_SEED) & (df["fold"].isin(SLICE_FOLDS))].copy()


def _labeled(train: pd.DataFrame) -> pd.DataFrame:
    sample = _slice(read_oof("lgbm15", "action"))[KEYS]
    labeled = attach_labels(sample.copy(), train)
    match = train.drop_duplicates("rally_uid").set_index("rally_uid")["match"]
    labeled["match"] = labeled["rally_uid"].map(match)
    return labeled.dropna(
        subset=["match", "actionId", "pointId", "serverGetPoint"]
    ).reset_index(drop=True)


def _probs(model: str, target: str, n_classes: int, keyframe: pd.DataFrame) -> np.ndarray:
    cols = ["p_1"] if target == "server" else [f"p_{i}" for i in range(n_classes)]
    df = _slice(read_oof(model, target))[KEYS + cols]
    merged = keyframe[KEYS].merge(df, on=KEYS, how="left")
    return merged[cols].fillna(0.0).to_numpy()


def _stack(
    bases: dict[str, list[str]],
    labels: pd.DataFrame,
    y_action: np.ndarray,
    y_point: np.ndarray,
    y_server: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    out = {}
    configs = [
        ("action", 19, y_action, "multiclass"),
        ("point", 10, y_point, "multiclass"),
        ("server", 1, y_server, "binary"),
    ]
    for target, n_classes, y, kind in configs:
        x = np.concatenate([_probs(model, target, n_classes, labels) for model in bases[target]], axis=1)
        stacked = np.zeros((len(y), n_classes if kind == "multiclass" else 1))
        for train_idx, valid_idx in GroupKFold(n_splits=5).split(x, y, groups):
            if kind == "multiclass":
                clf = LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=300,
                    C=1.0,
                ).fit(x[train_idx], y[train_idx])
                probs = clf.predict_proba(x[valid_idx])
                for col_idx, cls in enumerate(clf.classes_):
                    stacked[valid_idx, int(cls)] = probs[:, col_idx]
            else:
                clf = LogisticRegression(max_iter=300, C=1.0).fit(x[train_idx], y[train_idx])
                stacked[valid_idx, 0] = clf.predict_proba(x[valid_idx])[:, 1]
        out[target] = stacked
    return honest_scores(
        out["action"],
        out["point"],
        out["server"].ravel(),
        y_action,
        y_point,
        y_server,
        groups,
    )


def main() -> None:
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    labels = _labeled(train)
    y_action = labels["actionId"].astype(int).to_numpy()
    y_point = labels["pointId"].astype(int).to_numpy()
    y_server = labels["serverGetPoint"].astype(int).to_numpy()
    groups = labels["match"].to_numpy()
    print(f"pilot slice rows: {len(labels)} (seed {SLICE_SEED}, folds {SLICE_FOLDS})")

    standalone = {}
    for name in ("lgbm15", SEQ):
        standalone[name] = honest_scores(
            _probs(name, "action", 19, labels),
            _probs(name, "point", 10, labels),
            _probs(name, "server", 1, labels).ravel(),
            y_action,
            y_point,
            y_server,
            groups,
        )
    print("=== standalone honest scores (pilot slice) ===")
    print(json.dumps(standalone, indent=2))

    existing = {target: EXISTING + [CHAIN[target]] for target in ("action", "point", "server")}
    with_seq = {target: EXISTING + [CHAIN[target], SEQ] for target in ("action", "point", "server")}
    existing_scores = _stack(existing, labels, y_action, y_point, y_server, groups)
    seq_scores = _stack(with_seq, labels, y_action, y_point, y_server, groups)
    lift = seq_scores["overall"] - existing_scores["overall"]

    print("=== ensemble honest overall (pilot slice) ===")
    print(f"existing 5 bases      : {existing_scores['overall']:.4f}")
    print(f"existing + {SEQ}: {seq_scores['overall']:.4f}")
    print(f"seq ensemble lift     : {lift:+.4f} (noise floor {NOISE})")

    seq_action = standalone[SEQ]["action_f1"]
    lgbm_action = standalone["lgbm15"]["action_f1"]
    green = (seq_action >= 0.32) or (seq_action >= lgbm_action + 0.03) or (lift > NOISE)
    verdict = "GREEN -> proceed to Phase 2" if green else "YELLOW -> reassess; do NOT burn multi-day GPU"
    print(f"\nVERDICT: {verdict}")
    print(f"  seq action {seq_action:.4f} vs lgbm15 {lgbm_action:.4f}; ensemble lift {lift:+.4f}")


if __name__ == "__main__":
    main()
