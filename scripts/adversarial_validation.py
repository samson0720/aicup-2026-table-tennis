"""Adversarial validation (Idea 3 diagnostic): can a classifier tell train from test?

Builds the SAME next-stroke prefix features for train (one-sample-per-rally at the CV
cut) and test (real single cut), labels is_test, and 5-fold LGBM AUC. AUC ~0.5 => no
covariate shift (low private-LB shake-up risk); >0.55 => investigate the top features.
Bookkeeping identifiers (rally_uid/match/rally_id) are excluded — they trivially separate
the two id ranges and are not modelled features. DIAGNOSTIC ONLY: do not blindly prune
top features (e.g. gamePlayerId is both top-adversarial AND our most important signal).
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.make_lgbm_submission import build_test_dataset
from scripts.train_lgbm_baseline import feature_columns

EXCLUDE = {"rally_uid", "match", "rally_id", "y_actionId", "y_pointId", "y_serverGetPoint"}


def main() -> None:
    dd = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(dd / "train.csv")
    test = pd.read_csv(dd / "test_new.csv")
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    # one test-like row per train rally (seed 11 cuts) vs the real test rows
    s11 = splits[splits.seed == 11]
    tr = build_one_sample_per_rally(train, s11)
    te = build_test_dataset(test)

    feats = [c for c in feature_columns(tr) if c in te.columns and c not in EXCLUDE]
    Xtr = tr[feats].apply(pd.to_numeric, errors="coerce").fillna(-1.0)
    Xte = te[feats].apply(pd.to_numeric, errors="coerce").fillna(-1.0)
    X = pd.concat([Xtr, Xte], ignore_index=True)
    y = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    print(f"train rows={len(Xtr)} test rows={len(Xte)} feats={len(feats)}")

    aucs, imps = [], np.zeros(len(feats))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tri, vai in skf.split(X, y):
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                               class_weight="balanced", n_jobs=-1, verbose=-1)
        m.fit(X.iloc[tri], y[tri])
        aucs.append(roc_auc_score(y[vai], m.predict_proba(X.iloc[vai])[:, 1]))
        imps += m.feature_importances_
    auc = float(np.mean(aucs))
    print(f"\n=== adversarial AUC = {auc:.4f} (std {np.std(aucs):.4f}) ===")
    print("  ~0.5 => no covariate shift (robust private LB); >0.55 => shift present")
    order = np.argsort(imps)[::-1][:15]
    print("\ntop 15 train/test-separating features:")
    for i in order:
        print(f"  {feats[i]:32s} {imps[i]:.0f}")


if __name__ == "__main__":
    main()
