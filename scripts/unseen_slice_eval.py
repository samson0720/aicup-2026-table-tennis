"""Offline private proxy: evaluate base configs on the UNSEEN-PLAYER OOF slice.

The new test is disjoint matches with ~44% players unseen in train. Match-grouped CV averages
over folds where players RECUR, so it can't see that player-conditional / overfit bases
(shuttle, markovp) transfer worse to new players — why honest CV says "keep shuttle" while
public says "drop it". This scores the ENSEMBLE on the OOF rows whose target hitter was UNSEEN
in that fold's train (the new-test stress condition), per action/point base config. The config
that wins on the UNSEEN slice is the best PRIVATE (new-test) bet. No uploads, no retrain.
Server bases contain neither shuttle nor markovp, so server is identical across configs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.postprocess import apply_thresholds, prior_correct, tune_thresholds
from scripts.score_oof import attach_labels
from scripts.oof_loader import read_oof

KEYS = ["rally_uid", "seed", "fold", "cut_strikeNumber"]
BETAS = {"action": 0.6, "point": 0.5}
CORE = ["lgbm15", "lgbm31", "markov", "phase_lgbm", "cat"]  # always-on; +chain_{t}
SERVER_BASES = ["lgbm15", "lgbm31", "markov", "phase_lgbm", "chain_server", "cat"]
# action/point extra bases toggled:
CONFIGS = {
    "full (8-base)":         ["markovp", "shuttle"],
    "no-shuttle":            ["markovp"],
    "no-markovp":            ["shuttle"],
    "no-shuttle,no-markovp": [],
}


def _feat(bases, target, cols):
    base = None; fc = []
    for m in bases:
        df = read_oof(m, target)[KEYS + cols].rename(columns={c: f"{m}__{c}" for c in cols})
        fc += [f"{m}__{c}" for c in cols]
        base = df if base is None else base.merge(df, on=KEYS, how="inner")
    return base, fc


def _prep(bases, target, n_cls, ycol, train, cols):
    frame, fc = _feat(bases, target, cols)
    lab = attach_labels(frame[KEYS].copy(), train)[KEYS + [ycol]]
    frame = frame.merge(lab, on=KEYS, how="left")
    mpr = train.drop_duplicates("rally_uid").set_index("rally_uid")["match"]
    frame["match"] = frame["rally_uid"].map(mpr)
    frame = frame.dropna(subset=[ycol, "match"]).reset_index(drop=True)
    return frame, fc


def _nested_multi(frame, fc, ycol, n_cls, beta):
    X = frame[fc].fillna(0.0).to_numpy(); y = frame[ycol].astype(int).to_numpy()
    g = frame["match"].to_numpy()
    prior = np.bincount(y, minlength=n_cls).astype(float); prior /= prior.sum()
    yh = np.zeros_like(y)
    for tr, va in GroupKFold(5).split(X, y, g):
        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=300, C=1.0).fit(X[tr], y[tr])
        p = np.zeros((len(va), n_cls))
        for i, c in enumerate(clf.classes_):
            p[:, int(c)] = clf.predict_proba(X[va])[:, i]
        # tune thresholds on the held-out-fold prior-corrected probs, apply to same
        ptr = np.zeros((len(tr), n_cls))
        clf2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=300, C=1.0).fit(X[tr], y[tr])
        for i, c in enumerate(clf2.classes_):
            ptr[:, int(c)] = clf2.predict_proba(X[tr])[:, i]
        thr = tune_thresholds(prior_correct(ptr, prior, beta), y[tr], n_cls)
        yh[va] = apply_thresholds(prior_correct(p, prior, beta), thr)
    return y, yh


def main():
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*")) / "train.csv")
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    hit = train[["rally_uid", "strikeNumber", "gamePlayerId"]].rename(
        columns={"strikeNumber": "cut_strikeNumber", "gamePlayerId": "hitter"})
    foldpl = {}
    for s in splits.seed.unique():
        for f in splits.fold.unique():
            tr = splits[(splits.seed == s) & (splits.fold != f)].rally_uid.unique()
            foldpl[(s, f)] = set(train[train.rally_uid.isin(tr)].gamePlayerId.unique())

    def unseen_mask(frame):
        d = frame.merge(hit, on=["rally_uid", "cut_strikeNumber"], how="left")
        return ~np.array([h in foldpl[(s, f)] for h, s, f in zip(d.hitter, d.seed, d.fold)], bool)

    # server (constant across configs)
    sframe, sfc = _prep(SERVER_BASES, "server", 2, "serverGetPoint", train, ["p_1"])
    Xs = sframe[sfc].fillna(0.0).to_numpy(); ys = sframe["serverGetPoint"].astype(int).to_numpy()
    gs = sframe["match"].to_numpy(); ss = np.zeros(len(ys))
    for tr, va in GroupKFold(5).split(Xs, ys, gs):
        ss[va] = LogisticRegression(max_iter=300, C=1.0).fit(Xs[tr], ys[tr]).predict_proba(Xs[va])[:, 1]
    su = unseen_mask(sframe)
    srv_all, srv_un = roc_auc_score(ys, ss), roc_auc_score(ys[su], ss[su])

    print(f"{'config':24s} | {'overall ALL':>11} | {'overall UNSEEN':>14} | action/point UNSEEN")
    rows = {}
    for name, extra in CONFIGS.items():
        out = {}
        for target, n_cls, ycol in [("action", 19, "actionId"), ("point", 10, "pointId")]:
            bases = CORE + [f"chain_{target}"] + extra
            cols = [f"p_{i}" for i in range(n_cls)]
            frame, fc = _prep(bases, target, n_cls, ycol, train, cols)
            um = unseen_mask(frame)
            y, yh = _nested_multi(frame, fc, ycol, n_cls, BETAS[target])
            f_all = f1_score(y, yh, labels=range(n_cls), average="macro", zero_division=0)
            f_un = f1_score(y[um], yh[um], labels=range(n_cls), average="macro", zero_division=0)
            out[target] = (f_all, f_un)
        oa = 0.4 * out["action"][0] + 0.4 * out["point"][0] + 0.2 * srv_all
        ou = 0.4 * out["action"][1] + 0.4 * out["point"][1] + 0.2 * srv_un
        rows[name] = ou
        print(f"{name:24s} | {oa:11.5f} | {ou:14.5f} | {out['action'][1]:.4f}/{out['point'][1]:.4f}")
    best = max(rows, key=rows.get)
    print(f"\nunseen slice = {int(su.sum())}/{len(su)} ({su.mean():.1%}); new test ~44% unseen players")
    print(f"BEST on UNSEEN slice (= best private/new-test bet): {best} ({rows[best]:.5f})")


if __name__ == "__main__":
    main()
