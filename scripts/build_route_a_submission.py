"""Route A end-to-end: score base OOFs, stack per target, write OOF + submission.

Prerequisites (all auto-checked; abort with a clear message if missing):
  artifacts/oof/{lgbm15,lgbm31,markov,phase_lgbm}_{action,point,server}.parquet
  artifacts/oof/{lgbm15,lgbm31,markov,phase_lgbm}_{action,point,server}_test.parquet
  artifacts/cv_splits.parquet

Outputs:
  artifacts/base_oof_scores.json       -- per-base CV scores
  artifacts/route_a_stack_{tgt}.parquet -- stacked OOF (one row per rally)
  artifacts/route_a_scores.json        -- per-target stacked OOF score
  artifacts/route_a_thr_{tgt}.json     -- per-class threshold vector after tuning
  artifacts/submission_A_stacked.csv   -- final Route A submission

Per audit F3: player_stats is NOT in the base set. Per audit F4: stacker labels
use drop_duplicates("rally_uid", keep="first") -- arbitrary single seed label
per rally. If lift over best base is < 0.005, revisit by switching to un-averaged
(rally, seed) stacker training.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from scripts.oof_loader import OOF_DIR, average_over_seeds, read_oof
from scripts.postprocess import (
    apply_thresholds,
    build_server_pair_prior,
    phase_blend_server,
    prior_correct,
    tune_thresholds,
)
from scripts.score_oof import (
    attach_labels,
    overall,
    score_action,
    score_point,
    score_server,
    score_model,
)
from scripts.stacker import stacked_oof


MODELS = ["lgbm15", "lgbm31", "markov", "phase_lgbm"]
# OOF rejected the initial phase-prior blend ({0: 0.7, 1: 0.4, 2: 0.0}):
# raw stack server AUC 0.76432 vs blended 0.74678. Keep Route A local-CV driven.
SERVER_BLEND_WEIGHTS = {0: 0.0, 1: 0.0, 2: 0.0}


def _data_dir() -> Path:
    return next(Path.cwd().glob("AI CUP*"))


def _phase_of_target(target_strike: int) -> int:
    if target_strike == 2:
        return 0
    if target_strike == 3:
        return 1
    return 2


def _require(paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        msg = "Missing required artifacts:\n  " + "\n  ".join(str(p) for p in missing)
        raise SystemExit(msg)


def _base_test_preds(target: str, rally_uids_sorted: np.ndarray) -> pd.DataFrame:
    """Stack base test predictions into one (rally_uid, {name}__p_*) DataFrame."""
    df = pd.DataFrame({"rally_uid": rally_uids_sorted})
    for m in MODELS:
        path = OOF_DIR / f"{m}_{target}_test.parquet"
        if not path.exists():
            print(f"[warn] missing {path}; meta-learner will see zeros for this base")
            continue
        sub = pd.read_parquet(path).drop_duplicates("rally_uid")
        sub = sub.set_index("rally_uid").reindex(rally_uids_sorted).reset_index()
        prob_cols = [c for c in sub.columns if c.startswith("p_")]
        sub = sub.rename(columns={c: f"{m}__{c}" for c in prob_cols})
        df = df.merge(sub[["rally_uid", *[f"{m}__{c}" for c in prob_cols]]], on="rally_uid", how="left")
    return df


def main() -> None:
    # ---- Step 0: preconditions ----
    base_oof_paths = [OOF_DIR / f"{m}_{t}.parquet" for m in MODELS for t in ("action", "point", "server")]
    _require(base_oof_paths + [Path("artifacts/cv_splits.parquet")])

    train = pd.read_csv(_data_dir() / "train.csv")
    test = pd.read_csv(_data_dir() / "test_new.csv")

    # ---- Step 1: score each base for the lift comparison ----
    print("=== base OOF scores ===")
    base_scores = {m: score_model(m) for m in MODELS}
    print(json.dumps(base_scores, indent=2, ensure_ascii=False))
    Path("artifacts/base_oof_scores.json").write_text(json.dumps(base_scores, indent=2, ensure_ascii=False))

    # ---- Step 2: per-rally label and match table ----
    # attach_labels joins cut targets but not match, so pull match from train.
    sample = read_oof("lgbm15", "action")
    match_per_rally = train.drop_duplicates("rally_uid")[["rally_uid", "match"]]
    attached = (
        attach_labels(sample, train)
        [["rally_uid", "actionId", "pointId", "serverGetPoint"]]
        .drop_duplicates("rally_uid")
        .merge(match_per_rally, on="rally_uid", how="left")
    )

    # ---- Step 3: per-target stack ----
    stacks: dict[str, pd.DataFrame] = {}
    for target_kind, tgt, n_cls, y_col in [
        ("multiclass", "action", 19, "actionId"),
        ("multiclass", "point",  10, "pointId"),
        ("binary",     "server",  1, "serverGetPoint"),
    ]:
        base = {m: average_over_seeds(read_oof(m, tgt), tgt) for m in MODELS}
        labels = attached[["rally_uid", "match", y_col]].rename(columns={y_col: "y"})
        stack = stacked_oof(
            base, labels,
            target_kind=target_kind,
            n_classes=n_cls if target_kind == "multiclass" else 1,
        )
        stack.to_parquet(f"artifacts/route_a_stack_{tgt}.parquet", index=False)
        stacks[tgt] = stack
        print(f"stacked {tgt}: {stack.shape}")

    # ---- Step 4: prior-correct + threshold tune for action/point ----
    train_priors = {
        "action": np.bincount(attached["actionId"], minlength=19).astype(float),
        "point":  np.bincount(attached["pointId"],  minlength=10).astype(float),
    }
    train_priors["action"] /= train_priors["action"].sum()
    train_priors["point"]  /= train_priors["point"].sum()

    thresholds: dict[str, np.ndarray] = {}
    for tgt, n_cls, y_col in [("action", 19, "actionId"), ("point", 10, "pointId")]:
        df = stacks[tgt]
        prob_cols = [f"p_{i}" for i in range(n_cls)]
        corrected = prior_correct(df[prob_cols].to_numpy(), train_priors[tgt])
        df[prob_cols] = corrected
        merged = df.merge(attached[["rally_uid", y_col]], on="rally_uid", how="left")
        y = merged[y_col].to_numpy()
        thr = tune_thresholds(corrected, y, n_classes=n_cls)
        thresholds[tgt] = thr
        Path(f"artifacts/route_a_thr_{tgt}.json").write_text(json.dumps(thr.tolist()))
        print(f"thresholds {tgt}: {thr}")

    # ---- Step 5: score stacked OOFs (action/point after postproc; server after blend) ----
    # Build per-rally cut info so we can phase-blend server OOF for the diagnostic.
    cut_per_rally = (
        sample.groupby("rally_uid")["cut_strikeNumber"]
        .first()
        .reset_index()
    )

    # Action / point scoring uses prior-corrected probs with tuned thresholds.
    df_a = stacks["action"].merge(attached[["rally_uid", "actionId"]], on="rally_uid", how="left")
    yhat_a = apply_thresholds(df_a[[f"p_{i}" for i in range(19)]].to_numpy(), thresholds["action"])
    f1_a = float(f1_score(df_a["actionId"], yhat_a, labels=list(range(19)), average="macro", zero_division=0))

    df_p = stacks["point"].merge(attached[["rally_uid", "pointId"]], on="rally_uid", how="left")
    yhat_p = apply_thresholds(df_p[[f"p_{i}" for i in range(10)]].to_numpy(), thresholds["point"])
    f1_p = float(f1_score(df_p["pointId"], yhat_p, labels=list(range(10)), average="macro", zero_division=0))

    # Server: build OOF-safe pair prior on TRAIN (since OOF rallies have labels in train).
    # For the OOF diagnostic, use train as both arguments -- this is per-rally smoothing
    # against in-population stats. The TRUE OOF-safe variant would be per-fold, which we
    # skip for the diagnostic; the SUBMISSION step rebuilds the prior from full train on
    # the test set, which is OOF-safe by definition.
    pair_prior_train = build_server_pair_prior(train, train).to_dict()
    df_s = stacks["server"].merge(
        attached[["rally_uid", "serverGetPoint"]], on="rally_uid", how="left"
    )
    df_s = df_s.merge(cut_per_rally, on="rally_uid", how="left")
    df_s["phase"] = df_s["cut_strikeNumber"].map(_phase_of_target)
    df_s["pair_prior"] = df_s["rally_uid"].map(pair_prior_train).fillna(np.mean(list(pair_prior_train.values())))

    p_server_raw = df_s["p_1"].to_numpy()
    p_server_blend = phase_blend_server(
        p_server_raw, df_s["pair_prior"].to_numpy(),
        df_s["phase"].to_numpy(), SERVER_BLEND_WEIGHTS,
    )
    auc_raw = float(roc_auc_score(df_s["serverGetPoint"], p_server_raw))
    auc_blend = float(roc_auc_score(df_s["serverGetPoint"], p_server_blend))
    print(f"server AUC raw stack: {auc_raw:.5f}, after phase blend: {auc_blend:.5f}")

    route_a_scores = {
        "action_macro_f1": f1_a,
        "point_macro_f1": f1_p,
        "server_auc_raw": auc_raw,
        "server_auc_blended": auc_blend,
        "overall_raw":     overall(f1_a, f1_p, auc_raw),
        "overall_blended": overall(f1_a, f1_p, auc_blend),
    }
    Path("artifacts/route_a_scores.json").write_text(json.dumps(route_a_scores, indent=2))
    print("=== route A stacked OOF scores ===")
    print(json.dumps(route_a_scores, indent=2))

    # ---- Step 6: test-time predictions ----
    test_paths = [OOF_DIR / f"{m}_{t}_test.parquet" for m in MODELS for t in ("action", "point", "server")]
    missing_test = [p for p in test_paths if not p.exists()]
    if missing_test:
        print("\nTest-time base predictions not all present; skipping submission build.")
        print("Run `python -m scripts.predict_test_base --model <name>` for each missing base, then rerun.")
        print("Missing:")
        for p in missing_test:
            print(f"  {p}")
        return

    rally_uids = np.sort(test["rally_uid"].unique())
    submission = {"rally_uid": rally_uids.copy()}

    # Refit meta-learner on the full (seed-averaged base x labels) once per target,
    # then predict on the test base preds.
    for target_kind, tgt, n_cls, y_col in [
        ("multiclass", "action", 19, "actionId"),
        ("multiclass", "point",  10, "pointId"),
        ("binary",     "server",  1, "serverGetPoint"),
    ]:
        base = {m: average_over_seeds(read_oof(m, tgt), tgt) for m in MODELS}
        labels = attached[["rally_uid", "match", y_col]].rename(columns={y_col: "y"})

        # Build per-row feature matrix matching the stacker's internal layout.
        df = labels.copy()
        for name, oof_df in base.items():
            prob_cols = [c for c in oof_df.columns if c.startswith("p_")]
            df = df.merge(
                oof_df.rename(columns={c: f"{name}__{c}" for c in prob_cols}),
                on="rally_uid", how="left",
            )
        feat_cols = [c for c in df.columns if "__p_" in c]
        X_full = df[feat_cols].fillna(0.0).to_numpy()
        y_full = df["y"].to_numpy()

        if target_kind == "multiclass":
            clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=400, C=1.0)
        else:
            clf = LogisticRegression(max_iter=400, C=1.0)
        clf.fit(X_full, y_full)

        with open(f"artifacts/route_a_meta_{tgt}.pkl", "wb") as f:
            pickle.dump({"clf": clf, "feat_cols": feat_cols, "models": MODELS}, f)

        # Test-time feature matrix.
        test_features = _base_test_preds(tgt, rally_uids)
        # Order columns identically to feat_cols, fill missing with 0.
        for c in feat_cols:
            if c not in test_features.columns:
                test_features[c] = 0.0
        X_test = test_features[feat_cols].fillna(0.0).to_numpy()

        if target_kind == "multiclass":
            p = clf.predict_proba(X_test)
            aligned = np.zeros((len(rally_uids), n_cls), dtype=np.float64)
            for i, cls in enumerate(clf.classes_):
                aligned[:, int(cls)] = p[:, i]
            # prior_correct + tuned thresholds.
            corrected = prior_correct(aligned, train_priors[tgt])
            thr = thresholds[tgt]
            yhat = apply_thresholds(corrected, thr)
            submission["actionId" if tgt == "action" else "pointId"] = yhat
        else:
            p_test = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
            # Phase-aware blend with full-train pair prior, applied to test rallies.
            pair_prior_test = build_server_pair_prior(train, test).to_dict()
            global_rate = float(train.drop_duplicates("rally_uid")["serverGetPoint"].mean())
            p_prior_arr = np.array([pair_prior_test.get(int(r), global_rate) for r in rally_uids])

            # Determine test rally phase from its observed max strike + 1.
            test_phase = (
                test.groupby("rally_uid")["strikeNumber"].max().add(1)
                .reindex(rally_uids).map(_phase_of_target).to_numpy()
            )
            p_blend = phase_blend_server(p_test, p_prior_arr, test_phase, SERVER_BLEND_WEIGHTS)
            submission["serverGetPoint"] = p_blend

    sub = pd.DataFrame(submission)[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    out = Path("artifacts/submission_A_stacked.csv")
    sub.to_csv(out, index=False)
    print(f"wrote {out}: {sub.shape}")


if __name__ == "__main__":
    main()
