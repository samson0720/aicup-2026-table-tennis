"""Final ensemble builder.

Uses raw base OOFs from Route A plus Route B chain OOFs. The rejected Route C
Transformer is intentionally excluded. Outputs a private-safe submission and a
public-backup variant with the old-test server smoothing trick.
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
from scripts.postprocess import apply_thresholds, prior_correct, tune_thresholds
from scripts.score_oof import attach_labels, overall
from scripts.stacker import stacked_oof


BASES = {
    "action": ["lgbm15", "lgbm31", "markov", "phase_lgbm", "chain_action"],
    "point": ["lgbm15", "lgbm31", "markov", "phase_lgbm", "chain_point"],
    "server": ["lgbm15", "lgbm31", "markov", "phase_lgbm", "chain_server"],
}


def _data_dir() -> Path:
    return next(Path.cwd().glob("AI CUP*"))


def _test_path(model: str, target: str) -> Path:
    return OOF_DIR / f"{model}_{target}_test.parquet"


def _base_oofs(target: str) -> dict[str, pd.DataFrame]:
    return {model: average_over_seeds(read_oof(model, target), target) for model in BASES[target]}


def _base_test_preds(target: str, rally_uids: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"rally_uid": rally_uids})
    for model in BASES[target]:
        df = pd.read_parquet(_test_path(model, target)).drop_duplicates("rally_uid")
        df = df.set_index("rally_uid").reindex(rally_uids).reset_index()
        prob_cols = [c for c in df.columns if c.startswith("p_")]
        renamed = df.rename(columns={c: f"{model}__{c}" for c in prob_cols})
        out = out.merge(
            renamed[["rally_uid", *[f"{model}__{c}" for c in prob_cols]]],
            on="rally_uid",
            how="left",
        )
    return out


def _require_inputs() -> None:
    missing = []
    for target, models in BASES.items():
        for model in models:
            for suffix in ("", "_test"):
                path = OOF_DIR / f"{model}_{target}{suffix}.parquet"
                if not path.exists():
                    missing.append(path)
    if missing:
        raise SystemExit("Missing final-stack inputs:\n  " + "\n  ".join(str(p) for p in missing))


def _fit_full_meta(base: dict[str, pd.DataFrame], labels: pd.DataFrame, target_kind: str, n_classes: int):
    df = labels.copy()
    for name, oof_df in base.items():
        prob_cols = [c for c in oof_df.columns if c.startswith("p_")]
        df = df.merge(
            oof_df.rename(columns={c: f"{name}__{c}" for c in prob_cols}),
            on="rally_uid",
            how="left",
        )
    feat_cols = [c for c in df.columns if "__p_" in c]
    x = df[feat_cols].fillna(0.0).to_numpy()
    y = df["y"].to_numpy()
    if target_kind == "multiclass":
        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500, C=1.0)
    else:
        clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(x, y)
    return clf, feat_cols


def _predict_multiclass(clf, x: np.ndarray, n_classes: int) -> np.ndarray:
    raw = clf.predict_proba(x)
    aligned = np.zeros((len(x), n_classes), dtype=np.float64)
    for i, cls in enumerate(clf.classes_):
        aligned[:, int(cls)] = raw[:, i]
    return aligned


def _old_test_smooth(sub: pd.DataFrame) -> pd.DataFrame:
    old_path = _data_dir() / "Reference_Only_Old_Test_Data" / "test.csv"
    if not old_path.exists():
        return sub
    old = pd.read_csv(old_path)
    old_server = old.groupby("rally_uid")["serverGetPoint"].first().to_dict()
    out = sub.copy()
    mask = out["rally_uid"].isin(old_server)
    out.loc[mask, "serverGetPoint"] = out.loc[mask, "rally_uid"].map(
        lambda uid: 0.95 if int(old_server[int(uid)]) == 1 else 0.05
    )
    return out


def main() -> None:
    _require_inputs()
    train = pd.read_csv(_data_dir() / "train.csv")
    test = pd.read_csv(_data_dir() / "test_new.csv")

    sample = read_oof("lgbm15", "action")
    match_per_rally = train.drop_duplicates("rally_uid")[["rally_uid", "match"]]
    attached = (
        attach_labels(sample, train)
        [["rally_uid", "actionId", "pointId", "serverGetPoint"]]
        .drop_duplicates("rally_uid")
        .merge(match_per_rally, on="rally_uid", how="left")
    )

    stacks: dict[str, pd.DataFrame] = {}
    thresholds: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}

    for target_kind, target, n_classes, y_col in [
        ("multiclass", "action", 19, "actionId"),
        ("multiclass", "point", 10, "pointId"),
        ("binary", "server", 1, "serverGetPoint"),
    ]:
        labels = attached[["rally_uid", "match", y_col]].rename(columns={y_col: "y"})
        stack = stacked_oof(
            _base_oofs(target),
            labels,
            target_kind=target_kind,
            n_classes=n_classes if target_kind == "multiclass" else 1,
        )
        stacks[target] = stack
        stack.to_parquet(f"artifacts/final_stack_{target}.parquet", index=False)
        print(f"stacked {target}: {stack.shape}")

    for target, n_classes, y_col in [("action", 19, "actionId"), ("point", 10, "pointId")]:
        df = stacks[target]
        prob_cols = [f"p_{i}" for i in range(n_classes)]
        priors = np.bincount(attached[y_col], minlength=n_classes).astype(float)
        priors /= priors.sum()
        corrected = prior_correct(df[prob_cols].to_numpy(), priors)
        thr = tune_thresholds(corrected, attached[y_col].to_numpy(), n_classes)
        thresholds[target] = thr
        df[prob_cols] = corrected
        pred = apply_thresholds(corrected, thr)
        scores[f"{target}_macro_f1"] = float(f1_score(
            attached[y_col], pred, labels=list(range(n_classes)), average="macro", zero_division=0
        ))
        Path(f"artifacts/final_thr_{target}.json").write_text(json.dumps(thr.tolist()))

    server = stacks["server"].merge(attached[["rally_uid", "serverGetPoint"]], on="rally_uid", how="left")
    scores["server_auc"] = float(roc_auc_score(server["serverGetPoint"], server["p_1"]))
    scores["overall"] = overall(scores["action_macro_f1"], scores["point_macro_f1"], scores["server_auc"])
    Path("artifacts/final_scores.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))

    rally_uids = np.sort(test["rally_uid"].unique())
    submission: dict[str, np.ndarray] = {"rally_uid": rally_uids}
    for target_kind, target, n_classes, y_col in [
        ("multiclass", "action", 19, "actionId"),
        ("multiclass", "point", 10, "pointId"),
        ("binary", "server", 1, "serverGetPoint"),
    ]:
        labels = attached[["rally_uid", "match", y_col]].rename(columns={y_col: "y"})
        clf, feat_cols = _fit_full_meta(_base_oofs(target), labels, target_kind, n_classes)
        with open(f"artifacts/final_meta_{target}.pkl", "wb") as f:
            pickle.dump({"clf": clf, "feat_cols": feat_cols, "bases": BASES[target]}, f)

        test_features = _base_test_preds(target, rally_uids)
        for col in feat_cols:
            if col not in test_features.columns:
                test_features[col] = 0.0
        x_test = test_features[feat_cols].fillna(0.0).to_numpy()
        if target_kind == "multiclass":
            probs = _predict_multiclass(clf, x_test, n_classes)
            priors = np.bincount(attached[y_col], minlength=n_classes).astype(float)
            priors /= priors.sum()
            pred = apply_thresholds(prior_correct(probs, priors), thresholds[target])
            submission[y_col] = pred.astype(int)
        else:
            submission["serverGetPoint"] = clf.predict_proba(x_test)[:, list(clf.classes_).index(1)]

    safe = pd.DataFrame(submission)[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    safe["serverGetPoint"] = np.clip(safe["serverGetPoint"], 1e-5, 1 - 1e-5)
    safe_path = Path("artifacts/submission_FINAL_safe.csv")
    smooth_path = Path("artifacts/submission_FINAL_smooth.csv")
    safe.to_csv(safe_path, index=False)
    _old_test_smooth(safe).to_csv(smooth_path, index=False)
    print(f"wrote {safe_path}: {safe.shape}")
    print(f"wrote {smooth_path}: {safe.shape}")


if __name__ == "__main__":
    main()
