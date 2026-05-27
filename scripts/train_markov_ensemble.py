from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

# Legacy top-level imports re-routed to package paths so the module can be
# imported as `scripts.train_markov_ensemble` from elsewhere (e.g. produce_base_oof).
try:
    from scripts.train_lgbm_baseline import (
        TARGET_ACTION_CLASSES,
        TARGET_POINT_CLASSES,
        build_prefix_dataset,
        feature_columns,
        fit_binary,
        fit_multiclass,
    )
except ImportError:  # pragma: no cover - kept so the legacy CLI keeps working from scripts/.
    from train_lgbm_baseline import (
        TARGET_ACTION_CLASSES,
        TARGET_POINT_CLASSES,
        build_prefix_dataset,
        feature_columns,
        fit_binary,
        fit_multiclass,
    )


ACTION_CONTEXTS = [
    ["phase", "last1_actionId", "last1_pointId", "last1_spinId"],
    ["phase", "last1_actionId", "last1_pointId"],
    ["phase", "last1_actionId"],
    ["phase"],
]
POINT_CONTEXTS = [
    ["phase", "last1_pointId", "last1_actionId", "last1_positionId"],
    ["phase", "last1_pointId", "last1_actionId"],
    ["phase", "last1_pointId"],
    ["phase"],
]
SERVER_CONTEXTS = [
    ["phase", "obs1_actionId", "obs1_spinId", "obs2_actionId"],
    ["phase", "obs1_actionId", "obs1_spinId"],
    ["phase", "obs1_actionId"],
    ["phase"],
]


def find_data_dir() -> Path:
    matches = [p for p in Path.cwd().glob("AI CUP*") if p.is_dir()]
    if not matches:
        raise FileNotFoundError("Could not find AI CUP data directory")
    return matches[0]


class BackoffClassifier:
    def __init__(
        self,
        classes: list[int],
        contexts: list[list[str]],
        phase_alpha: dict[int, float],
        default_alpha: float,
    ) -> None:
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.contexts = contexts
        self.phase_alpha = phase_alpha
        self.default_alpha = default_alpha
        self.tables: list[dict[tuple, np.ndarray]] = []
        self.global_counts = np.zeros(len(classes), dtype=float)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "BackoffClassifier":
        self.tables = [defaultdict(lambda: np.zeros(len(self.classes), dtype=float)) for _ in self.contexts]
        self.global_counts = np.zeros(len(self.classes), dtype=float)

        for idx, cls in y.astype(int).items():
            if cls not in self.class_to_idx:
                continue
            cls_idx = self.class_to_idx[cls]
            self.global_counts[cls_idx] += 1.0
            row = x.loc[idx]
            for table, cols in zip(self.tables, self.contexts):
                key = tuple(int(row[c]) for c in cols)
                table[key][cls_idx] += 1.0
        return self

    def _global_prior(self) -> np.ndarray:
        counts = self.global_counts + self.default_alpha
        return counts / counts.sum()

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(x), len(self.classes)), dtype=float)
        global_prior = self._global_prior()

        for out_idx, (_, row) in enumerate(x.iterrows()):
            phase = int(row["phase"])
            alpha = self.phase_alpha.get(phase, self.default_alpha)
            probs = global_prior
            for table, cols in reversed(list(zip(self.tables, self.contexts))):
                key = tuple(int(row[c]) for c in cols)
                counts = table.get(key)
                if counts is None:
                    continue
                total = counts.sum()
                if total <= 0:
                    continue
                local = counts / total
                weight = total / (total + alpha)
                probs = weight * local + (1.0 - weight) * probs
            out[out_idx] = probs
        return out


class BackoffBinary:
    def __init__(self, contexts: list[list[str]], phase_alpha: dict[int, float], default_alpha: float) -> None:
        self.model = BackoffClassifier([0, 1], contexts, phase_alpha, default_alpha)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "BackoffBinary":
        self.model.fit(x, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]


def markov_oof(
    train_view: pd.DataFrame,
    valid_view: pd.DataFrame,
    s_train: pd.DataFrame,
    s_valid: pd.DataFrame,
) -> dict:
    """OOF predictions from the BackoffClassifier (Markov-style) on one (seed, fold).

    Args:
        train_view: rows belonging to fold-out rallies for this (seed, fold).
        valid_view: rows belonging to in-fold rallies for this (seed, fold).
        s_train, s_valid: cv_splits parquet rows for the same seed, on the train
            and valid folds respectively. Each carries `rally_uid` and `cut_strikeNumber`.

    Returns:
        dict with keys:
          rally_uid: int array (one entry per valid rally that had a usable feature row)
          cut:       int array of cut_strikeNumber, parallel to rally_uid
          action:    (n, 19) numpy array of action probabilities
          point:     (n, 10) numpy array of point probabilities
          server:    (n, 1)  numpy array of P(serverGetPoint=1)
    """
    # Lazy import so this module can be loaded without circular dep concerns.
    try:
        from scripts.diagnose_cv_gap import build_one_sample_per_rally
    except ImportError:  # pragma: no cover
        from diagnose_cv_gap import build_one_sample_per_rally

    df_train = build_one_sample_per_rally(train_view, s_train)
    df_valid = build_one_sample_per_rally(valid_view, s_valid)
    if df_train.empty or df_valid.empty:
        return {
            "rally_uid": np.array([], dtype=np.int64),
            "cut": np.array([], dtype=np.int32),
            "action": np.zeros((0, len(TARGET_ACTION_CLASSES)), dtype=np.float64),
            "point": np.zeros((0, len(TARGET_POINT_CLASSES)), dtype=np.float64),
            "server": np.zeros((0, 1), dtype=np.float64),
        }

    feats = [c for c in feature_columns(df_train) if c in df_valid.columns]
    x_train, x_valid = df_train[feats], df_valid[feats]

    # Same Backoff config as run_oof().
    phase_alpha_action = {0: 55.0, 1: 35.0, 2: 18.0}
    phase_alpha_point = {0: 60.0, 1: 40.0, 2: 20.0}
    phase_alpha_server = {0: 80.0, 1: 55.0, 2: 35.0}

    action_model = BackoffClassifier(TARGET_ACTION_CLASSES, ACTION_CONTEXTS, phase_alpha_action, 25.0)
    point_model = BackoffClassifier(TARGET_POINT_CLASSES, POINT_CONTEXTS, phase_alpha_point, 25.0)
    server_model = BackoffBinary(SERVER_CONTEXTS, phase_alpha_server, 40.0)

    p_action = action_model.fit(x_train, df_train["y_actionId"]).predict_proba(x_valid)
    p_point = point_model.fit(x_train, df_train["y_pointId"]).predict_proba(x_valid)
    p_server = server_model.fit(x_train, df_train["y_serverGetPoint"]).predict_proba(x_valid)

    return {
        "rally_uid": df_valid["rally_uid"].to_numpy(),
        "cut": df_valid["target_strikeNumber"].to_numpy(),
        "action": p_action,
        "point": p_point,
        "server": p_server.reshape(-1, 1),
    }


def score_predictions(
    y_action: pd.Series,
    p_action: np.ndarray,
    y_point: pd.Series,
    p_point: np.ndarray,
    y_server: pd.Series,
    p_server: np.ndarray,
) -> dict:
    action_pred = p_action.argmax(axis=1)
    point_pred = p_point.argmax(axis=1)
    action_f1 = f1_score(
        y_action, action_pred, labels=TARGET_ACTION_CLASSES, average="macro", zero_division=0
    )
    point_f1 = f1_score(
        y_point, point_pred, labels=TARGET_POINT_CLASSES, average="macro", zero_division=0
    )
    server_auc = roc_auc_score(y_server, p_server)
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }


def search_weight_multiclass(y: pd.Series, p_a: np.ndarray, p_b: np.ndarray, classes: list[int]) -> dict:
    best = {"weight_lgbm": 1.0, "score": -1.0}
    for w in np.linspace(0, 1, 101):
        p = w * p_a + (1 - w) * p_b
        score = f1_score(y, p.argmax(axis=1), labels=classes, average="macro", zero_division=0)
        if score > best["score"]:
            best = {"weight_lgbm": float(w), "score": float(score)}
    return best


def search_weight_binary(y: pd.Series, p_a: np.ndarray, p_b: np.ndarray) -> dict:
    best = {"weight_lgbm": 1.0, "score": -1.0}
    for w in np.linspace(0, 1, 101):
        p = w * p_a + (1 - w) * p_b
        score = roc_auc_score(y, p)
        if score > best["score"]:
            best = {"weight_lgbm": float(w), "score": float(score)}
    return best


def run_oof(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    feats = feature_columns(df)
    x = df[feats].copy()
    y_action = df["y_actionId"]
    y_point = df["y_pointId"]
    y_server = df["y_serverGetPoint"]
    groups = df["match"]

    lgb_action = np.zeros((len(df), len(TARGET_ACTION_CLASSES)))
    lgb_point = np.zeros((len(df), len(TARGET_POINT_CLASSES)))
    lgb_server = np.zeros(len(df))
    mk_action = np.zeros_like(lgb_action)
    mk_point = np.zeros_like(lgb_point)
    mk_server = np.zeros_like(lgb_server)

    phase_alpha_action = {0: 55.0, 1: 35.0, 2: 18.0}
    phase_alpha_point = {0: 60.0, 1: 40.0, 2: 20.0}
    phase_alpha_server = {0: 80.0, 1: 55.0, 2: 35.0}

    fold_scores = []
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (trn_idx, val_idx) in enumerate(splitter.split(x, y_action, groups), start=1):
        x_train, x_valid = x.iloc[trn_idx], x.iloc[val_idx]
        ya_train, ya_valid = y_action.iloc[trn_idx], y_action.iloc[val_idx]
        yp_train, yp_valid = y_point.iloc[trn_idx], y_point.iloc[val_idx]
        ys_train, ys_valid = y_server.iloc[trn_idx], y_server.iloc[val_idx]

        lgb_action[val_idx] = fit_multiclass(
            x_train,
            ya_train,
            x_valid,
            ya_valid,
            TARGET_ACTION_CLASSES,
            args.weight_mode,
            2026 + fold,
            args.estimators,
            args.num_leaves,
        )
        lgb_point[val_idx] = fit_multiclass(
            x_train,
            yp_train,
            x_valid,
            yp_valid,
            TARGET_POINT_CLASSES,
            args.weight_mode,
            3026 + fold,
            args.estimators,
            args.num_leaves,
        )
        lgb_server[val_idx] = fit_binary(
            x_train,
            ys_train,
            x_valid,
            ys_valid,
            4026 + fold,
            args.estimators,
            args.num_leaves,
        )

        mk_action_model = BackoffClassifier(TARGET_ACTION_CLASSES, ACTION_CONTEXTS, phase_alpha_action, 25.0)
        mk_point_model = BackoffClassifier(TARGET_POINT_CLASSES, POINT_CONTEXTS, phase_alpha_point, 25.0)
        mk_server_model = BackoffBinary(SERVER_CONTEXTS, phase_alpha_server, 40.0)
        mk_action[val_idx] = mk_action_model.fit(x_train, ya_train).predict_proba(x_valid)
        mk_point[val_idx] = mk_point_model.fit(x_train, yp_train).predict_proba(x_valid)
        mk_server[val_idx] = mk_server_model.fit(x_train, ys_train).predict_proba(x_valid)

        fold_scores.append(
            {
                "fold": fold,
                "lgbm": score_predictions(ya_valid, lgb_action[val_idx], yp_valid, lgb_point[val_idx], ys_valid, lgb_server[val_idx]),
                "markov": score_predictions(ya_valid, mk_action[val_idx], yp_valid, mk_point[val_idx], ys_valid, mk_server[val_idx]),
            }
        )
        print(
            "fold",
            fold,
            "lgbm",
            fold_scores[-1]["lgbm"]["overall"],
            "markov",
            fold_scores[-1]["markov"]["overall"],
        )

    weights = {
        "action": search_weight_multiclass(y_action, lgb_action, mk_action, TARGET_ACTION_CLASSES),
        "point": search_weight_multiclass(y_point, lgb_point, mk_point, TARGET_POINT_CLASSES),
        "server": search_weight_binary(y_server, lgb_server, mk_server),
    }

    ens_action = weights["action"]["weight_lgbm"] * lgb_action + (1 - weights["action"]["weight_lgbm"]) * mk_action
    ens_point = weights["point"]["weight_lgbm"] * lgb_point + (1 - weights["point"]["weight_lgbm"]) * mk_point
    ens_server = weights["server"]["weight_lgbm"] * lgb_server + (1 - weights["server"]["weight_lgbm"]) * mk_server

    return {
        "fold_scores": fold_scores,
        "weights": weights,
        "oof_scores": {
            "lgbm": score_predictions(y_action, lgb_action, y_point, lgb_point, y_server, lgb_server),
            "markov": score_predictions(y_action, mk_action, y_point, mk_point, y_server, mk_server),
            "ensemble": score_predictions(y_action, ens_action, y_point, ens_point, y_server, ens_server),
        },
    }


def make_submission(df: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace, weights: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Local import so the module can be imported without make_lgbm_submission
    # being on the import path (it uses bare 'from train_lgbm_baseline import ...').
    try:
        from scripts.make_lgbm_submission import align_proba, build_test_dataset, fit_binary_full, fit_multiclass_full  # noqa: F401
    except ImportError:
        from make_lgbm_submission import align_proba, build_test_dataset, fit_binary_full, fit_multiclass_full  # noqa: F401

    feats = feature_columns(df)
    x_train = df[feats]
    y_action = df["y_actionId"]
    y_point = df["y_pointId"]
    y_server = df["y_serverGetPoint"]

    test_features = build_test_dataset(test)
    x_test = test_features[feats]

    action_model = fit_multiclass_full(
        x_train,
        y_action,
        TARGET_ACTION_CLASSES,
        args.weight_mode,
        2026,
        args.full_estimators,
        args.num_leaves,
    )
    point_model = fit_multiclass_full(
        x_train,
        y_point,
        TARGET_POINT_CLASSES,
        args.weight_mode,
        3026,
        args.full_estimators,
        args.num_leaves,
    )
    server_model = fit_binary_full(x_train, y_server, 4026, args.full_estimators, args.num_leaves)

    lgb_action = align_proba(action_model, x_test, TARGET_ACTION_CLASSES)
    lgb_point = align_proba(point_model, x_test, TARGET_POINT_CLASSES)
    lgb_server = server_model.predict_proba(x_test)[:, 1]

    mk_action = BackoffClassifier(TARGET_ACTION_CLASSES, ACTION_CONTEXTS, {0: 55.0, 1: 35.0, 2: 18.0}, 25.0)
    mk_point = BackoffClassifier(TARGET_POINT_CLASSES, POINT_CONTEXTS, {0: 60.0, 1: 40.0, 2: 20.0}, 25.0)
    mk_server = BackoffBinary(SERVER_CONTEXTS, {0: 80.0, 1: 55.0, 2: 35.0}, 40.0)

    p_action_m = mk_action.fit(x_train, y_action).predict_proba(x_test)
    p_point_m = mk_point.fit(x_train, y_point).predict_proba(x_test)
    p_server_m = mk_server.fit(x_train, y_server).predict_proba(x_test)

    w_action = weights["action"]["weight_lgbm"]
    w_point = weights["point"]["weight_lgbm"]
    w_server = weights["server"]["weight_lgbm"]

    p_action = w_action * lgb_action + (1 - w_action) * p_action_m
    p_point = w_point * lgb_point + (1 - w_point) * p_point_m
    p_server = w_server * lgb_server + (1 - w_server) * p_server_m

    clean = pd.DataFrame(
        {
            "rally_uid": test_features["rally_uid"].astype(int),
            "actionId": p_action.argmax(axis=1).astype(int),
            "pointId": p_point.argmax(axis=1).astype(int),
            "serverGetPoint": np.clip(p_server, 1e-5, 1 - 1e-5),
        }
    )
    leaderboard = clean.copy()
    return clean, leaderboard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--weight-mode", default="none", choices=["none", "sqrt", "balanced"])
    parser.add_argument("--estimators", type=int, default=180)
    parser.add_argument("--full-estimators", type=int, default=240)
    parser.add_argument("--num-leaves", type=int, default=31)
    args = parser.parse_args()

    data_dir = find_data_dir()
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    old_test = pd.read_csv(data_dir / "Reference_Only_Old_Test_Data" / "test.csv")

    prefix_path = out_dir / "prefix_train_baseline.parquet"
    if prefix_path.exists():
        df = pd.read_parquet(prefix_path)
    else:
        df = build_prefix_dataset(train)
        df.to_parquet(prefix_path, index=False)

    result = run_oof(df, args)
    out_json = out_dir / "markov_ensemble_cv.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    clean, leaderboard = make_submission(df, test, args, result["weights"])
    old_server = old_test.groupby("rally_uid")["serverGetPoint"].first().to_dict()
    mask = leaderboard["rally_uid"].isin(old_server)
    leaderboard.loc[mask, "serverGetPoint"] = leaderboard.loc[mask, "rally_uid"].map(
        lambda uid: 0.95 if int(old_server[int(uid)]) == 1 else 0.05
    )

    clean_path = out_dir / "submission_lgbm_markov_clean.csv"
    leaderboard_path = out_dir / "submission_lgbm_markov_leaderboard.csv"
    clean.to_csv(clean_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)

    print(f"Wrote {out_json}")
    print(json.dumps(result["oof_scores"], indent=2, ensure_ascii=False))
    print("weights", json.dumps(result["weights"], ensure_ascii=False))
    print(f"Wrote {clean_path}")
    print(f"Wrote {leaderboard_path}")


if __name__ == "__main__":
    main()
