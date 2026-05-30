from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold


TARGET_ACTION_CLASSES = list(range(19))
TARGET_POINT_CLASSES = list(range(10))
RECENT_COLS = ["strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"]
COUNT_COLS = ["actionId", "pointId", "spinId", "strengthId", "handId", "positionId"]


def find_data_dir() -> Path:
    matches = [p for p in Path.cwd().glob("AI CUP*") if p.is_dir()]
    if not matches:
        raise FileNotFoundError("Could not find AI CUP data directory")
    return matches[0]


def phase_id(target_strike_number: int) -> int:
    if target_strike_number == 2:
        return 0
    if target_strike_number == 3:
        return 1
    return 2


def entropy_from_counts(counts: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    probs = np.array([v / total for v in counts.values() if v > 0], dtype=float)
    return float(-(probs * np.log(probs + 1e-12)).sum())


# --- Displacement / "pressure" proxy (opt-in; Idea 1, real geometry) ---------
# VERIFIED pointId semantics (NCU CSIE data dictionary + empirical check):
#   pointId 1..9 = in-play 3x3 landing grid, defined RELATIVE TO THE RECEIVER's
#     hand. lateral x: forehand{1,4,7}=+1, middle{2,5,8}=0, backhand{3,6,9}=-1;
#     depth y: short{1,2,3}=1, half-long{4,5,6}=2, long{7,8,9}=3.
#   pointId 0 = rally-ENDING stroke (off-table/winner/error); 100% of last
#     strokes, ~never mid-rally -> non-spatial, distance undefined (sentinel).
# Strokes alternate, and the frame is the receiver's hand, so the UPCOMING hitter
# receives the opponent's strokes prefix[-1] & prefix[-3] in the SAME (their own)
# frame => disp_my_* = how far that player was just run between receives. A tree
# cannot synthesise |x1-x2|+|y1-y2| across two last{k} features, so the explicit
# distance is genuinely new signal (the markovp rationale).
def _coord(z: int) -> tuple[float, float] | None:
    """Receiver-frame (x lateral, y depth) for zones 1..9; None for 0/out-of-range."""
    if z < 1 or z > 9:
        return None
    return (1.0 - float((z - 1) % 3), float((z - 1) // 3) + 1.0)


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _manhattan(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def displacement_features(prefix: pd.DataFrame) -> dict:
    """Leakage-free movement proxies from OBSERVED prefix landings only.

    Same-frame distances (same receiver); -1 sentinel when the prefix is too
    short OR an endpoint is the non-spatial zone 0, matching the last{k}_* convention.
    """
    pts = prefix["pointId"].astype(int).tolist()
    n = len(pts)
    feats: dict[str, float | int] = {}

    def disp(i: int, j: int) -> tuple[float, float]:
        if n < max(i, j):
            return (-1.0, -1.0)
        a, b = _coord(pts[-i]), _coord(pts[-j])
        if a is None or b is None:
            return (-1.0, -1.0)
        return (_manhattan(a, b), _euclid(a, b))

    # incoming ball the upcoming hitter must reach = prefix[-1] (their frame).
    inc = _coord(pts[-1])
    feats["disp_incoming_x"] = inc[0] if inc else -2.0  # forehand/middle/backhand
    feats["disp_incoming_y"] = inc[1] if inc else -1.0  # depth short/half/long
    # my run: my two most recent receives = opponent strokes at t-1, t-3.
    feats["disp_my_manh"], feats["disp_my_euclid"] = disp(1, 3)
    # opponent run: their two most recent receives = my strokes at t-2, t-4.
    feats["disp_opp_manh"], feats["disp_opp_euclid"] = disp(2, 4)
    return feats


def add_prefix_features(
    prefix: pd.DataFrame, target_strike_number: int, with_displacement: bool = False
) -> dict:
    prefix = prefix.sort_values("strikeNumber")
    last = prefix.iloc[-1]
    first = prefix.iloc[0]
    row: dict[str, float | int] = {}

    prefix_len = int(len(prefix))
    score_self = int(last["scoreSelf"])
    score_other = int(last["scoreOther"])

    row.update(
        {
            "rally_uid": int(last["rally_uid"]),
            "match": int(last["match"]),
            "sex": int(last["sex"]),
            "numberGame": int(last["numberGame"]),
            "rally_id": int(last["rally_id"]),
            "prefix_len": prefix_len,
            "last_strikeNumber": int(last["strikeNumber"]),
            "target_strikeNumber": int(target_strike_number),
            "phase": phase_id(int(target_strike_number)),
            "scoreSelf": score_self,
            "scoreOther": score_other,
            "score_diff": score_self - score_other,
            "score_sum": score_self + score_other,
            "abs_score_diff": abs(score_self - score_other),
            "is_deuce_like": int(score_self >= 10 and score_other >= 10),
            "is_close_score": int(abs(score_self - score_other) <= 1),
            "target_is_even": int(target_strike_number % 2 == 0),
            "prefix_is_short": int(prefix_len <= 2),
            "prefix_is_long": int(prefix_len >= 6),
            "first_gamePlayerId": int(first["gamePlayerId"]),
            "first_gamePlayerOtherId": int(first["gamePlayerOtherId"]),
            "last_gamePlayerId": int(last["gamePlayerId"]),
            "last_gamePlayerOtherId": int(last["gamePlayerOtherId"]),
            "next_gamePlayerId_inferred": int(last["gamePlayerOtherId"]),
            "next_gamePlayerOtherId_inferred": int(last["gamePlayerId"]),
        }
    )

    # First/second/third observed strokes encode serve and receive-attack phases.
    for pos in range(1, 4):
        if len(prefix) >= pos:
            src = prefix.iloc[pos - 1]
            for col in RECENT_COLS:
                row[f"obs{pos}_{col}"] = int(src[col])
        else:
            for col in RECENT_COLS:
                row[f"obs{pos}_{col}"] = -1

    # Last strokes are often the strongest predictors of the next stroke.
    for back in range(1, 6):
        if len(prefix) >= back:
            src = prefix.iloc[-back]
            for col in RECENT_COLS:
                row[f"last{back}_{col}"] = int(src[col])
        else:
            for col in RECENT_COLS:
                row[f"last{back}_{col}"] = -1

    for col in COUNT_COLS:
        values = prefix[col].astype(int).tolist()
        counts = Counter(values)
        total = len(values)
        row[f"{col}_nunique"] = len(counts)
        row[f"{col}_entropy"] = entropy_from_counts(counts, total)
        max_val = max(values) if values else 0
        # Keep all observed IDs compact enough for this dataset.
        for val in range(max(20, max_val + 1)):
            if col == "pointId" and val > 9:
                break
            if col == "positionId" and val > 3:
                break
            if col == "spinId" and val > 5:
                break
            if col in {"strengthId", "handId"} and val > 3:
                break
            row[f"{col}_cnt_{val}"] = counts.get(val, 0)
            row[f"{col}_rate_{val}"] = counts.get(val, 0) / total

    # Compact transition crosses as categorical integer hashes.
    row["last_action_point"] = int(row["last1_actionId"]) * 100 + int(row["last1_pointId"])
    row["last_action_spin"] = int(row["last1_actionId"]) * 100 + int(row["last1_spinId"])
    row["last_point_strength"] = int(row["last1_pointId"]) * 100 + int(row["last1_strengthId"])
    row["phase_last_action"] = int(row["phase"]) * 100 + int(row["last1_actionId"])
    row["phase_last_point"] = int(row["phase"]) * 100 + int(row["last1_pointId"])

    if len(prefix) >= 2:
        row["last_action_changed"] = int(row["last1_actionId"] != row["last2_actionId"])
        row["last_point_changed"] = int(row["last1_pointId"] != row["last2_pointId"])
    else:
        row["last_action_changed"] = -1
        row["last_point_changed"] = -1

    if with_displacement:
        row.update(displacement_features(prefix))

    return row


def build_prefix_dataset(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, grp in train.groupby("rally_uid", sort=False):
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        if len(grp) < 2:
            continue
        server_target = int(grp["serverGetPoint"].iloc[0])
        for target_idx in range(1, len(grp)):
            target = grp.iloc[target_idx]
            feats = add_prefix_features(grp.iloc[:target_idx], int(target["strikeNumber"]))
            feats["y_actionId"] = int(target["actionId"])
            feats["y_pointId"] = int(target["pointId"])
            feats["y_serverGetPoint"] = server_target
            rows.append(feats)
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"y_actionId", "y_pointId", "y_serverGetPoint", "rally_uid"}
    return [c for c in df.columns if c not in exclude]


def categorical_feature_names(cols: list[str]) -> list[str]:
    continuous_exact = {
        "prefix_len",
        "last_strikeNumber",
        "scoreSelf",
        "scoreOther",
        "score_diff",
        "score_sum",
        "abs_score_diff",
    }
    categorical = []
    for col in cols:
        if col in continuous_exact:
            continue
        if "_cnt_" in col or "_rate_" in col or col.endswith("_entropy") or col.endswith("_nunique"):
            continue
        categorical.append(col)
    return categorical


def class_weights(y: pd.Series, classes: list[int], mode: str) -> dict[int, float] | None:
    if mode == "none":
        return None
    counts = y.value_counts().to_dict()
    n = len(y)
    k = len(classes)
    weights = {}
    for cls in classes:
        if cls not in counts:
            continue
        cnt = max(counts.get(cls, 0), 1)
        base = n / (k * cnt)
        weights[cls] = float(np.sqrt(base) if mode == "sqrt" else base)
    return weights


def fit_multiclass(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    classes: list[int],
    weight_mode: str,
    seed: int,
    n_estimators: int,
    num_leaves: int,
) -> np.ndarray:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=num_leaves,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=seed,
        n_jobs=-1,
        class_weight=class_weights(y_train, classes, weight_mode),
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    raw = model.predict_proba(x_valid)
    aligned = np.zeros((len(x_valid), len(classes)), dtype=float)
    for src_idx, cls in enumerate(model.classes_):
        if int(cls) in classes:
            aligned[:, classes.index(int(cls))] = raw[:, src_idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    missing = row_sum.squeeze() == 0
    if missing.any():
        aligned[missing, :] = 1.0 / len(classes)
        row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / row_sum


def fit_binary(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int,
    n_estimators: int,
    num_leaves: int,
) -> np.ndarray:
    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=num_leaves,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=neg / pos,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model.predict_proba(x_valid)[:, 1]


def run_cv(
    df: pd.DataFrame,
    weight_mode: str,
    n_splits: int,
    n_estimators: int,
    num_leaves: int,
) -> dict:
    feats = feature_columns(df)
    x = df[feats].copy()
    groups = df["match"]

    oof_action = np.zeros((len(df), len(TARGET_ACTION_CLASSES)))
    oof_point = np.zeros((len(df), len(TARGET_POINT_CLASSES)))
    oof_server = np.zeros(len(df))
    fold_scores = []

    splitter = GroupKFold(n_splits=n_splits)
    for fold, (trn_idx, val_idx) in enumerate(splitter.split(x, df["y_actionId"], groups), start=1):
        x_train, x_valid = x.iloc[trn_idx], x.iloc[val_idx]
        ya_train, ya_valid = df["y_actionId"].iloc[trn_idx], df["y_actionId"].iloc[val_idx]
        yp_train, yp_valid = df["y_pointId"].iloc[trn_idx], df["y_pointId"].iloc[val_idx]
        ys_train, ys_valid = (
            df["y_serverGetPoint"].iloc[trn_idx],
            df["y_serverGetPoint"].iloc[val_idx],
        )

        pa = fit_multiclass(
            x_train,
            ya_train,
            x_valid,
            ya_valid,
            TARGET_ACTION_CLASSES,
            weight_mode,
            2026 + fold,
            n_estimators,
            num_leaves,
        )
        pp = fit_multiclass(
            x_train,
            yp_train,
            x_valid,
            yp_valid,
            TARGET_POINT_CLASSES,
            weight_mode,
            3026 + fold,
            n_estimators,
            num_leaves,
        )
        ps = fit_binary(x_train, ys_train, x_valid, ys_valid, 4026 + fold, n_estimators, num_leaves)

        oof_action[val_idx] = pa
        oof_point[val_idx] = pp
        oof_server[val_idx] = ps

        action_pred = pa.argmax(axis=1)
        point_pred = pp.argmax(axis=1)
        action_f1 = f1_score(
            ya_valid, action_pred, labels=TARGET_ACTION_CLASSES, average="macro", zero_division=0
        )
        point_f1 = f1_score(
            yp_valid, point_pred, labels=TARGET_POINT_CLASSES, average="macro", zero_division=0
        )
        server_auc = roc_auc_score(ys_valid, ps)
        overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
        fold_scores.append(
            {
                "fold": fold,
                "rows": int(len(val_idx)),
                "action_macro_f1": float(action_f1),
                "point_macro_f1": float(point_f1),
                "server_auc": float(server_auc),
                "overall": float(overall),
            }
        )
        print(f"fold {fold}: action={action_f1:.5f} point={point_f1:.5f} server={server_auc:.5f} overall={overall:.5f}")

    action_pred = oof_action.argmax(axis=1)
    point_pred = oof_point.argmax(axis=1)
    action_f1 = f1_score(
        df["y_actionId"], action_pred, labels=TARGET_ACTION_CLASSES, average="macro", zero_division=0
    )
    point_f1 = f1_score(
        df["y_pointId"], point_pred, labels=TARGET_POINT_CLASSES, average="macro", zero_division=0
    )
    server_auc = roc_auc_score(df["y_serverGetPoint"], oof_server)
    overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc

    phase_scores = {}
    for phase, grp in df.groupby("phase"):
        idx = grp.index.to_numpy()
        phase_scores[str(int(phase))] = {
            "rows": int(len(idx)),
            "action_macro_f1": float(
                f1_score(
                    df.loc[idx, "y_actionId"],
                    action_pred[idx],
                    labels=TARGET_ACTION_CLASSES,
                    average="macro",
                    zero_division=0,
                )
            ),
            "point_macro_f1": float(
                f1_score(
                    df.loc[idx, "y_pointId"],
                    point_pred[idx],
                    labels=TARGET_POINT_CLASSES,
                    average="macro",
                    zero_division=0,
                )
            ),
            "server_auc": float(roc_auc_score(df.loc[idx, "y_serverGetPoint"], oof_server[idx])),
        }

    return {
        "weight_mode": weight_mode,
        "features": feats,
        "fold_scores": fold_scores,
        "oof": {
            "action_macro_f1": float(action_f1),
            "point_macro_f1": float(point_f1),
            "server_auc": float(server_auc),
            "overall": float(overall),
        },
        "phase_scores": phase_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["none"])
    parser.add_argument("--estimators", type=int, default=180)
    parser.add_argument("--num-leaves", type=int, default=31)
    args = parser.parse_args()

    data_dir = find_data_dir()
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    prefix_path = out_dir / "prefix_train_baseline.parquet"
    if prefix_path.exists():
        df = pd.read_parquet(prefix_path)
        print(f"Loaded {prefix_path}: {df.shape}")
    else:
        df = build_prefix_dataset(train)
        df.to_parquet(prefix_path, index=False)
        print(f"Wrote {prefix_path}: {df.shape}")

    results = {}
    for mode in args.modes:
        print(f"\nRunning CV weight_mode={mode}")
        results[mode] = run_cv(df, mode, args.folds, args.estimators, args.num_leaves)

    best_key = max(results, key=lambda k: results[k]["oof"]["overall"])
    payload = {
        "config": {
            "folds": args.folds,
            "modes": args.modes,
            "estimators": args.estimators,
            "num_leaves": args.num_leaves,
        },
        "best_weight_mode": best_key,
        "results": results,
    }
    out_path = out_dir / "lgbm_baseline_cv.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(payload["results"][best_key]["oof"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
