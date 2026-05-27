"""Build OOF-safe enriched prefix features for Route B.

Each output row is one (rally_uid, seed, fold) validation sample from
cv_splits.parquet. Target encoders are fit only on the corresponding fold-out
train rows, then applied to that fold's validation rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.feature_ngrams import ngram_features
from scripts.feature_semisupervised import build_player_feature_dist
from scripts.target_encoding import build_player_encoders


def _semi_features(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return build_player_feature_dist(train, test).set_index("gamePlayerId")


def _ngram_frame(valid_view: pd.DataFrame, s_valid: pd.DataFrame) -> pd.DataFrame:
    cut_by_rally = dict(zip(s_valid["rally_uid"], s_valid["cut_strikeNumber"]))
    rows: list[dict] = []
    for rally_uid, group in valid_view.groupby("rally_uid", sort=False):
        cut = cut_by_rally.get(int(rally_uid))
        if cut is None:
            continue
        prefix = group[group["strikeNumber"] < cut]
        rows.append({"rally_uid": int(rally_uid), **ngram_features(prefix)})
    return pd.DataFrame(rows)


def _encoder_train_frame(df_train: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "player": df_train["next_gamePlayerId_inferred"].astype(int),
        "phase": df_train["phase"].astype(int),
        "opponent": df_train["next_gamePlayerOtherId_inferred"].astype(int),
        "y_action": df_train["y_actionId"].astype(int),
        "y_point": df_train["y_pointId"].astype(int),
        "y_server": df_train["y_serverGetPoint"].astype(int),
    })


def _encoder_apply_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "player": df["next_gamePlayerId_inferred"].astype(int),
        "phase": df["phase"].astype(int),
        "opponent": df["next_gamePlayerOtherId_inferred"].astype(int),
    })


def main() -> None:
    data_dir = next(Path.cwd().glob("AI CUP*"))
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_new.csv")
    semi = _semi_features(train, test)

    all_rows: list[pd.DataFrame] = []
    for seed, fold, train_view, valid_view in iter_cv_folds(train, splits):
        s_train = splits[(splits["seed"] == seed) & (splits["fold"] != fold)]
        s_valid = splits[(splits["seed"] == seed) & (splits["fold"] == fold)]

        df_train = build_one_sample_per_rally(train_view, s_train)
        df_valid = build_one_sample_per_rally(valid_view, s_valid)
        if df_train.empty or df_valid.empty:
            continue

        df_valid = df_valid.merge(_ngram_frame(valid_view, s_valid), on="rally_uid", how="left")
        df_valid = df_valid.merge(
            semi.add_prefix("plr_dist_")
            .reset_index()
            .rename(columns={"gamePlayerId": "next_gamePlayerId_inferred"}),
            on="next_gamePlayerId_inferred",
            how="left",
        )

        encoders = build_player_encoders(_encoder_train_frame(df_train), n_action=19, n_point=10)
        te = encoders.transform(_encoder_apply_frame(df_valid))
        te_cols = [f"te_{i}" for i in range(te.shape[1])]
        df_valid = pd.concat(
            [df_valid.reset_index(drop=True), pd.DataFrame(te, columns=te_cols)],
            axis=1,
        )
        df_valid["seed"] = int(seed)
        df_valid["fold"] = int(fold)
        df_valid = df_valid.fillna(0.0)
        all_rows.append(df_valid)
        print(f"seed={seed} fold={fold} rows={len(df_valid)} cols={df_valid.shape[1]}", flush=True)

    out = pd.concat(all_rows, ignore_index=True)
    path = Path("artifacts/prefix_train_v2.parquet")
    out.to_parquet(path, index=False)
    print(f"wrote {path}: {out.shape}", flush=True)


if __name__ == "__main__":
    main()
