from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def find_data_dir() -> Path:
    matches = [p for p in Path.cwd().glob("AI CUP*") if p.is_dir()]
    if not matches:
        raise FileNotFoundError("Could not find AI CUP data directory")
    return matches[0]


def phase_from_target_strike(strike_number: int) -> str:
    if strike_number == 2:
        return "receive"
    if strike_number == 3:
        return "third_ball"
    return "rally"


def continuity_report(df: pd.DataFrame) -> dict:
    bad_start = 0
    bad_step = 0
    max_gap = 0
    examples: list[dict] = []

    for rally_uid, grp in df.groupby("rally_uid", sort=False):
        strikes = grp.sort_values("strikeNumber")["strikeNumber"].astype(int).tolist()
        if not strikes:
            continue
        starts_ok = strikes[0] == 1
        steps = [b - a for a, b in zip(strikes, strikes[1:])]
        step_ok = all(s == 1 for s in steps)
        if not starts_ok:
            bad_start += 1
        if not step_ok:
            bad_step += 1
            max_gap = max(max_gap, max(steps) if steps else 0)
        if (not starts_ok or not step_ok) and len(examples) < 10:
            examples.append({"rally_uid": int(rally_uid), "strikeNumber": strikes[:20]})

    return {
        "rallies": int(df["rally_uid"].nunique()),
        "bad_start_count": int(bad_start),
        "bad_step_count": int(bad_step),
        "max_gap": int(max_gap),
        "examples": examples,
    }


def prefix_label_distribution(train: pd.DataFrame) -> dict:
    rows: list[dict] = []
    for _, grp in train.groupby("rally_uid", sort=False):
        grp = grp.sort_values("strikeNumber")
        if len(grp) < 2:
            continue
        targets = grp.iloc[1:]
        rows.append(
            pd.DataFrame(
                {
                    "target_strikeNumber": targets["strikeNumber"].astype(int).to_numpy(),
                    "actionId": targets["actionId"].astype(int).to_numpy(),
                    "pointId": targets["pointId"].astype(int).to_numpy(),
                }
            )
        )

    labels = pd.concat(rows, ignore_index=True)
    labels["phase"] = labels["target_strikeNumber"].map(phase_from_target_strike)

    out: dict[str, object] = {
        "prefix_samples": int(len(labels)),
        "target_strikeNumber_counts": {
            str(k): int(v)
            for k, v in labels["target_strikeNumber"].value_counts().sort_index().items()
        },
        "phase_counts": {
            str(k): int(v) for k, v in labels["phase"].value_counts().sort_index().items()
        },
        "action_counts": {
            str(k): int(v) for k, v in labels["actionId"].value_counts().sort_index().items()
        },
        "point_counts": {
            str(k): int(v) for k, v in labels["pointId"].value_counts().sort_index().items()
        },
    }

    by_phase = {}
    for phase, grp in labels.groupby("phase"):
        by_phase[phase] = {
            "rows": int(len(grp)),
            "action_counts": {
                str(k): int(v)
                for k, v in grp["actionId"].value_counts().sort_index().items()
            },
            "point_counts": {
                str(k): int(v)
                for k, v in grp["pointId"].value_counts().sort_index().items()
            },
        }
    out["by_phase"] = by_phase
    return out


def file_summary(path: Path) -> dict:
    df = pd.read_csv(path)
    out: dict[str, object] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    if "rally_uid" in df.columns and "strikeNumber" in df.columns:
        rally_sizes = df.groupby("rally_uid")["strikeNumber"].agg(["count", "max"])
        out["rallies"] = int(len(rally_sizes))
        out["rally_rows"] = {
            "min": int(rally_sizes["count"].min()),
            "median": float(rally_sizes["count"].median()),
            "mean": float(rally_sizes["count"].mean()),
            "max": int(rally_sizes["count"].max()),
        }
        out["strike_continuity"] = continuity_report(df)
        out["prefix_len_counts"] = {
            str(k): int(v)
            for k, v in rally_sizes["count"].value_counts().sort_index().items()
        }
    if "serverGetPoint" in df.columns and "rally_uid" in df.columns and len(df) > 0:
        y = df.groupby("rally_uid")["serverGetPoint"].first()
        out["serverGetPoint_counts"] = {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        }
        out["serverGetPoint_max_nunique_per_rally"] = int(
            df.groupby("rally_uid")["serverGetPoint"].nunique(dropna=False).max()
        )
    return out


def main() -> None:
    data_dir = find_data_dir()
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    files = {
        "train": data_dir / "train.csv",
        "test_new": data_dir / "test_new.csv",
        "old_test": data_dir / "Reference_Only_Old_Test_Data" / "test.csv",
        "sample_submission": data_dir / "sample_submission.csv",
    }

    summary = {name: file_summary(path) for name, path in files.items()}
    train = pd.read_csv(files["train"])
    summary["train_prefix_labels"] = prefix_label_distribution(train)

    new = pd.read_csv(files["test_new"])
    old = pd.read_csv(files["old_test"])
    summary["old_test_overlap"] = {
        "old_rally_subset_of_test_new": bool(
            set(old["rally_uid"]).issubset(set(new["rally_uid"]))
        ),
        "overlap_rallies": int(len(set(old["rally_uid"]) & set(new["rally_uid"]))),
        "old_rallies": int(old["rally_uid"].nunique()),
        "test_new_rallies": int(new["rally_uid"].nunique()),
        "test_new_only_rallies": int(
            new.loc[~new["rally_uid"].isin(old["rally_uid"]), "rally_uid"].nunique()
        ),
    }

    out_path = out_dir / "eda_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(summary["train"]["rally_rows"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["train_prefix_labels"]["phase_counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
