"""Compare a ShuttleNet Track-B pilot against the same-slice ShuttleNet baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.score_oof import attach_labels, score_action, score_point


def _score(model: str, train: pd.DataFrame) -> dict[str, float | int]:
    action = attach_labels(pd.read_parquet(f"artifacts/oof/{model}_action.parquet"), train)
    point = attach_labels(pd.read_parquet(f"artifacts/oof/{model}_point.parquet"), train)
    action_keys = action[["rally_uid", "seed", "fold", "cut_strikeNumber"]]
    point_keys = point[["rally_uid", "seed", "fold", "cut_strikeNumber"]]
    assert action_keys.equals(point_keys), f"{model}: action/point OOF key mismatch"
    return {
        "rows": len(action),
        "action_macro_f1": score_action(action),
        "point_macro_f1": score_point(point),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--baseline", default="shuttle_pilot")
    args = parser.parse_args()

    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    baseline = _score(args.baseline, train)
    candidate = _score(args.candidate, train)
    assert candidate["rows"] == baseline["rows"], "candidate and baseline slices differ"

    report = {
        "baseline_model": args.baseline,
        "candidate_model": args.candidate,
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "action_macro_f1": candidate["action_macro_f1"] - baseline["action_macro_f1"],
            "point_macro_f1": candidate["point_macro_f1"] - baseline["point_macro_f1"],
        },
    }
    out = Path(f"artifacts/{args.candidate}_gate.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
