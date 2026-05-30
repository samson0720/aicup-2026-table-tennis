"""Pilot gate for the displacement/pressure features (Idea 1).

Standalone argmax macro-F1 of `cat_disp` vs the shipped `cat`, both restricted to
the SAME (seed, fold) slice the candidate was produced on. Floors: action 0.00525,
point 0.00506.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.score_oof import attach_labels, score_action, score_point

KEYS = ["rally_uid", "seed", "fold", "cut_strikeNumber"]


def _slice_to(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows of df whose (seed,fold) appear in ref (the candidate slice)."""
    pairs = set(map(tuple, ref[["seed", "fold"]].drop_duplicates().to_numpy()))
    mask = df.apply(lambda r: (r["seed"], r["fold"]) in pairs, axis=1)
    return df[mask].reset_index(drop=True)


def _score(model: str, train: pd.DataFrame, ref_keys: pd.DataFrame | None):
    a = pd.read_parquet(f"artifacts/oof/{model}_action.parquet")
    p = pd.read_parquet(f"artifacts/oof/{model}_point.parquet")
    if ref_keys is not None:
        a = _slice_to(a, ref_keys)
        p = _slice_to(p, ref_keys)
    a = attach_labels(a, train).sort_values(KEYS).reset_index(drop=True)
    p = attach_labels(p, train).sort_values(KEYS).reset_index(drop=True)
    assert a[KEYS].equals(p[KEYS]), f"{model}: action/point key mismatch"
    return a, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="cat_disp")
    ap.add_argument("--baseline", default="cat")
    args = ap.parse_args()

    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    ca, cp = _score(args.candidate, train, None)
    ba, bp = _score(args.baseline, train, ca)  # slice baseline to candidate's seeds/folds
    assert len(ba) == len(ca), f"slice mismatch: baseline {len(ba)} vs candidate {len(ca)}"

    base = {"rows": len(ba), "action": score_action(ba), "point": score_point(bp)}
    cand = {"rows": len(ca), "action": score_action(ca), "point": score_point(cp)}
    report = {
        "baseline_model": args.baseline,
        "candidate_model": args.candidate,
        "baseline": base,
        "candidate": cand,
        "delta": {"action": cand["action"] - base["action"], "point": cand["point"] - base["point"]},
        "floors": {"action": 0.00525, "point": 0.00506},
    }
    Path(f"artifacts/{args.candidate}_pilot_gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
