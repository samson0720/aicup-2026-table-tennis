"""Diagnostic (A): per-class F1 / support for each target on OOF predictions.

Read-only. Does NOT change any model, pipeline, or submission. It loads existing
OOF parquets, attaches the cut-target labels, and reports for action / point:
  - per-class support (how many true examples)
  - per-class precision / recall / F1 under plain argmax
  - how many times each class is PREDICTED vs how often it TRULY occurs
so we can see which (rare) classes argmax never emits and how much macro-F1
headroom they represent.

Usage:
  python -m scripts.diag_per_class_f1 [model ...]
Defaults to whichever Route B / Route C OOF parquets exist locally.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

# Reuse the canonical label-join so the diagnostic matches score_oof exactly.
try:
    from scripts.score_oof import attach_labels
except ImportError:  # allow running as a loose script
    from score_oof import attach_labels


TARGETS = {"action": ("actionId", 19), "point": ("pointId", 10)}

# Map a stored OOF parquet stem to (model_label, target). The repo names files
# <model>_<target>.parquet, but Route B uses chain_<target>_<target>.
def discover(models: list[str]) -> list[tuple[str, str, Path]]:
    found = []
    oof = Path("artifacts/oof")
    for tgt in TARGETS:
        for m in models:
            for stem in (f"{m}_{tgt}", f"{m}_{tgt}_{tgt}"):
                p = oof / f"{stem}.parquet"
                if p.exists():
                    found.append((m, tgt, p))
                    break
    return found


def diag_one(model: str, target: str, path: Path, train: pd.DataFrame) -> dict:
    label_col, n_class = TARGETS[target]
    df = pd.read_parquet(path)
    df = attach_labels(df, train)
    df = df[df[label_col].notna()].copy()

    cols = [f"p_{i}" for i in range(n_class)]
    y = df[label_col].astype(int).to_numpy()
    yhat = df[cols].to_numpy().argmax(1)

    macro = f1_score(y, yhat, labels=list(range(n_class)), average="macro", zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y, yhat, labels=list(range(n_class)), zero_division=0
    )
    pred_counts = np.bincount(yhat, minlength=n_class)

    rows = []
    for c in range(n_class):
        rows.append({
            "class": c,
            "true_support": int(sup[c]),
            "true_share": round(float(sup[c]) / len(y), 4),
            "pred_count": int(pred_counts[c]),
            "precision": round(float(prec[c]), 4),
            "recall": round(float(rec[c]), 4),
            "f1": round(float(f1[c]), 4),
        })
    return {"model": model, "target": target, "n": int(len(y)),
            "macro_f1": round(float(macro), 4), "per_class": rows}


def main(models: list[str]) -> None:
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    found = discover(models)
    if not found:
        raise SystemExit(f"No OOF parquets found for models={models} in artifacts/oof/")

    report = []
    for model, target, path in found:
        d = diag_one(model, target, path, train)
        report.append(d)
        print(f"\n=== {model} / {target}  (n={d['n']}, macro_f1={d['macro_f1']}) ===")
        print(f"{'cls':>3} {'true_sup':>8} {'true%':>7} {'pred':>7} "
              f"{'prec':>6} {'recall':>6} {'f1':>6}  flag")
        for r in d["per_class"]:
            flag = ""
            if r["pred_count"] == 0 and r["true_support"] > 0:
                flag = "NEVER-PREDICTED"
            elif r["f1"] == 0 and r["true_support"] > 0:
                flag = "zero-f1"
            print(f"{r['class']:>3} {r['true_support']:>8} {r['true_share']:>7} "
                  f"{r['pred_count']:>7} {r['precision']:>6} {r['recall']:>6} "
                  f"{r['f1']:>6}  {flag}")
        # crude headroom: macro-F1 if every currently-zero-F1 class hit f1=0.20
        zero_classes = [r for r in d["per_class"]
                        if r["f1"] == 0 and r["true_support"] > 0]
        per_class_f1 = [r["f1"] for r in d["per_class"]]
        hypo = per_class_f1.copy()
        for r in zero_classes:
            hypo[r["class"]] = 0.20
        print(f"  -> {len(zero_classes)} classes at f1=0 with real support. "
              f"If each reached f1=0.20, macro_f1 {d['macro_f1']} -> "
              f"{round(float(np.mean(hypo)), 4)}")

    Path("artifacts/diag_per_class_f1.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print("\nwrote artifacts/diag_per_class_f1.json")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or ["chain", "seq"])
