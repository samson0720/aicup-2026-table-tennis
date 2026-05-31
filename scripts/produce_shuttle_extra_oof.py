"""Produce shuttle_extra OOF + test parquets (ShuttleNet + another_data pretraining).

Runs the shipped shuttle config (d256/l4/h4/ffn768, 50ep patience12) but with
--pretrain-transition-epochs 3 --pretrain-extra-data another_data/train.csv so
pretraining sees both fold-train transitions AND all another_data transitions.
Fine-tuning (OOF generation) uses only official fold-train data, same as shipped shuttle.

Writes:
  artifacts/oof/shuttle_extra_{action,point}.parquet
  artifacts/oof/shuttle_extra_{action,point}_test.parquet

Usage:
  conda run -n aicup-tt python -m scripts.produce_shuttle_extra_oof
  conda run -n aicup-tt python -m scripts.produce_shuttle_extra_oof --seeds 11 --folds 0 1 2  # pilot
"""
from __future__ import annotations

import sys
from scripts.train_shuttle import main as shuttle_main


def run_oof(seeds: list[int], folds: list[int]) -> None:
    sys.argv = [
        "train_shuttle",
        "--seeds", *[str(s) for s in seeds],
        "--folds", *[str(f) for f in folds],
        "--epochs", "50",
        "--patience", "12",
        "--batch-size", "256",
        "--d-model", "256",
        "--nhead", "4",
        "--layers", "4",
        "--ffn", "768",
        "--dropout", "0.2",
        "--num-workers", "6",
        "--pretrain-transition-epochs", "3",
        "--pretrain-lr", "3e-4",
        "--pretrain-extra-data", "another_data/train.csv",
        "--model-name", "shuttle_extra",
    ]
    shuttle_main()


def run_predict_test() -> None:
    sys.argv = [
        "train_shuttle",
        "--seeds", "11",
        "--epochs", "50",
        "--patience", "12",
        "--batch-size", "256",
        "--d-model", "256",
        "--nhead", "4",
        "--layers", "4",
        "--ffn", "768",
        "--dropout", "0.2",
        "--num-workers", "6",
        "--pretrain-transition-epochs", "3",
        "--pretrain-lr", "3e-4",
        "--pretrain-extra-data", "another_data/train.csv",
        "--model-name", "shuttle_extra",
        "--predict-test",
    ]
    shuttle_main()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true", help="run seed11 folds 0-2 only")
    p.add_argument("--predict-test", action="store_true")
    args = p.parse_args()

    if args.predict_test:
        run_predict_test()
    elif args.pilot:
        run_oof([11], [0, 1, 2])
    else:
        all_seeds = [11, 22, 33, 44, 55]
        all_folds = [0, 1, 2, 3, 4]
        run_oof(all_seeds, all_folds)
