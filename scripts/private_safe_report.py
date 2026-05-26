from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from test_like_validation import (
    ModelConfig,
    build_test_like_samples,
    evaluate_config,
    summarize_gate,
)
from train_lgbm_baseline import build_prefix_dataset, run_cv


def find_data_dir() -> Path:
    matches = [p for p in Path.cwd().glob("AI CUP*") if p.is_dir()]
    if not matches:
        raise FileNotFoundError("Could not find AI CUP data directory")
    return matches[0]


def load_prefix_train(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    prefix_path = out_dir / "prefix_train_baseline.parquet"
    if prefix_path.exists():
        return pd.read_parquet(prefix_path)
    train = pd.read_csv(data_dir / "train.csv")
    df = build_prefix_dataset(train)
    df.to_parquet(prefix_path, index=False)
    return df


def run_test_like_scale(
    train: pd.DataFrame,
    test_new: pd.DataFrame,
    split_mode: str,
    configs: list[ModelConfig],
    seeds: list[int],
    folds: int,
) -> dict:
    seed_results = []
    for seed in seeds:
        df = build_test_like_samples(train, test_new, seed)
        cfg_results = []
        for config in configs:
            result = evaluate_config(df, config, split_mode, seed, folds)
            cfg_results.append(result)
            print(
                f"{split_mode} seed={seed} {config.name} "
                f"overall={result['oof']['overall']:.6f} "
                f"action={result['oof']['action_macro_f1']:.6f} "
                f"point={result['oof']['point_macro_f1']:.6f} "
                f"server={result['oof']['server_auc']:.6f}"
            )
        seed_results.append(
            {
                "seed": seed,
                "sample_rows": int(len(df)),
                "prefix_len_mean": float(df["prefix_len"].mean()),
                "configs": cfg_results,
            }
        )
    return {
        "seed_results": seed_results,
        "gate_leaves31_vs_leaves15": summarize_gate(seed_results, "leaves31", "leaves15"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--estimators", type=int, default=180)
    parser.add_argument("--weight-mode", default="sqrt", choices=["none", "sqrt", "balanced"])
    args = parser.parse_args()

    data_dir = find_data_dir()
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    test_new = pd.read_csv(data_dir / "test_new.csv")
    prefix_df = load_prefix_train(data_dir, out_dir)

    configs = [
        ModelConfig("leaves31", num_leaves=31, n_estimators=args.estimators, weight_mode=args.weight_mode),
        ModelConfig("leaves15", num_leaves=15, n_estimators=args.estimators, weight_mode=args.weight_mode),
    ]

    report: dict[str, object] = {
        "config": {
            "seeds": args.seeds,
            "folds": args.folds,
            "estimators": args.estimators,
            "weight_mode": args.weight_mode,
        },
        "scales": {},
    }

    print("\n=== all_prefix_match ===")
    all_prefix = {}
    for config in configs:
        result = run_cv(
            prefix_df,
            config.weight_mode,
            args.folds,
            config.n_estimators,
            config.num_leaves,
        )
        all_prefix[config.name] = result
        print(f"all_prefix_match {config.name}: {result['oof']}")
    report["scales"]["all_prefix_match"] = all_prefix

    print("\n=== test_like_rally ===")
    report["scales"]["test_like_rally"] = run_test_like_scale(
        train, test_new, "rally", configs, args.seeds, args.folds
    )

    print("\n=== test_like_match ===")
    report["scales"]["test_like_match"] = run_test_like_scale(
        train, test_new, "match", configs, args.seeds, args.folds
    )

    out_path = out_dir / "private_safe_three_scale_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
