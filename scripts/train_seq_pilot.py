"""Improved Route C pilot trainer.

This keeps the smoke-tested sequence dataset/model path intact and adds
warmup-cosine LR scheduling, gradient clipping, and early stopping on a
validation monitor. It writes per-row validation OOF parquets for the requested
seed/fold slice.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from scripts.oof_loader import OOF_DIR, write_oof
from scripts.seq_dataset import RallyPrefixDataset, collate_batch
from scripts.seq_eval import monitor_score, warmup_cosine_lambda
from scripts.seq_model import RallyTransformer


def _device(cpu: bool) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)
    return device


def _class_weights(labels: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.clip(np.bincount(labels.astype(int), minlength=n_classes).astype(float), 1.0, None)
    weights = np.sqrt(len(labels) / (n_classes * counts))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _split_indices(ds: RallyPrefixDataset, fold: int) -> tuple[list[int], list[int]]:
    folds = [ds._folds[rally_uid] for rally_uid in ds._rallies]
    train_idx = [i for i, item_fold in enumerate(folds) if item_fold != fold]
    valid_idx = [i for i, item_fold in enumerate(folds) if item_fold == fold]
    return train_idx, valid_idx


def _labels(ds: RallyPrefixDataset, indices: list[int], key: str) -> np.ndarray:
    return np.array([int(ds[i][key]) for i in indices], dtype=np.int64)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    cols: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "action",
            "point",
            "server",
            "y_action",
            "y_point",
            "y_server",
            "rally_uid",
            "fold",
            "cut",
        )
    }
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["tokens"].to(device),
                batch["floats"].to(device),
                batch["mask"].to(device),
            )
            cols["action"].append(torch.softmax(out["action_logits"], 1).float().cpu().numpy())
            cols["point"].append(torch.softmax(out["point_logits"], 1).float().cpu().numpy())
            cols["server"].append(torch.sigmoid(out["server_logit"]).float().cpu().numpy())
            cols["y_action"].append(batch["y_action"].numpy())
            cols["y_point"].append(batch["y_point"].numpy())
            cols["y_server"].append(batch["y_server"].numpy())
            cols["rally_uid"].append(batch["rally_uid"].numpy())
            cols["fold"].append(batch["fold"].numpy())
            cols["cut"].append(batch["target_strike"].numpy())
    out = {key: np.concatenate(values, axis=0) for key, values in cols.items()}
    out["server"] = out["server"].reshape(-1, 1)
    return out


def run_one_fold(
    args: argparse.Namespace,
    train: pd.DataFrame,
    splits: pd.DataFrame,
    seed: int,
    fold: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    torch.manual_seed(seed * 100 + fold)
    np.random.seed(seed * 100 + fold)

    ds = RallyPrefixDataset(train, splits, seed=seed)
    train_idx, valid_idx = _split_indices(ds, fold)
    if args.max_train:
        train_idx = train_idx[: args.max_train]
    if args.max_valid:
        valid_idx = valid_idx[: args.max_valid]

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        Subset(ds, valid_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    model = RallyTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
    ).to(device)
    action_loss = nn.CrossEntropyLoss(weight=_class_weights(_labels(ds, train_idx, "y_action"), 19, device))
    point_loss = nn.CrossEntropyLoss(weight=_class_weights(_labels(ds, train_idx, "y_point"), 10, device))
    server_loss = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        warmup_cosine_lambda(int(args.warmup_frac * total_steps), total_steps),
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    best = {"overall": -1.0, "action_f1": 0.0, "point_f1": 0.0, "server_auc": 0.5}
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                out = model(
                    batch["tokens"].to(device),
                    batch["floats"].to(device),
                    batch["mask"].to(device),
                )
                loss = (
                    0.4 * action_loss(out["action_logits"], batch["y_action"].to(device))
                    + 0.4 * point_loss(out["point_logits"], batch["y_point"].to(device))
                    + 0.2 * server_loss(out["server_logit"], batch["y_server"].to(device))
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

        val = _predict(model, valid_loader, device)
        monitor = monitor_score(
            val["action"],
            val["point"],
            val["server"].ravel(),
            val["y_action"],
            val["y_point"],
            val["y_server"],
        )
        print(
            f"seed{seed} fold{fold} epoch{epoch} val_overall={monitor['overall']:.4f} "
            f"a={monitor['action_f1']:.4f} p={monitor['point_f1']:.4f} "
            f"auc={monitor['server_auc']:.4f}",
            flush=True,
        )
        if monitor["overall"] > best["overall"] + 1e-5:
            best = monitor
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch} (best overall {best['overall']:.4f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val = _predict(model, valid_loader, device)
    val["seed"] = np.array(seed)
    best["best_epoch"] = float(epoch - bad_epochs)
    return val, best


def _write_oof(parts: list[dict[str, np.ndarray]], model_name: str) -> None:
    rally = np.concatenate([part["rally_uid"] for part in parts])
    fold = np.concatenate([part["fold"] for part in parts])
    cut = np.concatenate([part["cut"] for part in parts])
    seed = np.concatenate(
        [np.full(len(part["rally_uid"]), int(part["seed"]), dtype=np.int32) for part in parts]
    )
    for target in ("action", "point", "server"):
        probs = np.concatenate([part[target] for part in parts], axis=0)
        out = write_oof(model_name, target, rally, seed, fold, cut, probs)
        print(f"wrote {out}: rows={len(rally)}", flush=True)


def build_test_dataset(test: pd.DataFrame, seed: int) -> RallyPrefixDataset:
    """Synthetic-cut test dataset reusing RallyPrefixDataset unchanged.

    test_new.csv gives each rally its observed strokes (strikeNumber 1..max) and
    no target row. We set cut = max+1 (so the prefix = all observed strokes) and
    append one synthetic target row per rally so __getitem__ does not raise.
    serverGetPoint is absent in test (it is a label), so we add a placeholder;
    all synthetic-row labels are ignored — only the model's predictions are used.
    """
    test = test.copy()
    if "serverGetPoint" not in test.columns:
        test["serverGetPoint"] = 0
    cut_by_rally = test.groupby("rally_uid")["strikeNumber"].max().astype(int) + 1
    synth = (
        test.sort_values("strikeNumber")
        .groupby("rally_uid", as_index=False, sort=False)
        .tail(1)
        .copy()
    )
    synth["strikeNumber"] = synth["rally_uid"].map(cut_by_rally).astype(int)
    synth["actionId"] = 0
    synth["pointId"] = 0
    combined = pd.concat([test, synth], ignore_index=True)
    synth_splits = pd.DataFrame(
        {
            "rally_uid": cut_by_rally.index.astype("int64"),
            "seed": np.int32(seed),
            "fold": np.int32(0),
            "cut_strikeNumber": cut_by_rally.to_numpy().astype("int32"),
        }
    )
    return RallyPrefixDataset(combined, synth_splits, seed=seed)


def _write_test_oof(pred: dict[str, np.ndarray], model_name: str) -> None:
    rally = pred["rally_uid"].astype(np.int64)
    assert len(rally) == len(np.unique(rally)), "duplicate rally_uid in test predictions"
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    for target in ("action", "point", "server"):
        probs = pred[target]
        df = pd.DataFrame({"rally_uid": rally})
        if target == "server":
            df["p_1"] = probs[:, 0].astype(np.float32)
        else:
            for c in range(probs.shape[1]):
                df[f"p_{c}"] = probs[:, c].astype(np.float32)
        out = OOF_DIR / f"{model_name}_{target}_test.parquet"
        df.to_parquet(out, index=False)
        print(f"wrote {out}: rows={len(df)}", flush=True)


def run_predict_test(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
    splits: pd.DataFrame,
    device: torch.device,
) -> None:
    """Train one full-train model (single seed, no held-out fold, fixed epochs)
    and write single-cut test predictions, distribution-matched to the OOF models.
    """
    seed = args.seeds[0]
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = RallyPrefixDataset(train, splits, seed=seed)
    all_idx = list(range(len(ds)))
    train_loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch
    )
    model = RallyTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
    ).to(device)
    action_loss = nn.CrossEntropyLoss(weight=_class_weights(_labels(ds, all_idx, "y_action"), 19, device))
    point_loss = nn.CrossEntropyLoss(weight=_class_weights(_labels(ds, all_idx, "y_point"), 10, device))
    server_loss = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine_lambda(int(args.warmup_frac * total_steps), total_steps)
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    print(
        f"predict-test: full-train seed {seed}, {len(ds)} examples, "
        f"{args.epochs} epochs (no early stop)",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                out = model(
                    batch["tokens"].to(device),
                    batch["floats"].to(device),
                    batch["mask"].to(device),
                )
                loss = (
                    0.4 * action_loss(out["action_logits"], batch["y_action"].to(device))
                    + 0.4 * point_loss(out["point_logits"], batch["y_point"].to(device))
                    + 0.2 * server_loss(out["server_logit"], batch["y_server"].to(device))
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
        print(f"predict-test epoch{epoch} train done", flush=True)

    test_ds = build_test_dataset(test, seed)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch
    )
    pred = _predict(model, test_loader, device)
    _write_test_oof(pred, args.model_name)


def run(args: argparse.Namespace) -> None:
    device = _device(args.cpu)
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    splits = pd.read_parquet("artifacts/cv_splits.parquet")

    if args.predict_test:
        test = pd.read_csv(data_dir / "test_new.csv")
        run_predict_test(args, train, test, splits, device)
        return

    parts: list[dict[str, np.ndarray]] = []
    logs: dict[str, dict[str, float]] = {}
    for seed in args.seeds:
        for fold in args.folds:
            val, best = run_one_fold(args, train, splits, seed, fold, device)
            parts.append(val)
            logs[f"{seed}_{fold}"] = best

    _write_oof(parts, args.model_name)
    Path("artifacts").mkdir(exist_ok=True)
    Path(f"artifacts/{args.model_name}_run_log.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "folds": args.folds,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "layers": args.layers,
                "ffn": args.ffn,
                "dropout": args.dropout,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_frac": args.warmup_frac,
                "clip": args.clip,
                "per_fold_best": logs,
            },
            indent=2,
        )
    )
    print("wrote OOF + run log", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-valid", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--predict-test", action="store_true",
                        help="train one full-train model and write single-cut test predictions")
    parser.add_argument("--model-name", default="seq_pilot")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
