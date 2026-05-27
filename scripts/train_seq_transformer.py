"""Train/smoke-test the Route C sequence Transformer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from scripts.oof_loader import write_oof
from scripts.seq_dataset import RallyPrefixDataset, collate_batch
from scripts.seq_model import RallyTransformer


def _indices_for_fold(ds: RallyPrefixDataset, fold: int, train: bool, limit: int | None = None) -> list[int]:
    idx: list[int] = []
    for i in range(len(ds)):
        item = ds[i]
        ok = int(item["fold"]) != fold if train else int(item["fold"]) == fold
        if ok:
            idx.append(i)
            if limit is not None and len(idx) >= limit:
                break
    return idx


def _class_weights(labels: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=n_classes).astype(float)
    counts = np.clip(counts, 1.0, None)
    weights = np.sqrt(len(labels) / (n_classes * counts))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _labels_for_indices(ds: RallyPrefixDataset, indices: list[int], key: str) -> np.ndarray:
    return np.array([int(ds[i][key]) for i in indices], dtype=np.int64)


def _device(args: argparse.Namespace) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)
    return device


def run_one_fold(
    args: argparse.Namespace,
    train: pd.DataFrame,
    splits: pd.DataFrame,
    seed: int,
    fold: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed * 100 + fold)
    np.random.seed(seed * 100 + fold)

    ds = RallyPrefixDataset(train, splits, seed=seed)
    train_idx = _indices_for_fold(ds, fold, train=True, limit=args.max_train)
    valid_idx = _indices_for_fold(ds, fold, train=False, limit=args.max_valid)
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
    action_loss = nn.CrossEntropyLoss(weight=_class_weights(_labels_for_indices(ds, train_idx, "y_action"), 19, device))
    point_loss = nn.CrossEntropyLoss(weight=_class_weights(_labels_for_indices(ds, train_idx, "y_point"), 10, device))
    server_loss = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            floats = batch["floats"].to(device)
            mask = batch["mask"].to(device)
            ya = batch["y_action"].to(device)
            yp = batch["y_point"].to(device)
            ys = batch["y_server"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                out = model(tokens, floats, mask)
                loss = (
                    0.4 * action_loss(out["action_logits"], ya)
                    + 0.4 * point_loss(out["point_logits"], yp)
                    + 0.2 * server_loss(out["server_logit"], ys)
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach().cpu()) * len(ya)
            n += len(ya)
        print(f"epoch={epoch} train_loss={total / max(n, 1):.5f}", flush=True)

    model.eval()
    rows = []
    with torch.no_grad():
        for batch in valid_loader:
            tokens = batch["tokens"].to(device)
            floats = batch["floats"].to(device)
            mask = batch["mask"].to(device)
            out = model(tokens, floats, mask)
            pa = torch.softmax(out["action_logits"], dim=1).cpu().numpy()
            pp = torch.softmax(out["point_logits"], dim=1).cpu().numpy()
            ps = torch.sigmoid(out["server_logit"]).cpu().numpy()
            rows.append((
                batch["rally_uid"].numpy(),
                batch["fold"].numpy(),
                batch["target_strike"].numpy(),
                pa,
                pp,
                ps,
            ))
    n_valid = sum(len(r[0]) for r in rows)
    print(f"seed={seed} fold={fold} valid_pred_rows={n_valid}", flush=True)

    return {
        "rally_uid": np.concatenate([r[0] for r in rows]),
        "fold": np.concatenate([r[1] for r in rows]),
        "cut": np.concatenate([r[2] for r in rows]),
        "action": np.concatenate([r[3] for r in rows], axis=0),
        "point": np.concatenate([r[4] for r in rows], axis=0),
        "server": np.concatenate([r[5] for r in rows]).reshape(-1, 1),
    }


def _write_seq_oof(parts: list[dict[str, np.ndarray]], model_name: str) -> None:
    rally = np.concatenate([p["rally_uid"] for p in parts])
    fold = np.concatenate([p["fold"] for p in parts])
    cut = np.concatenate([p["cut"] for p in parts])
    seed = np.concatenate([
        np.full(len(p["rally_uid"]), int(p["seed"]), dtype=np.int32) for p in parts
    ])
    for target in ("action", "point", "server"):
        probs = np.concatenate([p[target] for p in parts], axis=0)
        out = write_oof(model_name, target, rally, seed, fold, cut, probs)
        print(f"wrote {out}: rows={len(rally)}", flush=True)


def run(args: argparse.Namespace) -> None:
    device = _device(args)
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    splits = pd.read_parquet("artifacts/cv_splits.parquet")

    seeds = args.seeds
    folds = list(range(5)) if args.fold < 0 else [args.fold]
    parts: list[dict[str, np.ndarray]] = []
    for seed in seeds:
        for fold in folds:
            pred = run_one_fold(args, train, splits, seed, fold, device)
            pred["seed"] = np.array(seed)
            parts.append(pred)
            if args.write_partial:
                _write_seq_oof(parts, args.model_name)

    if args.write_oof:
        _write_seq_oof(parts, args.model_name)
        meta = {
            "model_name": args.model_name,
            "seeds": seeds,
            "folds": folds,
            "rows": int(sum(len(p["rally_uid"]) for p in parts)),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "layers": args.layers,
        }
        Path(f"artifacts/{args.model_name}_run_log.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--fold", type=int, default=0, help="0..4, or -1 for all folds")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-valid", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--model-name", default="seq")
    parser.add_argument("--write-oof", action="store_true")
    parser.add_argument("--write-partial", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
