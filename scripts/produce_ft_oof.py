"""FT-Transformer OOF producer (per-row, honest). Pilot via --seeds/--folds.

Standardized tabular prefix features -> FT-Transformer -> multi-task heads,
class-weighted CE, warmup-cosine LR, early stop on combined monitor. GPU.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.cv_splits import iter_cv_folds
from scripts.diagnose_cv_gap import build_one_sample_per_rally
from scripts.ft_transformer import FTTransformer
from scripts.oof_loader import write_oof
from scripts.seq_eval import monitor_score, warmup_cosine_lambda
from scripts.train_lgbm_baseline import class_weights, feature_columns


def _device(cpu: bool) -> torch.device:
    d = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    print(f"device={d}", flush=True)
    if d.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)
    return d


def _cw(y: np.ndarray, n_cls: int, device: torch.device) -> torch.Tensor:
    cw = class_weights(pd.Series(y), list(range(n_cls)), "sqrt") or {}
    return torch.tensor([cw.get(c, 1.0) for c in range(n_cls)], dtype=torch.float32, device=device)


def _standardize(x_tr: np.ndarray, x_va: np.ndarray):
    mu = x_tr.mean(0)
    sd = x_tr.std(0)
    sd[sd == 0] = 1.0
    return (x_tr - mu) / sd, (x_va - mu) / sd


def _infer(model, x, device):
    model.eval()
    with torch.no_grad():
        o = model(x.to(device))
        pa = torch.softmax(o["action_logits"], 1).float().cpu().numpy()
        pp = torch.softmax(o["point_logits"], 1).float().cpu().numpy()
        ps = torch.sigmoid(o["server_logit"]).float().cpu().numpy()
    return pa, pp, ps


def run_fold(args, df_train, df_valid, feats, device):
    x_tr = df_train[feats].fillna(0).to_numpy(np.float32)
    x_va = df_valid[feats].fillna(0).to_numpy(np.float32)
    x_tr, x_va = _standardize(x_tr, x_va)
    ya = df_train["y_actionId"].to_numpy(); yp = df_train["y_pointId"].to_numpy()
    ys = df_train["y_serverGetPoint"].to_numpy().astype(np.float32)
    va_a = df_valid["y_actionId"].to_numpy(); va_p = df_valid["y_pointId"].to_numpy()
    va_s = df_valid["y_serverGetPoint"].to_numpy().astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(ya), torch.from_numpy(yp), torch.from_numpy(ys)),
        batch_size=args.batch_size, shuffle=True,
    )
    va_x = torch.from_numpy(x_va)
    model = FTTransformer(len(feats), d_model=args.d_model, nhead=args.nhead,
                          num_layers=args.layers, dim_feedforward=args.ffn, dropout=args.dropout).to(device)
    al = nn.CrossEntropyLoss(weight=_cw(ya, 19, device))
    pl = nn.CrossEntropyLoss(weight=_cw(yp, 10, device))
    sl = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total = args.epochs * max(1, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine_lambda(int(0.05 * total), total))

    best = {"overall": -1.0}; best_state = None; bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, ab, pb, sb in loader:
            opt.zero_grad(set_to_none=True)
            out = model(xb.to(device))
            loss = (0.4 * al(out["action_logits"], ab.to(device))
                    + 0.4 * pl(out["point_logits"], pb.to(device))
                    + 0.2 * sl(out["server_logit"], sb.to(device)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        pa, pp, ps = _infer(model, va_x, device)
        m = monitor_score(pa, pp, ps, va_a, va_p, va_s)
        print(f"  epoch{epoch} overall={m['overall']:.4f} a={m['action_f1']:.4f} p={m['point_f1']:.4f} auc={m['server_auc']:.4f}", flush=True)
        if m["overall"] > best["overall"] + 1e-5:
            best = m; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pa, pp, ps = _infer(model, va_x, device)
    return pa, pp, ps.reshape(-1, 1)


def run(args) -> None:
    device = _device(args.cpu)
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action", "point", "server")}
    for seed, fold, tv, vv in iter_cv_folds(train, splits):
        if args.seeds and seed not in args.seeds:
            continue
        if args.folds and fold not in args.folds:
            continue
        st = splits[(splits.seed == seed) & (splits.fold != fold)]
        sv = splits[(splits.seed == seed) & (splits.fold == fold)]
        dtr = build_one_sample_per_rally(tv, st); dva = build_one_sample_per_rally(vv, sv)
        if dtr.empty or dva.empty:
            continue
        feats = [c for c in feature_columns(dtr) if c in dva.columns]
        torch.manual_seed(seed * 100 + fold); np.random.seed(seed * 100 + fold)
        pa, pp, ps = run_fold(args, dtr, dva, feats, device)
        rally = dva["rally_uid"].to_numpy(); sid = np.full(len(rally), seed)
        fid = np.full(len(rally), fold); cut = dva["target_strikeNumber"].to_numpy()
        for tgt, p in (("action", pa), ("point", pp), ("server", ps)):
            bag[tgt]["r"].append(rally); bag[tgt]["s"].append(sid)
            bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cut); bag[tgt]["p"].append(p)
        print(f"ft seed={seed} fold={fold} valid_n={len(rally)}", flush=True)
    for tgt in ("action", "point", "server"):
        r = np.concatenate(bag[tgt]["r"]); s = np.concatenate(bag[tgt]["s"])
        f = np.concatenate(bag[tgt]["f"]); c = np.concatenate(bag[tgt]["c"])
        p = np.concatenate(bag[tgt]["p"], axis=0)
        out = write_oof(args.model_name, tgt, r, s, f, c, p)
        print(f"wrote {out}: rows={len(r)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--ffn", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--model-name", default="ft")
    run(p.parse_args())


if __name__ == "__main__":
    main()
