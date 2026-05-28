# Sequence-Model Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the true ceiling of a properly-trained Route C sequence Transformer on action/point, scored on the honest per-row pipeline vs LGBM, to decide (GREEN/YELLOW) whether to commit multi-day GPU to a full 25-fold integration toward a public-smooth score of 0.5.

**Architecture:** Keep the existing `seq_model.py` / `seq_dataset.py` unchanged (they are smoke-tested). Add a new improved trainer (`train_seq_pilot.py`) with cosine LR + warmup, early stopping on a validation combined macro-F1 monitor, and gradient clipping. Add a reusable honest scorer (`seq_eval.py`) and a pilot comparison CLI (`score_seq_pilot.py`). Run on a small representative slice (seed 11 × folds 0–2), score honestly, and emit a GREEN/YELLOW verdict. Phase 2 (full 25-fold + ensemble integration) is gated on GREEN.

**Tech Stack:** Python 3, PyTorch (CUDA on RTX 3090 = device 0), scikit-learn (LogisticRegression meta + metrics), pandas/numpy, pytest. conda env **`aicup-tt`** only.

**Spec:** `docs/superpowers/specs/2026-05-28-aicup-sequence-model-pilot-design.md`

## Critical context (read before starting)

- **Honest ruler only.** The repo's old stack scores were inflated by seed-averaging (see PROGRESS "CRITICAL CORRECTION"). Score everything on the **per-row** `(rally_uid, seed, fold, cut_strikeNumber)` population. NEVER average predictions over seeds before scoring.
- **Baselines on the honest per-row ruler:** lgbm15 overall **0.3027** (action 0.2587, point 0.1730, server AUC 0.6499); current per-row ensemble overall **0.3206**. Noise floor (across-seed std on overall) = **0.00168**.
- **GPU:** RTX 3090 is PyTorch device 0. Always prefix GPU commands with `env CUDA_VISIBLE_DEVICES=0`. `CUDA_VISIBLE_DEVICES=1` is a GTX 1080 — do not use.
- **conda:** run everything via `conda run -n aicup-tt ...`. Never base/system Python, never `pip install --user`.
- **OOF parquet schema** (from `scripts/oof_loader.py`): columns `rally_uid:int64, seed:int32, fold:int32, cut_strikeNumber:int32`, then `p_0..p_{k-1}` (action k=19, point k=10) or single `p_1` (server). 74,975 rows for a full 5-seed run; the pilot writes only seed 11 × folds 0–2.
- **Data:** `next(Path.cwd().glob("AI CUP*"))` is the data dir; `train.csv` (84,707 rows, 14,995 rallies), `test_new.csv` (1,845 rallies). `artifacts/cv_splits.parquet` holds `(rally_uid, seed, fold, cut_strikeNumber)`.
- **Existing helpers to reuse (do not reimplement):**
  - `scripts/postprocess.py`: `prior_correct(probs, prior)`, `tune_thresholds(probs, y, n_classes)`, `apply_thresholds(probs, thr)`.
  - `scripts/score_oof.py`: `attach_labels(df, train)` joins cut-target `actionId/pointId/serverGetPoint` on `(rally_uid, cut_strikeNumber)`.
  - `scripts/oof_loader.py`: `read_oof(model, target)`, `write_oof(model, target, rally_uid, seed, fold, cut, probs)`.
  - `scripts/seq_dataset.py`: `RallyPrefixDataset(train, splits, seed)`, `collate_batch`. The dataset exposes `._rallies` (ordered rally_uid list) and `._folds` (rally_uid→fold).
  - `scripts/seq_model.py`: `RallyTransformer(d_model, nhead, num_layers, dim_feedforward, dropout)`.

---

## PHASE 1 — Pilot (build now)

### Task 1: Honest scorer module `seq_eval.py` (TDD)

**Files:**
- Create: `scripts/seq_eval.py`
- Test: `tests/test_seq_eval.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_seq_eval.py`:

```python
import numpy as np

from scripts.seq_eval import honest_scores, monitor_score, warmup_cosine_lambda


def test_warmup_cosine_lambda_shape():
    fn = warmup_cosine_lambda(warmup_steps=10, total_steps=110)
    assert fn(0) < fn(9)            # ramps up during warmup
    assert abs(fn(10) - 1.0) < 1e-6 # peaks right after warmup
    assert fn(110) < 0.01           # cosine-decays to ~0 at the end


def _perfect_probs(y: np.ndarray, n_cls: int) -> np.ndarray:
    p = np.full((len(y), n_cls), 0.01)
    p[np.arange(len(y)), y] = 0.99
    return p


def test_honest_scores_perfect_predictions():
    rng = np.random.default_rng(0)
    ya = rng.integers(0, 19, 200)
    yp = rng.integers(0, 10, 200)
    ys = rng.integers(0, 2, 200)
    groups = rng.integers(0, 10, 200)
    s = honest_scores(
        _perfect_probs(ya, 19), _perfect_probs(yp, 10),
        ys * 0.99 + (1 - ys) * 0.01, ya, yp, ys, groups,
    )
    assert s["action_f1"] > 0.95
    assert s["point_f1"] > 0.95
    assert s["server_auc"] > 0.99
    assert 0.0 <= s["overall"] <= 1.0


def test_monitor_score_keys_and_value():
    rng = np.random.default_rng(1)
    ya = rng.integers(0, 19, 50)
    yp = rng.integers(0, 10, 50)
    ys = rng.integers(0, 2, 50)
    m = monitor_score(
        _perfect_probs(ya, 19), _perfect_probs(yp, 10),
        ys * 0.9 + 0.05, ya, yp, ys,
    )
    assert set(m) == {"action_f1", "point_f1", "server_auc", "overall"}
    assert m["action_f1"] > 0.9
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n aicup-tt python -m pytest tests/test_seq_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.seq_eval'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/seq_eval.py`:

```python
"""Honest per-row scoring + LR schedule helper for the sequence pilot.

All scoring matches the ensemble's honest pipeline: prior-correct probabilities,
then for action/point either a cheap argmax (monitor) or nested per-class
threshold tuning (final verdict). NEVER averages over seeds.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.postprocess import apply_thresholds, prior_correct, tune_thresholds


def warmup_cosine_lambda(warmup_steps: int, total_steps: int):
    """LambdaLR multiplier: linear warmup to 1.0, then cosine-decay to ~0."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


def _prior(y: np.ndarray, n_cls: int) -> np.ndarray:
    p = np.bincount(y.astype(int), minlength=n_cls).astype(float)
    return p / p.sum()


def _auc(y_server: np.ndarray, p_server: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_server, p_server))
    except ValueError:
        return 0.5


def monitor_score(action_probs, point_probs, server_probs,
                  y_action, y_point, y_server) -> dict:
    """Cheap early-stopping monitor: prior-corrected argmax F1 + server AUC."""
    n_a, n_p = action_probs.shape[1], point_probs.shape[1]
    pa = prior_correct(action_probs, _prior(y_action, n_a)).argmax(1)
    pp = prior_correct(point_probs, _prior(y_point, n_p)).argmax(1)
    af = f1_score(y_action, pa, labels=list(range(n_a)), average="macro", zero_division=0)
    pf = f1_score(y_point, pp, labels=list(range(n_p)), average="macro", zero_division=0)
    auc = _auc(y_server, server_probs)
    return {"action_f1": float(af), "point_f1": float(pf),
            "server_auc": auc, "overall": float(0.4 * af + 0.4 * pf + 0.2 * auc)}


def _nested_f1(probs, y, groups, n_cls, n_folds=5) -> float:
    corrected = prior_correct(probs, _prior(y, n_cls))
    yhat = np.zeros(len(y), dtype=int)
    for tr, va in GroupKFold(n_splits=n_folds).split(corrected, y, groups):
        thr = tune_thresholds(corrected[tr], y[tr], n_cls)
        yhat[va] = apply_thresholds(corrected[va], thr)
    return float(f1_score(y, yhat, labels=list(range(n_cls)), average="macro", zero_division=0))


def honest_scores(action_probs, point_probs, server_probs,
                  y_action, y_point, y_server, groups) -> dict:
    """Final verdict scoring: prior-correct + nested-threshold F1, server AUC."""
    af = _nested_f1(action_probs, y_action, groups, action_probs.shape[1])
    pf = _nested_f1(point_probs, y_point, groups, point_probs.shape[1])
    auc = _auc(y_server, server_probs)
    return {"action_f1": af, "point_f1": pf, "server_auc": auc,
            "overall": float(0.4 * af + 0.4 * pf + 0.2 * auc)}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n aicup-tt python -m pytest tests/test_seq_eval.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/seq_eval.py tests/test_seq_eval.py
git commit -m "feat(seq): honest per-row scorer + warmup-cosine schedule for pilot"
```

---

### Task 2: Improved trainer `train_seq_pilot.py`

**Files:**
- Create: `scripts/train_seq_pilot.py`

This trainer reuses `RallyTransformer` and `RallyPrefixDataset` unchanged. It adds: warmup+cosine LR (per-batch step), gradient clipping, and early stopping on the validation `monitor_score` overall (best checkpoint restored before writing OOF). It writes val-fold OOF in the standard parquet schema.

- [ ] **Step 1: Write the implementation**

Create `scripts/train_seq_pilot.py`:

```python
"""Improved Route C trainer for the pilot: warmup+cosine LR, early stopping,
grad clipping. Writes per-row val OOF for the requested seed/fold slice.

Existing scripts/train_seq_transformer.py is left intact as the smoke reference.
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

from scripts.oof_loader import write_oof
from scripts.seq_dataset import RallyPrefixDataset, collate_batch
from scripts.seq_eval import monitor_score, warmup_cosine_lambda
from scripts.seq_model import RallyTransformer


def _device(cpu: bool) -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    print(f"device={dev}", flush=True)
    if dev.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)
    return dev


def _class_weights(labels: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.clip(np.bincount(labels.astype(int), minlength=n_classes).astype(float), 1.0, None)
    return torch.tensor(np.sqrt(len(labels) / (n_classes * counts)), dtype=torch.float32, device=device)


def _split_indices(ds: RallyPrefixDataset, fold: int) -> tuple[list[int], list[int]]:
    """Index split by fold using the dataset's cheap (rally_uid -> fold) map."""
    folds = [ds._folds[r] for r in ds._rallies]
    train_idx = [i for i, f in enumerate(folds) if f != fold]
    valid_idx = [i for i, f in enumerate(folds) if f == fold]
    return train_idx, valid_idx


def _labels(ds: RallyPrefixDataset, indices: list[int], key: str) -> np.ndarray:
    return np.array([int(ds[i][key]) for i in indices], dtype=np.int64)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    cols: dict[str, list] = {k: [] for k in
                             ("action", "point", "server", "y_action", "y_point",
                              "y_server", "rally_uid", "fold", "cut")}
    with torch.no_grad():
        for b in loader:
            out = model(b["tokens"].to(device), b["floats"].to(device), b["mask"].to(device))
            cols["action"].append(torch.softmax(out["action_logits"], 1).float().cpu().numpy())
            cols["point"].append(torch.softmax(out["point_logits"], 1).float().cpu().numpy())
            cols["server"].append(torch.sigmoid(out["server_logit"]).float().cpu().numpy())
            cols["y_action"].append(b["y_action"].numpy())
            cols["y_point"].append(b["y_point"].numpy())
            cols["y_server"].append(b["y_server"].numpy())
            cols["rally_uid"].append(b["rally_uid"].numpy())
            cols["fold"].append(b["fold"].numpy())
            cols["cut"].append(b["target_strike"].numpy())
    out = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    out["server"] = out["server"].reshape(-1, 1)
    return out


def run_one_fold(args, train, splits, seed, fold, device) -> tuple[dict, dict]:
    torch.manual_seed(seed * 100 + fold)
    np.random.seed(seed * 100 + fold)

    ds = RallyPrefixDataset(train, splits, seed=seed)
    train_idx, valid_idx = _split_indices(ds, fold)
    if args.max_train:
        train_idx = train_idx[:args.max_train]
    if args.max_valid:
        valid_idx = valid_idx[:args.max_valid]
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(Subset(ds, valid_idx), batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_batch)

    model = RallyTransformer(d_model=args.d_model, nhead=args.nhead, num_layers=args.layers,
                             dim_feedforward=args.ffn, dropout=args.dropout).to(device)
    ya = _labels(ds, train_idx, "y_action")
    yp = _labels(ds, train_idx, "y_point")
    action_loss = nn.CrossEntropyLoss(weight=_class_weights(ya, 19, device))
    point_loss = nn.CrossEntropyLoss(weight=_class_weights(yp, 10, device))
    server_loss = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine_lambda(int(args.warmup_frac * total_steps), total_steps))
    scaler = GradScaler(enabled=device.type == "cuda")

    best = {"overall": -1.0}
    best_state = None
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for b in train_loader:
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                out = model(b["tokens"].to(device), b["floats"].to(device), b["mask"].to(device))
                loss = (0.4 * action_loss(out["action_logits"], b["y_action"].to(device))
                        + 0.4 * point_loss(out["point_logits"], b["y_point"].to(device))
                        + 0.2 * server_loss(out["server_logit"], b["y_server"].to(device)))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

        val = _predict(model, valid_loader, device)
        m = monitor_score(val["action"], val["point"], val["server"].ravel(),
                          val["y_action"], val["y_point"], val["y_server"])
        print(f"seed{seed} fold{fold} epoch{epoch} val_overall={m['overall']:.4f} "
              f"a={m['action_f1']:.4f} p={m['point_f1']:.4f} auc={m['server_auc']:.4f}", flush=True)
        if m["overall"] > best["overall"] + 1e-5:
            best = m
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch} (best overall {best['overall']:.4f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val = _predict(model, valid_loader, device)
    val["seed"] = seed
    return val, best


def _write_oof(parts: list[dict], model_name: str) -> None:
    rally = np.concatenate([p["rally_uid"] for p in parts])
    fold = np.concatenate([p["fold"] for p in parts])
    cut = np.concatenate([p["cut"] for p in parts])
    seed = np.concatenate([np.full(len(p["rally_uid"]), int(p["seed"]), dtype=np.int32) for p in parts])
    for target in ("action", "point", "server"):
        probs = np.concatenate([p[target] for p in parts], axis=0)
        out = write_oof(model_name, target, rally, seed, fold, cut, probs)
        print(f"wrote {out}: rows={len(rally)}", flush=True)


def run(args) -> None:
    device = _device(args.cpu)
    data_dir = next(Path.cwd().glob("AI CUP*"))
    train = pd.read_csv(data_dir / "train.csv")
    splits = pd.read_parquet("artifacts/cv_splits.parquet")

    parts: list[dict] = []
    logs: dict[str, dict] = {}
    for seed in args.seeds:
        for fold in args.folds:
            val, best = run_one_fold(args, train, splits, seed, fold, device)
            parts.append(val)
            logs[f"{seed}_{fold}"] = best
    _write_oof(parts, args.model_name)
    Path(f"artifacts/{args.model_name}_run_log.json").write_text(json.dumps(
        {"seeds": args.seeds, "folds": args.folds, "epochs": args.epochs,
         "patience": args.patience, "d_model": args.d_model, "layers": args.layers,
         "dropout": args.dropout, "lr": args.lr, "per_fold_best": logs}, indent=2))
    print("wrote OOF + run log", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[11])
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ffn", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--max-train", type=int)
    p.add_argument("--max-valid", type=int)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--model-name", default="seq_pilot")
    run(p.parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify with a fast CPU smoke run**

Run:
```bash
conda run -n aicup-tt python -u -m scripts.train_seq_pilot --cpu --seeds 11 --folds 0 \
  --epochs 2 --patience 5 --d-model 64 --layers 1 --ffn 128 --batch-size 64 \
  --max-train 256 --max-valid 128 --model-name seq_pilot_smoke
```
Expected: prints `device=cpu`, two `epoch` lines with `val_overall=...`, then
`wrote artifacts/oof/seq_pilot_smoke_action.parquet: rows=128` (and point/server),
then `wrote OOF + run log`. Completes in under ~2 minutes.

- [ ] **Step 3: Verify the smoke OOF parquet schema**

Run:
```bash
conda run -n aicup-tt python -c "import pandas as pd; d=pd.read_parquet('artifacts/oof/seq_pilot_smoke_action.parquet'); print(d.shape, list(d.columns)[:6], d.seed.unique(), sorted(d.fold.unique()))"
```
Expected: shape `(128, 23)`, columns start `['rally_uid','seed','fold','cut_strikeNumber','p_0','p_1']`, `seed.unique()=[11]`, `fold.unique()=[0]`.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_seq_pilot.py
git commit -m "feat(seq): improved pilot trainer (warmup-cosine, early stop, grad clip)"
```

---

### Task 3: Run the GPU base pilot (seed 11 × folds 0–2)

**Files:** none created — produces `artifacts/oof/seq_pilot_{action,point,server}.parquet`.

- [ ] **Step 1: Launch the base pilot on the 3090**

Run (this takes roughly 30–90 min; run in the background and watch the log):
```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt python -u -m scripts.train_seq_pilot \
  --seeds 11 --folds 0 1 2 --epochs 80 --patience 12 \
  --d-model 192 --layers 4 --nhead 4 --ffn 512 --dropout 0.2 \
  --batch-size 256 --lr 3e-4 --warmup-frac 0.05 --clip 1.0 \
  --model-name seq_pilot
```
Expected: prints `device=cuda`, `NVIDIA GeForce RTX 3090`, rising `val_overall`
per epoch, an early-stop line, then three `wrote artifacts/oof/seq_pilot_*` lines
with `rows≈9000` (seed 11 folds 0–2 ≈ 60% of ~14,995 rallies).

- [ ] **Step 2: Sanity-check the run log**

Run:
```bash
conda run -n aicup-tt python -c "import json; print(json.dumps(json.load(open('artifacts/seq_pilot_run_log.json'))['per_fold_best'], indent=2))"
```
Expected: three folds, each with `overall`, `action_f1`, `point_f1`, `server_auc`.
These are MONITOR (argmax) scores — the honest verdict comes in Task 4.

- [ ] **Step 3: Commit the OOF + run log**

```bash
git add -f artifacts/oof/seq_pilot_action.parquet artifacts/oof/seq_pilot_point.parquet artifacts/oof/seq_pilot_server.parquet
git add artifacts/seq_pilot_run_log.json
git commit -m "feat(seq): base pilot OOF on seed 11 folds 0-2"
```
Note: `artifacts/oof/*.parquet` is gitignored, so `-f` is required (matches the repo's existing `git add -f` convention for OOF parquets).

---

### Task 4: Pilot comparison CLI `score_seq_pilot.py` + verdict

**Files:**
- Create: `scripts/score_seq_pilot.py`

- [ ] **Step 1: Write the implementation**

Create `scripts/score_seq_pilot.py`:

```python
"""Honest pilot comparison on the seed-11 folds-0-2 slice.

Compares the sequence model standalone vs lgbm15, and a per-row stack of the
existing 5 bases with vs without the sequence model. Prints a GREEN/YELLOW
verdict. Pass the seq OOF model name as argv[1] (default 'seq_pilot').
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from scripts.oof_loader import read_oof
from scripts.score_oof import attach_labels
from scripts.seq_eval import honest_scores

SEQ = sys.argv[1] if len(sys.argv) > 1 else "seq_pilot"
SLICE_SEED = 11
SLICE_FOLDS = (0, 1, 2)
EXISTING = ["lgbm15", "lgbm31", "markov", "phase_lgbm"]
CHAIN = {"action": "chain_action", "point": "chain_point", "server": "chain_server"}
KEYS = ["rally_uid", "seed", "fold", "cut_strikeNumber"]
NOISE = 0.00168


def _slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["seed"] == SLICE_SEED) & (df["fold"].isin(SLICE_FOLDS))].copy()


def _labeled(train: pd.DataFrame) -> pd.DataFrame:
    sample = _slice(read_oof("lgbm15", "action"))[KEYS]
    lab = attach_labels(sample.copy(), train)
    match = train.drop_duplicates("rally_uid").set_index("rally_uid")["match"]
    lab["match"] = lab["rally_uid"].map(match)
    return lab.dropna(subset=["match", "actionId", "pointId", "serverGetPoint"]).reset_index(drop=True)


def _probs(model: str, target: str, n_cls: int, keyframe: pd.DataFrame) -> np.ndarray:
    cols = ["p_1"] if target == "server" else [f"p_{i}" for i in range(n_cls)]
    df = _slice(read_oof(model, target))[KEYS + cols]
    merged = keyframe.merge(df, on=KEYS, how="left")
    return merged[cols].fillna(0.0).to_numpy()


def _stack(bases: dict, lab, ya, yp, ys, g) -> dict:
    out = {}
    for kind, target, n_cls, y in [("mc", "action", 19, ya), ("mc", "point", 10, yp), ("bin", "server", 1, ys)]:
        X = np.concatenate([_probs(m, target, n_cls, lab) for m in bases[target]], axis=1)
        stk = np.zeros((len(y), n_cls if kind == "mc" else 1))
        for tr, va in GroupKFold(n_splits=5).split(X, y, g):
            if kind == "mc":
                clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=300, C=1.0).fit(X[tr], y[tr])
                p = clf.predict_proba(X[va])
                for i, c in enumerate(clf.classes_):
                    stk[va, int(c)] = p[:, i]
            else:
                clf = LogisticRegression(max_iter=300, C=1.0).fit(X[tr], y[tr])
                stk[va, 0] = clf.predict_proba(X[va])[:, 1]
        out[target] = stk
    return honest_scores(out["action"], out["point"], out["server"].ravel(), ya, yp, ys, g)


def main() -> None:
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    lab = _labeled(train)
    ya = lab["actionId"].astype(int).to_numpy()
    yp = lab["pointId"].astype(int).to_numpy()
    ys = lab["serverGetPoint"].astype(int).to_numpy()
    g = lab["match"].to_numpy()
    print(f"pilot slice rows: {len(lab)} (seed {SLICE_SEED}, folds {SLICE_FOLDS})")

    standalone = {}
    for name in ["lgbm15", SEQ]:
        standalone[name] = honest_scores(
            _probs(name, "action", 19, lab), _probs(name, "point", 10, lab),
            _probs(name, "server", 1, lab).ravel(), ya, yp, ys, g)
    print("=== standalone honest scores (pilot slice) ===")
    print(json.dumps(standalone, indent=2))

    existing = {t: EXISTING + [CHAIN[t]] for t in ("action", "point", "server")}
    with_seq = {t: EXISTING + [CHAIN[t], SEQ] for t in ("action", "point", "server")}
    e0 = _stack(existing, lab, ya, yp, ys, g)
    e1 = _stack(with_seq, lab, ya, yp, ys, g)
    lift = e1["overall"] - e0["overall"]
    print("=== ensemble honest overall (pilot slice) ===")
    print(f"existing 5 bases      : {e0['overall']:.4f}")
    print(f"existing + {SEQ}: {e1['overall']:.4f}")
    print(f"seq ensemble lift     : {lift:+.4f} (noise floor {NOISE})")

    seq_a = standalone[SEQ]["action_f1"]
    lg_a = standalone["lgbm15"]["action_f1"]
    green = (seq_a >= 0.32) or (seq_a >= lg_a + 0.03) or (lift > NOISE)
    verdict = "GREEN -> proceed to Phase 2" if green else "YELLOW -> reassess; do NOT burn multi-day GPU"
    print(f"\nVERDICT: {verdict}")
    print(f"  seq action {seq_a:.4f} vs lgbm15 {lg_a:.4f}; ensemble lift {lift:+.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the comparison**

Run:
```bash
conda run -n aicup-tt python -m scripts.score_seq_pilot seq_pilot 2>&1 | grep -vi "warning\|n_iter_i"
```
Expected: prints the slice row count, standalone JSON (lgbm15 vs seq_pilot),
the ensemble overall with/without seq, and a `VERDICT: GREEN ...` or
`VERDICT: YELLOW ...` line.

- [ ] **Step 3: Commit the scorer**

```bash
git add scripts/score_seq_pilot.py
git commit -m "feat(seq): pilot honest comparison CLI + GREEN/YELLOW verdict"
```

---

### Task 5: Light hyperparameter sweep (2 alternates)

**Files:** none created — produces `seq_pilot_d256` / `seq_pilot_drop1` OOFs.

Only run this if the base pilot (Task 4) is borderline or to confirm the best
config. Keep it to two extra runs (short sequences rarely need more).

- [ ] **Step 1: Run alternate A — wider model**

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt python -u -m scripts.train_seq_pilot \
  --seeds 11 --folds 0 1 2 --epochs 80 --patience 12 \
  --d-model 256 --layers 4 --nhead 4 --ffn 768 --dropout 0.2 \
  --batch-size 256 --lr 3e-4 --warmup-frac 0.05 --clip 1.0 \
  --model-name seq_pilot_d256
```
Expected: same shape of output as Task 3, writing `seq_pilot_d256_*` OOFs.

- [ ] **Step 2: Run alternate B — lower dropout**

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt python -u -m scripts.train_seq_pilot \
  --seeds 11 --folds 0 1 2 --epochs 80 --patience 12 \
  --d-model 192 --layers 4 --nhead 4 --ffn 512 --dropout 0.1 \
  --batch-size 256 --lr 3e-4 --warmup-frac 0.05 --clip 1.0 \
  --model-name seq_pilot_drop1
```

- [ ] **Step 3: Score both and pick the winner**

```bash
conda run -n aicup-tt python -m scripts.score_seq_pilot seq_pilot_d256 2>&1 | grep -i "VERDICT\|seq action\|overall"
conda run -n aicup-tt python -m scripts.score_seq_pilot seq_pilot_drop1 2>&1 | grep -i "VERDICT\|seq action\|overall"
```
Expected: a verdict + scores for each. Record which `--d-model/--dropout`
config gives the highest honest standalone action_f1 / ensemble lift — that is
the **winning config** carried into Phase 2.

- [ ] **Step 4: Commit the winning OOF**

```bash
git add -f artifacts/oof/<winning_model>_action.parquet artifacts/oof/<winning_model>_point.parquet artifacts/oof/<winning_model>_server.parquet
git add artifacts/<winning_model>_run_log.json
git commit -m "feat(seq): pilot sweep; record winning config OOF"
```

---

### Task 6: Record the pilot verdict in PROGRESS.md

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1: Append a pilot-results section**

Add to `PROGRESS.md` (under the CRITICAL CORRECTION block) a new section with:
the winning config (`--d-model`, `--dropout`), the honest standalone scores
(seq action/point/server vs lgbm15), the ensemble lift on the slice, the
GREEN/YELLOW verdict, and the literal go/no-go decision for Phase 2. Example
template to fill with the real numbers:

```markdown
## Sequence-model pilot results (2026-05-28)

Slice: seed 11 × folds 0–2 (~9k rows). Honest per-row scoring.
Winning config: d_model=<W>, dropout=<W>, early-stopped at ~<E> epochs.

| model      | action F1 | point F1 | server AUC | overall |
|------------|----------:|---------:|-----------:|--------:|
| lgbm15     | <...>     | <...>    | <...>      | <...>   |
| seq (best) | <...>     | <...>    | <...>      | <...>   |

Ensemble (5 bases) overall <...> vs (+seq) <...> -> lift <...> (noise 0.00168).
VERDICT: <GREEN/YELLOW>. Decision: <proceed to Phase 2 / reassess>.
```

- [ ] **Step 2: Commit**

```bash
git add PROGRESS.md
git commit -m "docs(progress): record sequence-model pilot verdict"
```

---

## PHASE 2 — Full integration (ONLY if Task 6 verdict is GREEN; multi-day GPU)

Do not start Phase 2 unless the pilot verdict is GREEN. Use the **winning config**
from Task 5 everywhere below; substitute its `--d-model`/`--dropout` for `<W>`.

### Task 7: Full 25-fold OOF

- [ ] **Step 1: Train all 5 seeds × 5 folds**

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt python -u -m scripts.train_seq_pilot \
  --seeds 11 22 33 44 55 --folds 0 1 2 3 4 --epochs 80 --patience 12 \
  --d-model <W> --layers 4 --nhead 4 --ffn 512 --dropout <W> \
  --batch-size 256 --lr 3e-4 --warmup-frac 0.05 --clip 1.0 \
  --model-name seq2
```
Expected: 25 folds trained; `artifacts/oof/seq2_{action,point,server}.parquet`
each with ~74,975 rows. This is the multi-hour/overnight run — launch in the
background and monitor the log.

- [ ] **Step 2: Score seq2 standalone on the full OOF**

```bash
conda run -n aicup-tt python -m scripts.score_oof seq2
```
Expected: prints/writes `artifacts/base_oof_scores.json` including `seq2`.
(`score_oof.score_model` scores per-row across all cuts — already honest.)

- [ ] **Step 3: Commit**

```bash
git add -f artifacts/oof/seq2_action.parquet artifacts/oof/seq2_point.parquet artifacts/oof/seq2_server.parquet
git add artifacts/seq2_run_log.json
git commit -m "feat(seq2): full 25-fold sequence-model OOF"
```

### Task 8: Full-train test inference for seq2

The pilot trainer only writes val OOF. Phase 2 needs single-cut test predictions
written to `artifacts/oof/seq2_{action,point,server}_test.parquet` (1,845 rows,
one per test rally), matching the base `*_test.parquet` schema.

> **Handoff note (do a 15-min design pass before coding this task):** the exact
> code is intentionally NOT pre-written because it depends on (a) the pilot's
> winning config and (b) how a test rally's "cut" maps into `RallyPrefixDataset`
> — `test_new.csv` gives each rally its observed prefix, and the target stroke is
> unobserved, so `attach_labels` cannot be used for test. First verify: for a
> test rally, the prefix = all observed strokes, and the model predicts the next
> (cut = max observed strikeNumber + 1). Confirm `test_new.csv` has no row at the
> target cut. Then implement the steps below. Keep `train_seq_transformer.py`
> and the Phase-1 path untouched.

- [ ] **Step 1: Add a test-inference path**

In `scripts/train_seq_pilot.py`, add a `--predict-test` flag. When set: after
the seed/fold loop, train one model on the **entire** train set (no held-out
fold) for `args.epochs` (no early stopping — no val set; use a fixed epoch count
equal to the median early-stop epoch observed in Task 7, default 40), build a
`RallyPrefixDataset` over `test_new.csv` (note: test rallies' cut = max observed
strikeNumber + 1; the dataset's prefix logic already uses `strikeNumber < cut`),
run `_predict`, and write `artifacts/oof/{model_name}_{target}_test.parquet` with
columns `rally_uid` + `p_*` (drop seed/fold/cut for the test file to match the
base `*_test.parquet` shape). Implement the test-cut dataset by appending a
synthetic cut row per test rally so `attach_labels` is not needed (labels are
unknown for test). Show the test loader producing exactly 1,845 rows.

- [ ] **Step 2: Run test inference**

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt python -u -m scripts.train_seq_pilot \
  --predict-test --seeds 11 --folds 0 --d-model <W> --dropout <W> --epochs 40 \
  --model-name seq2
```
Expected: writes `artifacts/oof/seq2_{action,point,server}_test.parquet`, each
1,845 rows, columns `rally_uid` + `p_*`.

- [ ] **Step 3: Commit**

```bash
git add -f artifacts/oof/seq2_action_test.parquet artifacts/oof/seq2_point_test.parquet artifacts/oof/seq2_server_test.parquet
git commit -m "feat(seq2): full-train test-time inference"
```

### Task 9: Integrate seq2 into the per-row ensemble

- [ ] **Step 1: Add seq2 to the base set**

In `scripts/build_final_perrow.py`, add to the `BASES` dict: `"seq2_action"`
to the `action` list, `"seq2_point"` to `point`, `"seq2_server"` to `server`.
(The OOF and `_test` parquets created in Tasks 7–8 must use those exact model
names — i.e. rename via `--model-name seq2_action`-style if needed, OR adjust
`_perrow_features`/`_test_features` to read `seq2_{target}` for the seq base.
Simplest: in `build_final_perrow.py`, special-case the seq base name per target
to `f"seq2"` reading `seq2_{target}.parquet`.)

- [ ] **Step 2: Rebuild and score the ensemble**

```bash
conda run -n aicup-tt python -m scripts.build_final_perrow 2>&1 | grep -vi "warning\|n_iter_i"
```
Expected: prints honest per-row ensemble scores. Compare `overall` to the
current **0.3206**. Lift must exceed the **0.00168** noise floor to be real.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_final_perrow.py artifacts/final_perrow_scores.json artifacts/submission_FINAL_safe_perrow.csv artifacts/submission_FINAL_smooth_perrow.csv
git commit -m "feat(seq2): integrate sequence model into per-row ensemble"
```

### Task 10: Final submissions + record

- [ ] **Step 1: Verify guardrails and inspect the new submission**

```bash
conda run -n aicup-tt python -c "import pandas as pd; s=pd.read_csv('artifacts/submission_FINAL_safe_perrow.csv'); assert s.rally_uid.nunique()==1845; assert s.actionId.between(0,18).all() and s.pointId.between(0,9).all() and s.serverGetPoint.between(0,1).all(); print('guardrails OK', s.shape)"
```
Expected: `guardrails OK (1845, 4)`.

- [ ] **Step 2: Record results in PROGRESS.md**

Add the new honest ensemble overall (with seq2), the lift vs 0.3206, and the
implied public-smooth projection. State whether the public-smooth 0.5 target now
looks reachable. Commit:
```bash
git add PROGRESS.md
git commit -m "docs(progress): record seq2 ensemble result"
```

- [ ] **Step 3: Decide on at most one public upload**

Per the user's constraint (daily-limited, teammate-shared uploads): only suggest
uploading `submission_FINAL_safe_perrow.csv` if the honest local overall improved
by > noise floor. The smooth variant is the public-sprint upload toward 0.5.

---

## Self-review checklist (run before handing off)

- [ ] `conda run -n aicup-tt python -m pytest -q` — all tests green (existing 32 + new seq_eval 3 = 35).
- [ ] No seed-averaging anywhere in new code (per-row scoring only).
- [ ] Every new function referenced in a later task is defined in an earlier task (`monitor_score`, `honest_scores`, `warmup_cosine_lambda`, `_split_indices`, `_predict`, `_stack`, `_probs`).
- [ ] GPU commands all use `env CUDA_VISIBLE_DEVICES=0`.
- [ ] OOF parquet `git add` uses `-f` (artifacts/oof/*.parquet is gitignored).
- [ ] Phase 2 is clearly gated on a GREEN Task 6 verdict.
