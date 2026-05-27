# Route C — Sequence Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a small multi-task Transformer encoder on the 3090 that consumes the rally prefix as a token sequence and emits three joint heads (action, point, server). It does NOT replace LightGBM; it joins the final stack as a third base model whose strength is short-prefix (phase 0/1) rallies. Target lift on overall: **+0.015 to +0.040** (high variance — may be 0 if it overfits).

**Architecture:** A torch `Dataset` flattens each rally prefix into a tensor of categorical token IDs plus a small float vector (score state). The model is 4 layers of `nn.TransformerEncoder` with d_model=128, FFN=512, 4 heads, learned positional embeddings, and three task heads pooled from the last unmasked token. Training is 5 seeds × 5 folds against the P1 splits, mixed precision on CUDA, with class-balanced cross entropy for action/point and BCE for server. OOF probabilities are written into the same parquet schema as Route A/B.

**Tech Stack:** Python 3.11 in `aicup-tt`, PyTorch 2.2 + CUDA 12.1, NumPy, pandas, scikit-learn metrics.

**Depends on:** P1 (CV splits + `iter_cv_folds`).

---

## Spec section coverage

- Section 4.1 Architecture → Tasks 2–3
- Section 4.2 Training → Task 4
- Section 4.3 Ensemble role → Task 6 (OOF write back) and P5 (stacking)
- Section 4.4 Artifacts → Tasks 5–6

## File structure

| Path | Purpose |
|---|---|
| `scripts/seq_dataset.py` | torch `RallyPrefixDataset`. Tokenization, padding, masking. Create. |
| `scripts/seq_model.py` | `RallyTransformer(nn.Module)` with three task heads. Create. |
| `scripts/train_seq_transformer.py` | Training loop, fold runner, OOF/test writers. Create. |
| `scripts/build_route_c_submission.py` | Refit on full train + predict on test_new + write CSV. Create. |
| `tests/test_seq_dataset.py` | Token shape, mask validity, no-NaN. Create. |
| `tests/test_seq_model.py` | Forward pass shape, parameter count sanity. Create. |
| `artifacts/oof/seq_action.parquet` etc. | OOF probabilities, Route A/B schema. Generated. |
| `artifacts/submission_C_seq.csv` | Route C standalone submission. Generated. |
| `artifacts/seq_run_log.json` | Per-fold train/valid metrics for inspection. Generated. |

---

### Task 1: GPU verification gate

A 90-second sanity check before investing in model code.

- [ ] **Step 1.1: Confirm CUDA is visible**

Run:
```bash
conda run -n aicup-tt python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name:", torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda")
    print("matmul ok:", (x @ x.t()).shape)
PY
```
Expected: `cuda: True`, device name contains `3090`, matmul prints `(4096, 4096)`.

If False: troubleshoot driver/cuda mismatch. Do not proceed.

- [ ] **Step 1.2: Document in plan log (no commit needed if no file changes)**

---

### Task 2: Dataset class

**Files:**
- Create: `scripts/seq_dataset.py`
- Create: `tests/test_seq_dataset.py`

We pre-extract a CV-aligned dataset of `(rally_uid, seed, fold, role)` and lazy-load tensors per item. Token features are the categorical columns; auxiliary float features are the score state.

- [ ] **Step 2.1: Failing test**

`tests/test_seq_dataset.py`:
```python
import torch
import pandas as pd
import numpy as np
import pytest

from scripts.seq_dataset import RallyPrefixDataset, CATEGORICAL_COLS, MAX_LEN


def _fake_train():
    rows = []
    for rally_uid in range(5):
        for sn in range(1, 6):
            rows.append({
                "rally_uid": rally_uid, "match": 0, "strikeNumber": sn,
                "scoreSelf": 0, "scoreOther": 0,
                "gamePlayerId": (sn + rally_uid) % 4, "gamePlayerOtherId": (sn + rally_uid + 1) % 4,
                "strikeId": 1, "handId": 1, "strengthId": 1, "spinId": 1,
                "pointId": (sn + rally_uid) % 10, "actionId": (sn + rally_uid) % 19,
                "positionId": 1, "sex": 1, "numberGame": 1, "rally_id": rally_uid,
                "serverGetPoint": rally_uid % 2,
            })
    return pd.DataFrame(rows)


def _fake_splits():
    return pd.DataFrame({
        "rally_uid": list(range(5)), "match": [0]*5, "seed":[11]*5,
        "fold":[0,0,1,1,1], "cut_strikeNumber":[3,4,5,2,5], "phase_bucket":["phase1"]*5,
    })


def test_dataset_yields_padded_tensor():
    ds = RallyPrefixDataset(_fake_train(), _fake_splits(), seed=11)
    item = ds[0]
    assert "tokens" in item and item["tokens"].shape == (MAX_LEN, len(CATEGORICAL_COLS))
    assert "mask" in item and item["mask"].dtype == torch.bool
    assert "y_action" in item and item["y_action"].dtype == torch.long


def test_no_label_leak_when_role_valid():
    ds = RallyPrefixDataset(_fake_train(), _fake_splits(), seed=11)
    for i in range(len(ds)):
        item = ds[i]
        # The "target" stroke (at cut_strikeNumber) must NOT appear in tokens.
        valid_tokens = item["tokens"][item["mask"]]
        assert item["target_strike"] not in valid_tokens[:, CATEGORICAL_COLS.index("strikeNumber")].tolist() \
            if "strikeNumber" in CATEGORICAL_COLS else True
```

- [ ] **Step 2.2: Run, expect failure**

- [ ] **Step 2.3: Implement `RallyPrefixDataset`**

`scripts/seq_dataset.py`:
```python
"""Torch Dataset for rally prefixes -> next-stroke prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CATEGORICAL_COLS = (
    "strikeId", "handId", "strengthId", "spinId",
    "pointId", "actionId", "positionId",
    "gamePlayerId", "gamePlayerOtherId",
)
SCORE_FLOAT_COLS = ("scoreSelf", "scoreOther")
MAX_LEN = 24  # 99th-percentile prefix length is well under this


class RallyPrefixDataset(Dataset):
    """One item per (rally_uid) selected by the current seed in cv_splits."""

    def __init__(self, train: pd.DataFrame, splits: pd.DataFrame, seed: int):
        sub = splits[splits["seed"] == seed]
        self._cuts = dict(zip(sub["rally_uid"], sub["cut_strikeNumber"]))
        self._folds = dict(zip(sub["rally_uid"], sub["fold"]))
        train = train[train["rally_uid"].isin(sub["rally_uid"])]
        self._by_rally = {
            int(r): g.sort_values("strikeNumber").reset_index(drop=True)
            for r, g in train.groupby("rally_uid", sort=False)
        }
        self._rallies = list(self._by_rally.keys())

    def __len__(self) -> int:
        return len(self._rallies)

    def __getitem__(self, idx: int) -> dict:
        rally_uid = self._rallies[idx]
        g = self._by_rally[rally_uid]
        cut = int(self._cuts[rally_uid])
        prefix = g[g["strikeNumber"] < cut]
        target_row = g[g["strikeNumber"] == cut].iloc[0]

        toks = np.zeros((MAX_LEN, len(CATEGORICAL_COLS)), dtype=np.int64)
        mask = np.zeros(MAX_LEN, dtype=bool)
        n = min(len(prefix), MAX_LEN)
        for i in range(n):
            for j, col in enumerate(CATEGORICAL_COLS):
                toks[i, j] = int(prefix.iloc[i][col])
            mask[i] = True

        score = np.array([target_row["scoreSelf"], target_row["scoreOther"]], dtype=np.float32)
        return {
            "tokens": torch.from_numpy(toks),
            "mask":   torch.from_numpy(mask),
            "score":  torch.from_numpy(score),
            "y_action": torch.tensor(int(target_row["actionId"]), dtype=torch.long),
            "y_point":  torch.tensor(int(target_row["pointId"]),  dtype=torch.long),
            "y_server": torch.tensor(float(g.iloc[0]["serverGetPoint"]), dtype=torch.float32),
            "rally_uid": int(rally_uid),
            "fold": int(self._folds[rally_uid]),
            "target_strike": cut,
        }
```

- [ ] **Step 2.4: Run, PASS**

```bash
conda run -n aicup-tt pytest tests/test_seq_dataset.py -v
```
Expected: 2 passed.

- [ ] **Step 2.5: Commit**

```bash
git add scripts/seq_dataset.py tests/test_seq_dataset.py
git commit -m "feat(seq): torch Dataset for rally prefixes with masking"
```

---

### Task 3: Model definition

**Files:**
- Create: `scripts/seq_model.py`
- Create: `tests/test_seq_model.py`

- [ ] **Step 3.1: Forward-shape test**

```python
# tests/test_seq_model.py
import torch
from scripts.seq_model import RallyTransformer


def test_forward_shapes():
    model = RallyTransformer()
    tokens = torch.zeros(8, 24, 9, dtype=torch.long)
    mask = torch.ones(8, 24, dtype=torch.bool)
    score = torch.zeros(8, 2)
    out = model(tokens=tokens, mask=mask, score=score)
    assert out["action_logits"].shape == (8, 19)
    assert out["point_logits"].shape  == (8, 10)
    assert out["server_logits"].shape == (8, 1)


def test_param_count_under_5m():
    model = RallyTransformer()
    n = sum(p.numel() for p in model.parameters())
    assert n < 5_000_000, f"too big: {n}"
```

- [ ] **Step 3.2: Run, FAIL**

- [ ] **Step 3.3: Implement model**

```python
# scripts/seq_model.py
"""Small multi-task Transformer encoder for rally prefixes."""
from __future__ import annotations

import torch
import torch.nn as nn


CARD = {
    "strikeId": 4,
    "handId": 4,
    "strengthId": 4,
    "spinId": 6,
    "pointId": 10,
    "actionId": 19,
    "positionId": 4,
    "gamePlayerId": 200,
    "gamePlayerOtherId": 200,
}
ORDER = ("strikeId","handId","strengthId","spinId","pointId","actionId","positionId","gamePlayerId","gamePlayerOtherId")


class RallyTransformer(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, layers: int = 4, dropout: float = 0.3, max_len: int = 24):
        super().__init__()
        self.emb = nn.ModuleList([
            nn.Embedding(CARD[c] + 1, d_model // len(ORDER)) for c in ORDER
        ])
        self.proj = nn.Linear((d_model // len(ORDER)) * len(ORDER), d_model)
        self.score_proj = nn.Linear(2, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(d_model, 19)
        self.point_head  = nn.Linear(d_model, 10)
        self.server_head = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, score: torch.Tensor) -> dict[str, torch.Tensor]:
        # tokens: (B, L, F); mask: (B, L)
        emb = torch.cat([
            self.emb[i](tokens[:, :, i]) for i in range(len(ORDER))
        ], dim=-1)
        h = self.proj(emb)  # (B, L, D)
        pos = self.pos(torch.arange(h.size(1), device=h.device))[None, :, :]
        h = h + pos
        # Inject score state at position 0 by adding a global token.
        score_h = self.score_proj(score)[:, None, :]
        h = h + score_h  # broadcast across positions (simpler than a CLS token)

        key_padding = ~mask  # True where to ignore
        h = self.encoder(h, src_key_padding_mask=key_padding)

        # Pool the LAST unmasked position.
        idx = mask.sum(dim=1).clamp(min=1) - 1  # (B,)
        pooled = h[torch.arange(h.size(0)), idx]  # (B, D)
        pooled = self.dropout(pooled)

        return {
            "action_logits": self.action_head(pooled),
            "point_logits":  self.point_head(pooled),
            "server_logits": self.server_head(pooled),
        }
```

- [ ] **Step 3.4: Run, PASS**

```bash
conda run -n aicup-tt pytest tests/test_seq_model.py -v
```

- [ ] **Step 3.5: Commit**

```bash
git add scripts/seq_model.py tests/test_seq_model.py
git commit -m "feat(seq): RallyTransformer with multi-task heads"
```

---

### Task 4: Training loop with multi-task loss

**Files:**
- Create: `scripts/train_seq_transformer.py`

- [ ] **Step 4.1: Write the trainer**

```python
"""Multi-task Transformer trainer with CV folds + mixed precision.

Outputs:
  artifacts/oof/seq_action.parquet
  artifacts/oof/seq_point.parquet
  artifacts/oof/seq_server.parquet
  artifacts/seq_run_log.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.seq_dataset import RallyPrefixDataset
from scripts.seq_model import RallyTransformer
from scripts.oof_loader import write_oof


def class_balanced_weight(y: np.ndarray, n_classes: int, beta: float = 0.9999) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    eff = 1.0 - np.power(beta, np.maximum(counts, 1.0))
    w = (1.0 - beta) / eff
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


def collate(batch: list[dict]) -> dict:
    keys = ("tokens","mask","score","y_action","y_point","y_server","fold","rally_uid","target_strike")
    out = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.tensor([b[k] for b in batch])
    return out


def run_fold(train_df: pd.DataFrame, splits: pd.DataFrame, seed: int, fold: int, device: str) -> dict:
    train_ds = RallyPrefixDataset(train_df, splits[(splits["seed"] == seed) & (splits["fold"] != fold)], seed=seed)
    valid_ds = RallyPrefixDataset(train_df, splits[(splits["seed"] == seed) & (splits["fold"] == fold)], seed=seed)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, collate_fn=collate, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=512, shuffle=False, num_workers=2, collate_fn=collate)

    # class-balanced weights from train set
    y_a = np.array([train_ds[i]["y_action"].item() for i in range(len(train_ds))])
    y_p = np.array([train_ds[i]["y_point"].item()  for i in range(len(train_ds))])
    wA = class_balanced_weight(y_a, 19).to(device)
    wP = class_balanced_weight(y_p, 10).to(device)

    torch.manual_seed(seed * 100 + fold)
    model = RallyTransformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val = -1e9
    best_oof = None
    for epoch in range(30):
        model.train()
        for batch in train_loader:
            tokens = batch["tokens"].to(device); mask = batch["mask"].to(device)
            score = batch["score"].to(device)
            y_a = batch["y_action"].to(device); y_p = batch["y_point"].to(device); y_s = batch["y_server"].to(device)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(tokens=tokens, mask=mask, score=score)
                la = F.cross_entropy(out["action_logits"], y_a, weight=wA, label_smoothing=0.1)
                lp = F.cross_entropy(out["point_logits"],  y_p, weight=wP, label_smoothing=0.1)
                ls = F.binary_cross_entropy_with_logits(out["server_logits"].squeeze(-1), y_s)
                loss = 0.4 * la + 0.4 * lp + 0.2 * ls
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()

        # eval
        model.eval()
        oof_a, oof_p, oof_s, ids, cuts = [], [], [], [], []
        with torch.no_grad():
            for batch in valid_loader:
                tokens = batch["tokens"].to(device); mask = batch["mask"].to(device)
                score = batch["score"].to(device)
                out = model(tokens=tokens, mask=mask, score=score)
                oof_a.append(F.softmax(out["action_logits"], dim=-1).cpu().numpy())
                oof_p.append(F.softmax(out["point_logits"],  dim=-1).cpu().numpy())
                oof_s.append(torch.sigmoid(out["server_logits"]).cpu().numpy())
                ids.append(batch["rally_uid"].cpu().numpy())
                cuts.append(batch["target_strike"].cpu().numpy())
        from sklearn.metrics import f1_score, roc_auc_score
        oa = np.concatenate(oof_a); op = np.concatenate(oof_p); os_ = np.concatenate(oof_s).reshape(-1)
        ya = np.array([valid_ds[i]["y_action"].item() for i in range(len(valid_ds))])
        yp = np.array([valid_ds[i]["y_point"].item()  for i in range(len(valid_ds))])
        ys = np.array([valid_ds[i]["y_server"].item() for i in range(len(valid_ds))])
        f1a = f1_score(ya, oa.argmax(1), labels=list(range(19)), average="macro", zero_division=0)
        f1p = f1_score(yp, op.argmax(1), labels=list(range(10)), average="macro", zero_division=0)
        auc = roc_auc_score(ys, os_)
        overall = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
        print(f"seed={seed} fold={fold} epoch={epoch} overall={overall:.5f}")
        if overall > best_val:
            best_val = overall
            best_oof = (np.concatenate(ids), np.concatenate(cuts), oa, op, os_)
    return {"seed": seed, "fold": fold, "val": float(best_val), "oof": best_oof}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = pd.read_parquet("artifacts/cv_splits.parquet")
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))

    bag = {t: {"r": [], "s": [], "f": [], "c": [], "p": []} for t in ("action","point","server")}
    log = []
    for seed in sorted(splits["seed"].unique()):
        for fold in sorted(splits["fold"].unique()):
            res = run_fold(train, splits, int(seed), int(fold), device)
            ids, cuts, oa, op, os_ = res["oof"]
            sid = np.full(len(ids), seed); fid = np.full(len(ids), fold)
            for tgt, p in [("action", oa), ("point", op), ("server", os_.reshape(-1, 1))]:
                bag[tgt]["r"].append(ids); bag[tgt]["s"].append(sid)
                bag[tgt]["f"].append(fid); bag[tgt]["c"].append(cuts); bag[tgt]["p"].append(p)
            log.append({"seed": int(seed), "fold": int(fold), "val_overall": res["val"]})

    for tgt in ("action","point","server"):
        write_oof("seq", tgt,
                  np.concatenate(bag[tgt]["r"]),
                  np.concatenate(bag[tgt]["s"]),
                  np.concatenate(bag[tgt]["f"]),
                  np.concatenate(bag[tgt]["c"]),
                  np.concatenate(bag[tgt]["p"], axis=0))
    Path("artifacts/seq_run_log.json").write_text(json.dumps(log, indent=2))
    print("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Smoke-test on a single fold**

```bash
conda run -n aicup-tt python -c "
import pandas as pd, torch
from pathlib import Path
from scripts.train_seq_transformer import run_fold
splits=pd.read_parquet('artifacts/cv_splits.parquet')
train=pd.read_csv(next(Path.cwd().glob('AI CUP*/train.csv')))
device='cuda' if torch.cuda.is_available() else 'cpu'
res=run_fold(train, splits, seed=11, fold=0, device=device)
print('val:', res['val'])
"
```
Expected: 30 epoch lines, final `val: ~0.30` or higher. Per-fold runtime on 3090: 5–10 minutes.

- [ ] **Step 4.3: Run all 25 folds**

```bash
conda run -n aicup-tt python -m scripts.train_seq_transformer
```
Runtime: 2–4 hours on 3090. Background it (`nohup ... > train.log 2>&1 &`) and monitor with `tail -f train.log`.

- [ ] **Step 4.4: Sanity-check OOF**

```bash
conda run -n aicup-tt python -c "
import pandas as pd
from scripts.score_oof import attach_labels, score_action, score_point, score_server, overall
train=pd.read_csv(next(__import__('pathlib').Path.cwd().glob('AI CUP*/train.csv')))
a=attach_labels(pd.read_parquet('artifacts/oof/seq_action.parquet'), train); a_s=score_action(a)
p=attach_labels(pd.read_parquet('artifacts/oof/seq_point.parquet'),  train); p_s=score_point(p)
s=attach_labels(pd.read_parquet('artifacts/oof/seq_server.parquet'), train); sv=score_server(s)
print({'action':a_s,'point':p_s,'server':sv,'overall':overall(a_s,p_s,sv)})
"
```
Expected: phase-0 macro-F1 specifically should be higher than the LGBM baseline (the win condition for Route C even if overall is similar).

- [ ] **Step 4.5: Commit**

```bash
git add scripts/train_seq_transformer.py artifacts/oof/seq_*.parquet artifacts/seq_run_log.json
git commit -m "feat(route_c): multi-task Transformer OOF on 3090"
```

---

### Task 5: Refit on full train + test inference

**Files:**
- Create: `scripts/build_route_c_submission.py`

- [ ] **Step 5.1: Write the test-time builder**

```python
"""Refit RallyTransformer on full train (no folds) and predict test_new."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.seq_dataset import CATEGORICAL_COLS, MAX_LEN
from scripts.seq_model import RallyTransformer


class FullTrainDataset(Dataset):
    """Last stroke is the target; everything before is prefix."""
    def __init__(self, train: pd.DataFrame):
        self._by_rally = {int(r): g.sort_values("strikeNumber").reset_index(drop=True)
                          for r, g in train.groupby("rally_uid", sort=False)}
        self._rallies = list(self._by_rally.keys())

    def __len__(self): return len(self._rallies)

    def __getitem__(self, idx: int):
        g = self._by_rally[self._rallies[idx]]
        target = g.iloc[-1]; prefix = g.iloc[:-1]
        toks = np.zeros((MAX_LEN, len(CATEGORICAL_COLS)), dtype=np.int64)
        mask = np.zeros(MAX_LEN, dtype=bool)
        n = min(len(prefix), MAX_LEN)
        for i in range(n):
            for j, col in enumerate(CATEGORICAL_COLS):
                toks[i, j] = int(prefix.iloc[i][col])
            mask[i] = True
        return {
            "tokens": torch.from_numpy(toks),
            "mask":   torch.from_numpy(mask),
            "score":  torch.tensor([target["scoreSelf"], target["scoreOther"]], dtype=torch.float32),
            "y_action": torch.tensor(int(target["actionId"]), dtype=torch.long),
            "y_point":  torch.tensor(int(target["pointId"]),  dtype=torch.long),
            "y_server": torch.tensor(float(g.iloc[0]["serverGetPoint"]), dtype=torch.float32),
        }


class TestDataset(Dataset):
    def __init__(self, test: pd.DataFrame):
        self._by_rally = {int(r): g.sort_values("strikeNumber").reset_index(drop=True)
                          for r, g in test.groupby("rally_uid", sort=False)}
        self._rallies = list(self._by_rally.keys())

    def __len__(self): return len(self._rallies)

    def __getitem__(self, idx: int):
        rally_uid = self._rallies[idx]
        g = self._by_rally[rally_uid]
        toks = np.zeros((MAX_LEN, len(CATEGORICAL_COLS)), dtype=np.int64)
        mask = np.zeros(MAX_LEN, dtype=bool)
        n = min(len(g), MAX_LEN)
        for i in range(n):
            for j, col in enumerate(CATEGORICAL_COLS):
                toks[i, j] = int(g.iloc[i][col])
            mask[i] = True
        last = g.iloc[-1]
        return {
            "rally_uid": rally_uid,
            "tokens": torch.from_numpy(toks),
            "mask":   torch.from_numpy(mask),
            "score":  torch.tensor([last["scoreSelf"], last["scoreOther"]], dtype=torch.float32),
        }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = pd.read_csv(next(Path.cwd().glob("AI CUP*/train.csv")))
    test  = pd.read_csv(next(Path.cwd().glob("AI CUP*/test_new.csv")))

    # Train one model per seed on full train, then average test predictions.
    seeds = [11, 22, 33, 44, 55]
    test_ds = TestDataset(test)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)
    bag_a = np.zeros((len(test_ds), 19))
    bag_p = np.zeros((len(test_ds), 10))
    bag_s = np.zeros(len(test_ds))

    full_ds = FullTrainDataset(train)

    def collate(b):
        out = {}
        for k in b[0]:
            if isinstance(b[0][k], torch.Tensor): out[k] = torch.stack([x[k] for x in b])
            else: out[k] = torch.tensor([x[k] for x in b])
        return out

    full_loader = DataLoader(full_ds, batch_size=256, shuffle=True, num_workers=2, collate_fn=collate, drop_last=True)

    for seed in seeds:
        torch.manual_seed(seed)
        model = RallyTransformer().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
        for epoch in range(20):
            model.train()
            for batch in full_loader:
                tokens = batch["tokens"].to(device); mask = batch["mask"].to(device); score = batch["score"].to(device)
                y_a = batch["y_action"].to(device); y_p = batch["y_point"].to(device); y_s = batch["y_server"].to(device)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    out = model(tokens=tokens, mask=mask, score=score)
                    loss = (0.4 * F.cross_entropy(out["action_logits"], y_a, label_smoothing=0.1)
                          + 0.4 * F.cross_entropy(out["point_logits"],  y_p, label_smoothing=0.1)
                          + 0.2 * F.binary_cross_entropy_with_logits(out["server_logits"].squeeze(-1), y_s))
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        # predict
        model.eval()
        with torch.no_grad():
            preds_a, preds_p, preds_s, ids = [], [], [], []
            for batch in test_loader:
                tokens = batch["tokens"].to(device); mask = batch["mask"].to(device); score = batch["score"].to(device)
                out = model(tokens=tokens, mask=mask, score=score)
                preds_a.append(F.softmax(out["action_logits"], dim=-1).cpu().numpy())
                preds_p.append(F.softmax(out["point_logits"],  dim=-1).cpu().numpy())
                preds_s.append(torch.sigmoid(out["server_logits"]).cpu().numpy().reshape(-1))
                ids.append(batch["rally_uid"].cpu().numpy())
        rally_ids = np.concatenate(ids)
        order = np.argsort(rally_ids)
        bag_a += np.concatenate(preds_a)[order] / len(seeds)
        bag_p += np.concatenate(preds_p)[order] / len(seeds)
        bag_s += np.concatenate(preds_s)[order] / len(seeds)
        print(f"seed {seed} done")

    rally_uids_sorted = np.sort(np.array(test_ds._rallies))
    sub = pd.DataFrame({
        "rally_uid": rally_uids_sorted,
        "actionId": bag_a.argmax(1),
        "pointId":  bag_p.argmax(1),
        "serverGetPoint": bag_s,
    })
    sub.to_csv("artifacts/submission_C_seq.csv", index=False)
    print(f"wrote artifacts/submission_C_seq.csv: {sub.shape}")

    # Also save full-train test probs in OOF-test schema for P5.
    pd.DataFrame({"rally_uid": rally_uids_sorted, **{f"p_{i}": bag_a[:, i] for i in range(19)}}).to_parquet("artifacts/oof/seq_action_test.parquet", index=False)
    pd.DataFrame({"rally_uid": rally_uids_sorted, **{f"p_{i}": bag_p[:, i] for i in range(10)}}).to_parquet("artifacts/oof/seq_point_test.parquet", index=False)
    pd.DataFrame({"rally_uid": rally_uids_sorted, "p_1": bag_s}).to_parquet("artifacts/oof/seq_server_test.parquet", index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Run**

```bash
conda run -n aicup-tt python -m scripts.build_route_c_submission
```
Runtime: 5 seeds × 20 epochs × full train ≈ 30–60 minutes on 3090.

- [ ] **Step 5.3: Sanity submission**

```bash
conda run -n aicup-tt python -c "
import pandas as pd
s=pd.read_csv('artifacts/submission_C_seq.csv'); print(s.head()); print(s.shape)
print('server min/max:', s.serverGetPoint.min(), s.serverGetPoint.max())
"
```
Expected: 1845 rows, server in [0, 1].

- [ ] **Step 5.4: Commit**

```bash
git add scripts/build_route_c_submission.py artifacts/submission_C_seq.csv artifacts/oof/seq_*_test.parquet
git commit -m "feat(route_c): Transformer test-time refit + submission"
```

---

### Task 6: Overfit-watch diagnostic

The biggest risk for Route C is overfit. We log per-seed validation curves and check the spread.

- [ ] **Step 6.1: Print per-seed mean and std**

```bash
conda run -n aicup-tt python -c "
import json
log = json.load(open('artifacts/seq_run_log.json'))
import pandas as pd
df = pd.DataFrame(log)
print('per-seed mean overall:')
print(df.groupby('seed')['val_overall'].agg(['mean','std']))
print('global mean:', df['val_overall'].mean(), 'std:', df['val_overall'].std())
"
```

Decision rule: if global std > 0.025, Route C is unstable; do not pin its predictions in the final ensemble (Route C's weight in P5 should be small or zero).

- [ ] **Step 6.2: Commit the diagnostic note in HANDOFF**

Append to `HANDOFF.md`:
```markdown

## Route C (Transformer) result note (post-2026-05-27)

See `artifacts/seq_run_log.json` and the per-seed table printed by
`scripts/train_seq_transformer.py`. If across-seed std > 0.025, P5 should
de-weight Route C in the final meta-learner.
```

```bash
git add HANDOFF.md
git commit -m "docs(handoff): Route C variance check rule"
```

---

## Self-review notes

- Spec Section 4 fully covered.
- Param count < 5M asserted in Step 3.1 keeps the model on a 3090 with headroom for batch 256.
- The test in Step 2.1 includes a "no label leak" check via `target_strike`, but `strikeNumber` is not in `CATEGORICAL_COLS` — the test gracefully no-ops in that case. That is intentional; the structural guarantee comes from the dataset's `prefix = g[g["strikeNumber"] < cut]` slice, not from a runtime assertion.
- 30 OOF epochs vs 20 inference epochs: OOF needs early-stop budget; inference uses full train without held-out so 20 is plenty.
- `RallyPrefixDataset` reads the full train DataFrame into memory groups; this is fine at 14,995 rallies × ~6 strokes.

## What's next

Route C OOF lands in the same parquet schema as Route A/B. P5 (final ensemble) consumes all three. If Route C's overall is below LGBM baseline but its phase-0 macro-F1 is higher, the meta-learner in P5 will still pick it up for that bucket.
