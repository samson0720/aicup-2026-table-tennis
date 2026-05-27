# AI Cup 2026 Table Tennis — Score Improvement Design

Status: Draft, awaiting user review
Date: 2026-05-27
Goal: Improve private-leaderboard score on AI Cup 2026 table-tennis next-stroke prediction.
Out of scope: Hyperparameter tuning (deferred to final phase, see Section 5.4).

## 0. Context

### Task
For each rally in `test_new.csv`, given a prefix of strokes, predict three targets:
- `actionId` — next stroke action (19 classes), scored by macro-F1
- `pointId` — next stroke landing zone (10 classes), scored by macro-F1
- `serverGetPoint` — rally winner is server (binary), scored by AUC

Local proxy metric: `0.4 * action_macro_F1 + 0.4 * point_macro_F1 + 0.2 * server_AUC`.

### Current state
- Best public score `0.4102` (rank ~45/333), uses old-test overlap smoothing (leakage trick).
- Clean (no leakage) public score `0.3263` (rank ~157/335).
- Local OOF (LightGBM, leaves=15, GroupKFold by match):
  - `action_macro_F1 = 0.2948`
  - `point_macro_F1  = 0.2154`  <- weakest target
  - `server_AUC     = 0.6016`  <- near random in phase 0
  - `overall        = 0.3244`
- Phase breakdown shows phase 0 (first return after serve) is near random;
  larger prefix lengths perform best.

### Pain points
1. Local CV does not predict public LB (HANDOFF.md L74-76).
2. Public LB extremely sensitive to old-test smoothing.
3. `argmax` decision rule loses on macro-F1 with class imbalance.
4. Three targets trained independently; no joint signal.
5. CatBoost too slow on laptop. NOTE: a 3090 GPU is now available.

### Overlap analysis (train vs test_new)
- 216 train matches, 79 test matches, **0 match overlap** → match-aware fold grouping is required.
- 166 train players, 71 test players, **40 player overlap, 31 test-only players** →
  Route B target encoding MUST have an unseen-player fallback (smoothing toward the
  global prior). Models cannot assume every test player has historical stats.

### User goals and constraints
- Optimize private leaderboard, not public.
- GPU available (3090); install GPU-aware LightGBM and PyTorch.
- Conda environment must be isolated; system environment not touched.
- Hyperparameter tuning happens last.

## 1. Environment and Validation Foundation

### 1.1 Conda environment
- Env name: `aicup-tt`
- Python 3.11
- Pinned packages in `environment.yml` checked into repo:
  - `pandas`, `numpy`, `pyarrow`, `scikit-learn`
  - `lightgbm` built with GPU support
  - `xgboost` (ensemble diversity, GPU-capable)
  - `pytorch` with CUDA 12.1 for the 3090
  - `optuna` (used only in Section 5.4)
  - `tqdm`, `matplotlib` for diagnostics
- All commands run inside `conda activate aicup-tt`. No `pip install --user`,
  no system Python, no edits to existing envs.

### 1.2 Private-safe cross validation
The existing `GroupKFold by match` mismatches the public/private setup,
because `test_new.csv` is constructed by taking mid-rally cut points,
not by holding out whole matches.

New CV scheme (foundation for every experiment below):

1. Per-rally random cut point that mimics `test_new` prefix structure.
2. Stratification key: phase bucket (`target_strikeNumber in {2}`, `{3}`, `{>=4}`)
   to keep phase ratios constant across folds.
3. Match-aware grouping: all rallies of a match share the same fold to
   prevent player leakage.
4. 5 seeds x 5 folds. Any improvement smaller than the across-seed standard
   deviation is rejected as noise.

Output artifact: `artifacts/cv_splits.parquet` written once by
`scripts/build_cv_splits.py`. Every later script reads the same file so
results are comparable.

### 1.3 Smoothing trick policy
- Keep `submission_RECOMMENDED_upload.csv` on disk as a public backup.
- All new model outputs forbid old-test smoothing during training and OOF.
- Two final submissions decided in Section 5.2.

## 2. Route A — Post-processing and Stacking

Low-risk, fast wins by changing only inference and adding a meta-learner.

### 2.1 Prior-corrected prediction for macro-F1
`argmax p(c)` loses on macro-F1 when classes are imbalanced. Use
`score(c) = log p(c) - log pi(c)` where `pi(c)` is the train prior.
Then optional per-class threshold grid search on OOF, objective = macro-F1.

Applies to `actionId` (19 classes) and `pointId` (10 classes) independently.
Inference-only; no retraining.

### 2.2 Multi-seed bagging
Average OOF probability across 5 seeds. Arithmetic mean of probabilities,
not logits — empirically more stable for macro-F1 on this dataset.
Hooks into `private_safe_report.py` outputs.

### 2.3 Multi-base stacking
Collect OOF probabilities from existing experiments:
- LGBM leaves=15 (5-seed averaged)
- LGBM leaves=31 (5-seed averaged)
- Markov ensemble
- Phase-specific LGBM
- Player-stats LGBM (if recoverable; otherwise rebuild OOF-safe)

Train a per-target meta-learner (multinomial logistic regression with L2)
on OOF probabilities. Meta-learner itself is fit under an extra round of
GroupKFold to avoid in-bag overfitting.

### 2.4 Phase-aware server blending
Server-AUC in phase 0 (prefix length 1) is ~0.53. The model has no signal
to extract. Replace with a `(serverPlayerId, receiverPlayerId)` historical
serve win rate prior:
- Phase 0: `0.3 * p_LGBM + 0.7 * p_prior`
- Phase 1: `0.6 * p_LGBM + 0.4 * p_prior`
- Phase 2+: `1.0 * p_LGBM`

Weights selected by OOF grid search.

Prior computation MUST be OOF-safe: each fold computes prior from fold-out
train only. Assert this in code.

### 2.5 Artifacts
- `scripts/postprocess_calibrate.py`
- `scripts/stacker.py`
- `artifacts/stack_oof.parquet`
- `artifacts/submission_A_stacked.csv`

### 2.6 Expected gain
overall **+0.015 to +0.030** on private-safe CV.

## 3. Route B — Feature Engineering and Chain Multi-task

### 3.1 OOF-safe target encoding
Keys: `(gamePlayerId)`, `(gamePlayerId, phase)`, `(gamePlayerId, gamePlayerOtherId)`.
Encoded features per key:
- Historical frequency of each `actionId` (19 dim)
- Historical frequency of each `pointId` (10 dim)
- Player serve win rate (scalar)
- Player-vs-opponent serve win rate (scalar)

Smoothing: `(count * rate + alpha * global) / (count + alpha)`, alpha=20.

Strict OOF: each fold computes encodings from fold-out train only,
reading the cv_splits.parquet from Section 1. Code-level assertion:
`assert not encoding_uses_valid_rows`.

### 3.2 Sequence n-gram features
- `(action[t-1], action[t-2])` bigram integer hash
- `(spin[t-1], action_target)` transition entropy from train statistics
- Prefix action repetition count
- Longest run of identical action in prefix
- StrikeId switch frequency

### 3.3 Test_new prefix semi-supervised stats
`test_new` prefix rows contain known feature columns (no label).
Concatenate train + test_new for computing player-level feature
distributions only (no label use). This grows the sample size for
target encoding without leakage, because no test label is touched.

### 3.4 Chain prediction
```
LGBM_action(X)                              -> p_action  (OOF, 19 dim)
LGBM_point(X concat p_action)               -> p_point   (OOF, 10 dim)
LGBM_server(X concat p_action concat p_point) -> p_server (OOF, scalar)
```

`p_action` and `p_point` are OOF predictions (each fold predicted by a
model trained on the other folds), not in-bag predictions. This is the
only way to avoid leakage into downstream targets.

### 3.5 Artifacts
- `scripts/build_features_v2.py` produces `artifacts/prefix_train_v2.parquet`
- `scripts/train_chain_lgbm.py`
- `artifacts/chain_oof.parquet`
- `artifacts/submission_B_chain.csv`

### 3.6 Expected gain
overall **+0.015 to +0.030** on private-safe CV, independent of Route A.

## 4. Route C — Sequence Model on 3090

### 4.1 Architecture
Small Transformer encoder:
- Token per stroke = concat of embeddings for `strikeId`, `handId`,
  `strengthId`, `spinId`, `pointId`, `actionId`, `positionId`, plus a
  small projection of score state and player embeddings.
- 4 encoder layers, 4 heads, `d_model = 128`, FFN 512.
- Three multi-task heads:
  - `action_head` -> 19-way softmax
  - `point_head` -> 10-way softmax
  - `server_head` -> 1-way sigmoid
- Pooling: take the last unmasked token's hidden state.

### 4.2 Training
- Loss: `0.4 * CE_action + 0.4 * CE_point + 0.2 * BCE_server`, mirroring
  the metric.
- Class-balanced loss for the two multi-class targets, helps macro-F1.
- batch=256, AdamW, lr=3e-4, cosine schedule, 30 epochs, early stop on
  private-safe valid macro-F1.
- Mixed precision (fp16) on the 3090. Estimated 5-10 minutes per fold.
- 5 seeds x 5 folds. Per-fold OOF stored.
- Regularization: dropout 0.3, label smoothing 0.1.

### 4.3 Ensemble role
The Transformer does NOT replace LGBM. It joins as a third base model
in the Section 2.3 / Section 5 stack. The expectation is that the
sequence model is stronger on phase 0/1 (short prefixes) while LGBM
dominates phase 2+. Stacking captures this phase-conditional weighting.

### 4.4 Artifacts
- `scripts/seq_dataset.py`
- `scripts/train_seq_transformer.py`
- `artifacts/seq_oof.parquet`
- `artifacts/submission_C_seq.csv`

### 4.5 Expected gain
overall **+0.015 to +0.040** on private-safe CV, high variance.
May be zero in worst case (overfit on 14,995 rallies).

## 5. Final Ensemble and Submission Strategy

### 5.1 Final stack
Inputs (raw OOF probabilities, not the intermediate stacker from 2.3):
- 5 base OOFs from Route A's pool (LGBM leaves=15, LGBM leaves=31, Markov,
  phase-LGBM, player-stats LGBM), each already seed-averaged per 2.2
- Route B chain OOF (action, point, server heads)
- Route C seq OOF (action, point, server heads)

Total ~21 probability columns per row across the three targets. The
intermediate stacker from 2.3 is kept as a diagnostic artifact and as
the Route-A standalone submission, but is NOT chained into the final
stack to avoid double-stacking the same signal.

Final per-target meta-learner = multinomial logistic regression with L2,
fit under one more GroupKFold round (meta CV) using the cv_splits.parquet
from Section 1.2.

### 5.2 Final submissions
Competition allows multiple submissions but typically two are picked for
private evaluation. Plan:
1. `submission_FINAL_safe.csv` — pure model ensemble, no leakage trick.
   Bet on private leaderboard.
2. `submission_FINAL_smooth.csv` — same ensemble + old-test smoothing,
   public-leaderboard backup.

### 5.3 Guardrails
- Any experiment with private-safe CV mean improvement smaller than one
  across-seed standard deviation is rejected.
- Any feature whose computation reads valid-fold rows is rejected.
  Code-level assertions enforce this.
- Final ensemble weights are learned on valid OOF only. Public LB is NOT
  used to tune weights.
- Smoothing logic isolated to a single function that runs only when
  building `submission_FINAL_smooth.csv`.

### 5.4 HPO (last)
After A + B + C land and the stack is stable, run Optuna with
`n_trials = 50-100`, objective = private-safe overall metric, on the
strongest base learner. Expected additional gain
**+0.003 to +0.010**.

## 6. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| target encoding leakage | OOF-safe builder with assertion; same cv_splits used everywhere |
| Transformer overfit on 14,995 rallies | strong dropout / label smoothing / 5 seeds; ensemble role not replacement |
| CV-LB gap persists | private-safe CV redesign in Section 1.2 first; rerun a known baseline through new CV to verify the gap narrows before investing |
| GPU LightGBM build issues | fall back to CPU LightGBM; Section 4 is the GPU-critical path, not Section 2/3 |
| Conda env drift | `environment.yml` pinned and committed; rebuild from file is the only supported path |

## 7. Out of Scope (explicit)

- Hyperparameter tuning beyond Section 5.4
- CatBoost (slow on prior laptop; not worth re-evaluating on 3090 in the
  initial pass — diversity from Transformer is higher)
- AutoML platforms
- Public LB chasing beyond keeping the existing smoothed submission as a backup
