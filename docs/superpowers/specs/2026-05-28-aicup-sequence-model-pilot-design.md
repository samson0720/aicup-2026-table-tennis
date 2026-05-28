# AI Cup 2026 — Sequence-Model Pilot toward 0.5 (Design)

Status: Draft for user review (2026-05-28).
Builds on `2026-05-27-aicup-score-improvements-design.md` and the per-row
scoring fix committed in `e81d013`.

## 1. Goal and honest feasibility

The user wants to push the leaderboard score toward **0.5**. Current honest
state (per-row ruler, the only trustworthy one after the seed-averaging fix):

- Honest ensemble local overall = **0.3206** (action 0.2894, point 0.1842, server AUC 0.6557).
- Public clean = **0.3492** (validated; local→public offset ≈ +0.029).
- Public smooth (old-test `serverGetPoint` trick) ≈ 0.41–0.42 expected for the new ensemble.

Metric: `overall = 0.4·action_macroF1 + 0.4·point_macroF1 + 0.2·server_AUC`.

**Target framing (agreed with user):**
- **public-smooth 0.5 is the sprint target** and is *plausible* if a strong
  sequence model lifts action toward ~0.38 and point toward ~0.26. Back-of-envelope:
  clean local ~0.40 → public clean ~0.43 → +smoothing ≈ 0.51.
- **clean/private 0.5 is a stretch we likely cannot reach** (rallies are short —
  median 5 strokes, mean 5.6 — so next-stroke prediction has high irreducible
  uncertainty). We optimize private honestly and treat 0.5 as the public-smooth aim.

The biggest lever is **action macro-F1** (0.4 weight, currently 0.29, most
headroom). point (0.4 weight, 0.18) is weaker-ceiling. server (0.2 weight) is
gamed by smoothing for public and has limited honest headroom.

## 2. Why a sequence model, and why it underperformed

Route C (the multi-task Transformer in `scripts/seq_model.py` /
`scripts/train_seq_transformer.py`) was rejected at 0.2400 overall, but it was
**undertrained, not under-featured**:

- Per-stroke inputs already cover every meaningful column (strikeId, handId,
  strengthId, spinId, pointId, actionId, positionId, gamePlayerId/Other, sex,
  scoreSelf/Other). Feature coverage is essentially complete.
- The trainer has **no LR schedule, no early stopping, no gradient clipping**,
  a fixed epoch count, and dropout 0.3. The 2-epoch reject scored 0.24; a single
  10-epoch / 1-fold pilot already lifted that fold's action from 0.23 → 0.29
  (competitive with LGBM's 0.26). It was never trained to convergence.

Hypothesis: **a properly trained version reveals a materially higher action
ceiling, and adds ensemble diversity vs the tabular LGBM bases.**

## 3. Phase 1 — Pilot (build this first; ~1–2 days)

**Objective:** measure the true per-fold ceiling of a properly-trained sequence
model on action/point, scored on the honest per-row pipeline, against LGBM on
the same folds — before committing multi-day GPU.

### 3.1 Training improvements (new training path; do NOT delete the existing one)

Add to the sequence trainer:
1. **Cosine LR schedule with linear warmup** (warmup ~5–10% of steps).
2. **Early stopping** on validation combined score
   `0.4·action_F1 + 0.4·point_F1 + 0.2·server_AUC` (macro-F1 via the honest
   postproc below), restoring the best checkpoint. Budget up to ~60–80 epochs.
3. **Gradient clipping** (max_norm = 1.0).
4. **Light hyperparameter sweep** (small grid, not exhaustive): dropout ∈ {0.1, 0.2},
   d_model ∈ {128, 192, 256}. Keep `MAX_LEN=24` (covers the long tail; median len 5).

### 3.2 Honest scoring (reuse the per-row pipeline)

Score the pilot's validation OOF exactly like the ensemble:
- action/point: `prior_correct` + **nested** threshold tuning (tune on train
  folds, apply on held-out) — never in-sample.
- server: AUC directly.
- Compare to the LGBM base (`lgbm15`) on the **same** (seed, fold) slice, on the
  per-row population. No seed-averaging anywhere.

### 3.3 Pilot slice

Run **seed 11 × folds 0–2** (3 folds). Enough signal for a stable read, hours
not days. Use `env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt ...` (3090 is
PyTorch device 0; see PROGRESS GPU notes).

### 3.4 Decision gate

- **GREEN** → proceed to Phase 2 if either:
  - seq action macro-F1 ≥ ~0.32 on held-out (clearly above LGBM ~0.26), or
  - a per-row stack of {existing bases + seq} beats the existing-bases-only
    per-row stack by **> 0.00168** (noise floor) on the pilot slice.
- **YELLOW** → if seq merely ties LGBM (~0.26–0.29) with no ensemble lift: the
  seq model is only a diversity base, not a 0.5 path. Stop, report, and
  re-brainstorm (richer engineered features, different target, or accept
  incremental gains). Do NOT burn multi-day GPU.

## 4. Phase 2 — Full integration (only if Phase 1 is GREEN; multi-day GPU)

1. Train full **25-fold OOF** (5 seeds × 5 folds) with the converged config;
   write `artifacts/oof/seq2_{action,point,server}.parquet` (per-row schema).
2. Train full-train inference model; write `seq2_{target}_test.parquet`
   (single-cut per rally, distribution-matched to test).
3. Add `seq2` as a base in `scripts/build_final_perrow.py`; rebuild the per-row
   ensemble; record the honest overall and the lift vs the current 0.3206.
4. Regenerate `submission_FINAL_safe_perrow.csv` (private bet) and
   `submission_FINAL_smooth_perrow.csv` (public sprint).
5. Decide submission by the honest per-row ruler; reserve at most one public
   upload to confirm (uploads are daily-limited and shared with teammates).

## 5. Success criteria

- **Pilot success** = a clear GREEN/YELLOW verdict with honest numbers, not a
  specific score. The pilot's job is information, not a final model.
- **Phase 2 success** = honest per-row ensemble overall lifts by > noise floor
  (0.00168) over 0.3206, ideally toward ~0.36–0.40 to make public-smooth 0.5
  reachable.

## 6. Risks and constraints

- Short sequences cap the seq-model edge; the pilot exists precisely to test this.
- conda env `aicup-tt` only; never base/system Python. 3090 = PyTorch device 0.
- Keep the existing `train_seq_transformer.py` intact (it is the smoke-tested
  reference); add the improved trainer alongside it.
- Public LB is scarce (daily limit, shared with teammates) — judge everything on
  the per-row local ruler; upload only to confirm a final pick.

## 7. Out of scope (YAGNI)

- New raw features beyond what's already in the per-stroke representation
  (coverage is already complete).
- Optuna/large HPO (a small targeted sweep only).
- Replacing LGBM/chain bases — the seq model is *added*, not a replacement.
