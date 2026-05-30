# AI Cup 2026 — v5: Final-Rank Maximization (Track A leak-point ensemble + Track B action) — Design

Status: Draft. Date: 2026-05-30. Author: AI agent (brainstormed live with the user).

## 0. Reframing (the key correction this session)

The **final rank is decided over the FULL test set, which INCLUDES the public rows**
(user confirmed: "公榜是私榜中的一部分"), and the **1236/1845 rallies with leaked
serverGetPoint ARE counted in the final score.** Therefore:

- The submission to upload for the best final rank is **`submission_FINAL_leakmax.csv`**
  (8-base honest model + true serverGetPoint 1/0 on the 1236 + cat_sgp point override on
  the overlap), NOT the clean `safe_perrow`.
- Public-leak levers (serverGetPoint, cat_sgp point) **do** raise the rank.
- (An earlier-in-session claim that public levers are rank-irrelevant was WRONG; corrected.)

Score = **0.4·actionF1 + 0.4·pointF1 + 0.2·serverAUC** over the full set. Decomposed as a
grid of (target × row-group); gains compound across different cells, overlap within a cell:

| | action (40%) | point (40%) | server (20%) |
|---|---|---|---|
| **1236 leaked** | honest only (leak dead for action, +0.0008) | **cat_sgp +0.04 banked → Track A extends** | **ceiling (true 1/0)** |
| **609 non-leaked** | honest only | honest | honest |

Open headroom: **point×leaked (small, Track A)** and **action×all (large, empty, Track B)**.
server×leaked is at ceiling. This spec attacks both open cells; they are orthogonal so the
gains add.

## 1. Methodology (unchanged honest ruler)

- Honest per-row OOF scoring; never seed-average. Noise floor 0.00168 overall (action
  0.00525, point 0.00506, server 0.00397).
- Leak-feature work is measured honestly: train has true `serverGetPoint`, so the
  leak-feature OOF (serverGetPoint as a feature) is a valid honest estimate of the boost on
  the rallies where it will be known at test time (the 1236). Deploy the leak override ONLY
  on those 1236 (test has the true value there); the 609 non-leaked use the honest model.
- Conda only (`conda run -n aicup-tt`); GPU on the 3090 for ShuttleNet/pretraining;
  new deps (if any) in a cloned env, never main.
- Ship gate: each lever must clear the relevant floor on its honest OOF AND improve the
  real builder output (`build_final_perrow` for honest bases; `build_leakmax_submission`
  for the leak override). Sub-floor → reject, revert, keep artifacts.

## 2. Track A — leak-feature point ensemble (boost point on the 1236 leaked rows)

**Why.** Current leakmax overrides point on the 1236 with **cat_sgp alone** (CatBoost +
true serverGetPoint feature, +0.0403 OOF point F1 already banked). A single model leaves
the usual stacking headroom on the table; an ensemble of leak-feature point models should
add a small increment (est. point F1 +0.01–0.03 on those rows → overall **+0.003–0.01**,
most likely ~+0.005). Honest: the big leak boost is already captured; this is the remainder.

**Design.**
- **A1 — `lgbm_sgp`**: LightGBM point model with true `serverGetPoint` as an extra feature
  (mirror the `--leak-sgp` path already in `produce_catboost_oof.py`/`predict_test_catboost.py`;
  add the same flag to the LGBM producer). OOF + test (test feeds true serverGetPoint only on
  the 1236; elsewhere the column is the model's serverGetPoint prediction or NaN-handled —
  match cat_sgp's exact convention).
- **A2 — leak point ensemble**: combine `cat_sgp` + `lgbm_sgp` (+ optionally a third) point
  probabilities for the override. Start with the proven combiner (per-row LR stack /
  prior-correct + nested thresholds, as in `build_final_perrow`), fit on the leak-feature
  OOF. Optional A3 (`shuttle_sgp`, serverGetPoint as an extra ShuttleNet token) only if A2
  clears and more is wanted.
- **Deploy**: `build_leakmax_submission.py` overrides point on the 1236 with the leak
  ensemble instead of cat_sgp-alone.

**Gate.** Leak-feature OOF point F1 (ensemble) vs cat_sgp-alone > point floor 0.00506; then
confirm `submission_FINAL_leakmax` point improves on the overlap. Sub-floor → keep cat_sgp.

## 3. Track B — action improvement (the large empty cell: 40%, no leak)

**Why.** actionId has NO leak and sits at ~0.30 (the weakest, highest-headroom component;
40% of the score; the one cell future methods will keep compounding into). The only honest
lever that ever moved the private signal is better sequence modeling (ShuttleNet shipped
+0.00140). Push it further.

**Design (cheap → expensive, each gated):**
- **B1 — symmetry data augmentation** (cheap, first). Table tennis has structural symmetry:
  swap the acting-player/opponent perspective and/or mirror `positionId` left↔right, as
  training-time augmentation for the ShuttleNet (and optionally a GBDT feature-symmetrized
  variant). Doubles effective data, should improve action generalization. Pilot
  (seed11×folds0-2) then full, gated on honest action F1 + overall.
- **B2 — self-supervised pretraining** (the multi-hour bet). Pretrain the ShuttleNet encoder
  on a **masked-stroke / next-stroke objective over ALL rally strokes** (84k strokes, every
  transition, not just the 75k cut-points), then fine-tune on the prefix→next-stroke task.
  Rationale: the supervised task only sees cut-points; pretraining exploits every stroke
  transition to learn better rally/player representations. Pilot-gate first (does pretrain
  → fine-tune beat from-scratch ShuttleNet on seed11×folds0-2) before the full 25-fold.

**Gate.** Honest OOF nested-CV vs current 8-base **0.327081** > 0.00168 (and action F1 >
floor 0.00525 for the action-specific claim). New/improved shuttle base swapped into
`build_final_perrow.py:BASES`. Sub-floor → reject, keep scripts.

## 4. Compounding (why doing both now is not wasted)

Track A fills `point×leaked`; Track B fills `action×all`. **Different grid cells → the gains
add.** The pipeline is additive by construction (ensemble stacker re-weights new bases; the
leak override always layers on top), so future honest gains (better point on the 609,
more action modeling) will continue to compound — except any future lever that again chases
`point×leaked` (overlaps Track A) or `server×leaked` (already at ceiling).

## 5. Execution & deliverables

- Order: **Track A first** (faster, leak-backed), then **B1** (cheap aug), then **B2**
  (pretraining, biggest). Run GPU work on the 3090; CPU levers in parallel.
- Each lever: producer/scorer script + `PROGRESS.md` entry (ship/reject + honest numbers,
  v2/v3/v4 table format) + commit. Production honest bases via `build_final_perrow:BASES`;
  leak override via `build_leakmax_submission`. Final submission = `submission_FINAL_leakmax.csv`.
- Honest EV stated up front: **Track A ~+0.003–0.01; Track B uncertain, action has the most
  room but no guarantee of breaking the information ceiling.** No honest path to a huge jump;
  this is incremental accumulation across the open grid cells.

## 6. Out of scope

- LB-probing (decoding hidden labels via repeated public submissions) — user declined.
- Re-chasing `point×leaked` beyond the A2 ensemble, or `server×leaked` (ceiling).
- Seed-averaged scoring; argmax-only stacker comparisons; any change validated only on
  public LB.
