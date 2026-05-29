# AI Cup 2026 — Private-Push v4: Parallel Multi-Bet + Public Leak Expansion (Design)

Status: Draft, awaiting user review. Date: 2026-05-30.
Author: AI agent (brainstormed with the user; web-research informed).

## 0. Context and honest framing

Production today (per `PROGRESS.md`) is the **6-base per-row ensemble + markovp**,
honest per-row overall **0.32568** (action 0.2963, point 0.1895, server 0.6567).
Public clean validated 0.3492; public leak-max
(`submission_FINAL_leakmax.csv`) scored **0.420782**.

Two prior campaigns (v2, v3) concluded the ensemble is **near its information
ceiling**: every generic new model class (seq2 Transformer, FT-Transformer,
TabPFN, XGBoost), every rebalancing scheme (cat_bal, cat_os), feature pruning,
depth tuning, and the alternative stacker (hill-climbing) landed **below the
0.00168 noise floor**. Only three levers ever cleared the gate: CatBoost
(+0.00324), markovp player-conditional features (+0.00188), and — by the prior
record — nothing else. v3-P1 (decision-rule) was a measured **false positive**
(harness used per-fold priors; production A/B rejected it).

**Honest expectation for v4 is therefore small.** This spec does not promise a
win. It runs the **best remaining bets in parallel** (compute is free on the
3090) and lets the same strict honest gate decide what ships. The new element vs
v3 is **web-research-derived**: a domain-specific neural architecture
(ShuttleNet) and a training objective (focal loss) that the prior generic
attempts never used.

User directives for this campaign:
- **Run everything in parallel** ("C"). Maximize score by every permitted means;
  pursue **both** the honest private score and the public (leak/smoothing) score,
  **private-first**.
- **Compute is not a constraint. Run GPU-capable work on the 3090.**
- **Conda only**: `conda run -n aicup-tt <cmd>`. Never touch `base`, never
  `pip install --user`. New-dep experiments use a **cloned** env (the TabPFN
  isolation lesson), never the main `aicup-tt`.

## 1. Methodology (unchanged — the honest ruler)

- **Honest per-row scoring only.** Never seed-average before scoring/stacking
  (the inflation bug; see PROGRESS "CRITICAL CORRECTION").
- **Noise floor = 0.00168** overall (across-seed std). Per-target floors:
  action 0.00525, point 0.00506, server 0.00397.
- **Gate**: a base/lever SHIPS only if its honest OOF nested-CV ensemble lift over
  the current production (0.32568) **exceeds 0.00168**. Sub-floor → REJECT, revert,
  keep scripts/parquets for reproducibility, do **not** touch
  `build_final_perrow.py` BASES.
- **Canonical builder**: `scripts/build_final_perrow.py` (per-row stack +
  prior-correction + nested-threshold scoring + distribution-matched test meta).
- **Always confirm a SHIP in the real `build_final_perrow` A/B** before trusting
  any standalone harness number (the v3-P1 lesson: the gate baseline must
  reproduce the exact production rule + prior convention).
- **Neural levers gate twice**: a cheap pilot (seed 11 × folds 0–2) first; only a
  pilot that is competitive with the GBDT bases proceeds to the full 25-fold OOF
  (the seq/FT lesson — kill non-competitive neural bets before burning compute).

## 2. Campaign structure

Five levers run **in parallel**, each independently gated. Because the user wants
**all** answers, the v3 "stop after two sub-floor phases" rule is relaxed: every
lever runs to its gate decision. Shipping is still strict (gate > floor). P5 is a
deploy-only public track that runs last and depends on the final private bases.

| Lever | What | Compute | Confidence | Primary risk |
|---|---|---|---|---|
| L1 | ShuttleNet-style stroke-forecast neural base | GPU | Medium | neural lost 3× (generic) |
| L2 | Focal-loss GBDT base (`lgbm_focal`) | CPU | Low–Med | redundant with weighted GBDTs |
| L3 | higher-order player×context Markov | CPU | High | redundant with markovp |
| L4 | structured (action, point) joint base | CPU | Medium | combiner collapse |
| L5 | public leak expansion | CPU | — | leak coverage capped |

Targets for L1–L4 are **action + point** (the two weak macro-F1 targets);
server is left to the existing bases (consistent with markovp), unless a lever
trivially also helps server.

## 3. L1 — ShuttleNet-style stroke forecasting (neural diversity base)

**Why (web research).** The closest published analog to this task is **turn-based
stroke forecasting in badminton**: predict the next stroke's **shot type**
(≈ our `actionId`) and **landing location** (≈ our `pointId`) from the rally
prefix of a two-player exchange. SOTA is **ShuttleNet** (AAAI'22,
`wywyWang/ShuttleNet`); the CoachAI Badminton Challenge 2023 (IJCAI) top teams all
built on it. References:
- ShuttleNet: <https://arxiv.org/pdf/2112.01044>
- Official impl: <https://github.com/wywyWang/ShuttleNet>
- CoachAI 2023 winner (Badminseok): <https://github.com/stan5dard/IJCAI-CoachAI-Challenge-2023>
- ShuttleSet22 benchmark: <https://arxiv.org/html/2306.15664v1>

**Why this is NOT the already-rejected seq model.** The prior seq2/FT attempts
were a single generic multi-task Transformer encoder over engineered tabular
features. ShuttleNet's competitive edge is three domain-specific inductive biases
the prior attempts lacked:
1. **Dual encoder** — separate extractors for **rally progress** (the temporal
   exchange context) and **player style** (who is hitting), instead of one
   undifferentiated encoder.
2. **Player-style embeddings** — learned per-player embedding tables, fused into
   the decode step (`gamePlayerId`/`gamePlayerOtherId` as style tokens, not just
   raw categorical features a tree already sees).
3. **Position-aware gated fusion** — a gate that fuses rally and player contexts
   conditioned on the decode position in the rally.

The pilot is therefore a fair test of a *single hypothesis*: **does the
turn-based, player-aware inductive bias add signal the GBDT ensemble lacks?** A
non-competitive pilot answers "no" cheaply and we stop (as with seq/FT).

**Design.**
- **Tokens**: each stroke in the prefix → a token with embeddings for
  `actionId, pointId, handId, strengthId, spinId, positionId, strikeId`, the
  acting player and opponent (`gamePlayerId`, `gamePlayerOtherId`), plus a
  scalar/bucketed score context (`scoreSelf`, `scoreOther`, derived rally phase).
- **Architecture**: pure-PyTorch reimplementation of the ShuttleNet pattern in
  the existing `scripts/seq_*` module family — rally-progress encoder +
  player-style encoder + position-aware gated fusion → autoregressive decode of
  the next stroke. **Heads: action (19) + point (10) only** (no server head —
  server stays with the existing bases, mirroring markovp; reduces risk/scope).
- **Loss**: class-weighted (or focal) cross-entropy on both heads, to align with
  macro-F1 rather than accuracy.
- **Why reimplement (not vendor the repo)**: keeps everything in `aicup-tt` with
  **zero new dependencies and zero env pollution** (torch 2.2 already present).
  Only if a competitive pilot specifically needs the repo's exact components do we
  fall back to a **cloned** `aicup-tt-shuttle` env — never the main env.
- **OOF + test**: reuse the proven path from the seq2 work
  (`RallyPrefixDataset`, synthetic-cut test inference, per-row OOF on the 25 cv
  cells, full-train single-seed test inference).

**Gate.** Pilot (seed 11 × folds 0–2) competitive with GBDT bases → full 25-fold
OOF → add `shuttle_{action,point}` to BASES → honest nested-CV lift vs 0.32568 >
0.00168. Otherwise REJECT (keep scripts/pilot parquets only).

## 4. L2 — Focal-loss GBDT base (`lgbm_focal`)

**Why (web research).** We tried `sqrt` and full `balanced` sample weights
(cat_bal/cat_os, both rejected) but **never the focal-loss objective**. Focal loss
down-weights easy/majority samples *in the gradient* rather than reweighting
counts, and the literature reports it beats weighted loss for **macro-F1 on
imbalanced multiclass** and is more robust to its parameter than class weights.
Ref: <https://maxhalford.github.io/blog/lightgbm-focal-loss/>,
<https://arxiv.org/pdf/1908.01672> (Imbalance-XGBoost).

**Design.** LightGBM with a **custom multiclass focal `fobj`/`feval`** (softmax
focal: `-(1-p_t)^γ · log p_t`, gradient+hessian implemented), reusing the existing
LGBM OOF pipeline (`produce_base_oof.py` path). New base `lgbm_focal` for
**action + point**. γ swept over **{1.0, 2.0} only** (two honest comparisons, not
a grid — avoid ruler-overfit). CPU is fine (≈12k rows/fold, below GPU break-even,
per the GPU policy).

**Gate.** Standalone honest score reported; add to BASES; nested-CV lift vs
0.32568 > 0.00168, with a redundancy check vs the existing GBDT bases (a base
that's highly correlated with `cat`/`lgbm15` adds nothing — the depth-8 lesson).

## 5. L3 — higher-order player×context Markov (= v3-P2)

Unchanged from `2026-05-29-aicup-private-push-v3-design.md` §4. Hierarchical
Dirichlet backoff for P(next | context):
global → last1 → **last-2-gram** → (player, last1) → **(player, last-2-gram)**,
plus a **score-context-conditioned** variant (deuce / close / lead bucket).
OOF-safe (train-fold counts only), Dirichlet α tuned as markovp did. Targets:
action + point. New base; gate vs 0.32568; watch redundancy with markovp.

## 6. L4 — structured (action, point) joint base (= v3-P3)

Unchanged from v3 §5. OOF-safe smoothed P(point | action) (optionally also
phase/player with backoff), marginalized over the ensemble's predicted action
distribution: `P(point) = Σ_a P(point | a) · P̂(a)`. Added as a new point base
(optionally a symmetric `P(action | point)` rerank). **MUST be evaluated with the
full downstream pipeline** (prior-correction + nested thresholds), never
argmax-only (the hill-climb collapse lesson). Gate vs 0.32568.

## 7. L5 — public leak expansion (deploy-only, runs last; = v3-P5)

Depends on the final post-L1–L4 private bases.
- Verify the old-test overlap is exactly 1236/1845 rallies (or find more leakable
  rallies) to maximize coverage of the permitted serverGetPoint leak.
- Reconfirm action has no usable outcome-leak (+0.0008 measured → treat as dead).
- **Rebuild leak-max on top of the improved private base**: re-derive point with
  the best post-L4 model + the leak serverGetPoint feature, override point on the
  overlap rallies, keep server smoothing. Target: **beat 0.420782**.
- The honest private submission (`submission_FINAL_safe_perrow.csv`) stays
  **leak-free**; leak only touches the public variant.

## 8. Parallel execution plan

Compute is free, so the four private levers run concurrently on the box:
- **GPU lane**: L1 pilot (then full OOF if competitive).
- **CPU lanes**: L2 focal-LGBM OOF, L3 Markov counts, L4 joint counts — all
  independent producers writing `artifacts/oof/<base>_<target>{,_test}.parquet`.
- **Integration gate**: once producers finish, a single
  `build_final_perrow` A/B per candidate base (and a combined run for the survivors)
  measures honest nested-CV lift vs 0.32568. Only survivors enter BASES.
- **L5** runs after the private BASES are locked.

No two levers share mutable state; each is its own script + parquet output, so
parallel runs cannot corrupt each other.

## 9. Deliverables and tracking

- Each lever: a producer/scorer script + a `PROGRESS.md` entry recording
  ship/reject + honest lift vs 0.32568 (same table format as v2/v3).
- Production changes only via `build_final_perrow.py` BASES (L1–L4) or, for public,
  the leak-max builder (L5). Rejected artifacts kept for reproducibility, NOT in
  BASES.
- `submission_FINAL_safe_perrow.csv` (private, leak-free) regenerated only if a
  lever ships. `submission_FINAL_leakmax.csv` (public) regenerated only after the
  private bases are locked.
- The honest per-row overall is the single ruler for every ship decision.

## 10. Out of scope

- Seed-averaged scoring (banned — inflates).
- Re-trying already-rejected levers **unchanged**
  (seq2/TabPFN/XGBoost/FT/hill-climb/rebalancing/pruning/depth-8). L1 is in scope
  only because it is an architecturally distinct, domain-specific neural model, and
  L2 only because focal loss is a distinct objective never tried.
- LB-probing of the public leaderboard (COI caution; user did not select it).
- Any change validated only on public LB or only on argmax.
- A server-prediction head on L1 (kept out to reduce scope/risk; server stays with
  the existing bases).
