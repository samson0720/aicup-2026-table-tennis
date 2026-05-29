# AI Cup 2026 — Private-Push v3 + Public Leak Expansion (Design)

Status: Draft, awaiting user review. Date: 2026-05-29.
Author: AI agent (brainstormed with the user).

## 0. Context and honest framing

Production today (per `PROGRESS.md`) is the **6-base per-row ensemble + markovp**,
honest per-row overall **0.32568** (action 0.2963, point 0.1895, server 0.6567).
Public clean validated 0.3492; public leak-max (`submission_FINAL_leakmax.csv`)
uploaded and scored **0.420782**.

The prior two campaigns concluded the ensemble is **near its information ceiling**:
every new model class (seq2 Transformer, TabPFN, XGBoost, FT-Transformer),
rebalancing (cat_bal/cat_os), feature pruning, and alternative stacker
(hill-climbing) landed **below the 0.00168 noise floor**. Only two levers ever
cleared the gate: CatBoost (+0.00324) and markovp player-conditional features
(+0.00188). **Expected yield from v3 is therefore small.** This spec is honest
about that: the two highest-confidence bets are P1 (the decision rule is provably
truncated and benefits every target + both leaderboards) and P2 (extend the one
mechanism — markovp — that already cleared the gate). P3/P4 are more speculative
and exist to be killed cheaply by the gate.

User directives for this campaign:
- Maximize score by every permitted means; pursue **both** the honest private
  score and the public (smoothing/leak) score.
- **Compute is not a constraint. Run GPU-capable work on the 3090.**
- Conda only: `conda run -n aicup-tt <cmd>`. Never touch `base`, never
  `pip install --user`. New-dep experiments use a **cloned** env (per the TabPFN
  isolation lesson), never the main `aicup-tt`.

## 1. Methodology (unchanged from v1/v2 — the honest ruler)

- **Honest per-row scoring only.** Never seed-average before scoring/stacking
  (the inflation bug; see PROGRESS "CRITICAL CORRECTION").
- **Noise floor = 0.00168** overall (across-seed std). Per-target floors:
  action 0.00525, point 0.00506, server 0.00397.
- **Gate**: a phase SHIPS only if its honest OOF nested-CV ensemble lift over the
  current production (0.32568) **exceeds 0.00168**. Sub-floor → REJECT, revert,
  keep scripts/parquets for reproducibility, do not touch `build_final_perrow.py`
  BASES.
- **Canonical builder**: `scripts/build_final_perrow.py` (per-row stack +
  prior-correction + nested-threshold scoring + distribution-matched test meta).
- **Decision-rule experiments (P1) use nested-CV + a held-out seed** to rule out
  in-sample tuning artifacts (the hill-climb lesson: always compare with the FULL
  downstream pipeline, never argmax-only).

## 2. Campaign structure

Five phases, ordered by EV/effort. **Stop rule: after two consecutive phases land
sub-floor, stop the private push** (respect the information-ceiling finding; don't
burn compute indefinitely). P5 is a deploy-only public track that runs last and
depends on the final private bases.

| Phase | Lever | Retrain? | Confidence | Primary risk |
|---|---|---|---|---|
| P1 | macro-F1 decision rule | No | High | decision-rule overfit |
| P2 | higher-order player×context Markov | New base (cheap) | High | redundant with markovp |
| P3 | structured (action,point) joint | New base | Medium | combiner collapse |
| P4 | next-stroke attribute multi-task aux | New base(s) | Low | neural lost 3× before |
| P5 | public leak expansion | No (deploy) | — | leak coverage capped |

## 3. P1 — macro-F1 decision-rule optimization

**Why.** `postprocess.py:tune_thresholds` is a per-class **additive** threshold,
grid hard-capped at **±0.10** (9 values), greedy **2 passes only**. Calibration is
only `prior_correct` (divide by prior). macro-F1 is a **set-level** metric — the
optimal per-sample decision depends on the whole set's class frequencies — so an
additive threshold on lightly-calibrated probs is a crude proxy. Historically the
biggest lever lived in post-processing (argmax 0.24 → tuned 0.32). This lever
needs **no retraining** and improves **action + point on both leaderboards**.

**Sub-experiments (each gated nested-CV vs 0.32568):**
- **P1a — uncap + converge.** Widen grid to ≈ ±0.30 with a finer step; run
  coordinate ascent to full convergence (not 2 passes). Applies to action (19
  classes) and point (10 classes).
- **P1b — calibrate then threshold.** Per-class isotonic / temperature scaling on
  the meta-stacker probabilities (replacing or composed with `prior_correct`),
  then tune thresholds.
- **P1c — frequency-quota assignment.** Exploit the set-level nature of macro-F1:
  tune a per-class predicted-count quota to maximize OOF macro-F1, assign labels by
  within-class probability rank. Compare against the additive-threshold rule.

**Overfit guard.** All three tuned on the inner folds and scored on the outer
fold, plus a final seed-55 holdout sanity check (fit/select on seeds 11/22/33/44).
A finer grid has more capacity to overfit; the holdout is the kill switch.

**Ship rule.** Adopt the best decision rule only if outer-fold *and* holdout both
beat the current rule by > floor. P1 changes the scorer/builder, not the bases.

## 4. P2 — higher-order player×context Markov (new base)

**Why.** markovp (`produce_markovp_oof.py`) is the only v2-era idea that cleared
the gate (+0.00188), conditioning on (phase, last-stroke, next-player). It proved
explicit smoothed conditionals supply signal the tree can't estimate for sparse
combos. Extend the **same proven mechanism**, deeper.

**Design.** Hierarchical Dirichlet backoff chain for P(next | context):
global → last1 → **last-2-gram** → (player, last1) → **(player, last-2-gram)**,
plus a **score-context-conditioned** variant (e.g. deuce / close / lead bucket).
OOF-safe (counts from train folds only), Dirichlet alpha tuned as markovp did.
Targets: action + point (server conditioning unclear — leave as-is, like markovp).
Compute is trivial (counting); not GPU work but runs in `aicup-tt`.

**Gate.** New base added to `build_final_perrow.py` BASES, OOF nested-CV lift vs
0.32568 > 0.00168. Watch for redundancy with existing markovp (correlated bases
add nothing — the depth-8 CatBoost lesson).

## 5. P3 — structured (action, point) joint prediction (new base)

**Why.** pointId depends strongly on actionId. The chain (Route B) feeds action
OOF probs into the point model, but never models the **explicit joint
distribution**. New signal = the marginalized joint, not another chain.

**Design.** Build a smoothed conditional P(point | action) (OOF-safe counts,
optionally also conditioned on phase/player with backoff). Marginalize over the
predicted action distribution: P(point) = Σ_a P(point | a) · P̂(a), where P̂(a)
is the ensemble's action probability. Add the resulting point distribution as a
new base (and optionally a symmetric P(action | point) rerank).

**Risk + guard.** Hill-climb collapsed under threshold tuning while looking great
on argmax. **P3 MUST be evaluated with the full downstream pipeline**
(prior-correction + nested thresholds), never argmax-only. Gate vs 0.32568.

## 6. P4 — next-stroke attribute multi-task aux (new base[s])

**Why.** The next stroke's handId/strengthId/spinId/positionId are observable in
**train** (they are columns of the target stroke, just not scored). "What kind of
stroke is coming" is informative for action/point. Compute is free, so two bets in
parallel:

- **P4a — GBDT aux-target stacking** (dodges the "neural loses on these features"
  lesson). Train CatBoost/LGBM to predict next-stroke spin / strength / position
  (low cardinality, easier). Feed their OOF probabilities as features/bases into
  the main action/point stack. GPU CatBoost (`gpu_ram_part=0.45` per the v2 hang
  fix).
- **P4b — neural multi-task diversity base** (low priority). A shared-trunk model
  with aux heads (spin/strength/position) + main heads, as one diversity base.
  Neural lost 3× (seq, FT, TabPFN); included only because compute is free and the
  gate cheaply kills it. Pilot-gate first (seed 11 × folds 0–2) before full 25-fold,
  exactly like the seq/FT pilots — abandon if the pilot is non-competitive.

**Gate.** Each vs 0.32568, OOF nested-CV > floor.

## 7. P5 — public leak expansion (deploy-only, runs last)

Depends on the final private bases from P1–P4.

- Verify the old-test overlap is exactly 1236/1845 rallies (or find more leakable
  rallies) to maximize coverage of the permitted serverGetPoint leak.
- Confirm action has no usable outcome-leak (+0.0008 measured → treat as dead) and
  drop that avenue if reconfirmed.
- **Rebuild leak-max on top of the improved private base**: re-derive point with
  the best post-P4 model + the leak serverGetPoint feature, override point on the
  overlap rallies, keep server smoothing. Target: beat the current 0.420782.
- The honest private submission (`submission_FINAL_safe_perrow.csv`) stays
  leak-free; leak only touches the public variant.

## 8. Deliverables and tracking

- Each phase: a producer/scorer script + nested-CV gate result recorded in
  `PROGRESS.md` (ship/reject + honest lift vs 0.32568, same table format as v2).
- Production changes only via `build_final_perrow.py` BASES (P2/P3/P4) or the
  scorer (P1). Rejected artifacts kept for reproducibility, NOT in BASES.
- Public: `submission_FINAL_leakmax.csv` regenerated only after private is locked.
- The honest per-row overall is the single ruler for every ship decision.

## 9. Out of scope

- Seed-averaged scoring (banned — inflates).
- Re-trying already-rejected levers unchanged (seq2/TabPFN/XGBoost/FT/hill-climb/
  rebalancing/pruning/depth-8).
- LB-probing of the public leaderboard (user did not select it; COI caution).
- Any change validated only on public LB or only on argmax.
