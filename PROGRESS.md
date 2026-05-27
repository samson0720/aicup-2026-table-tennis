# AI Cup 2026 — Score-Improvement Plan Execution Progress

Status as of 2026-05-27. Updated by the active AI agent after every plan task.
Source-of-truth for the next agent: read this BEFORE touching code.

## Where we are

- Spec: `docs/superpowers/specs/2026-05-27-aicup-score-improvements-design.md` (Draft, awaiting user review — but execution has begun per user instruction).
- Plans: `docs/superpowers/plans/2026-05-27-{cv-foundation,route-a-postprocess-stack,route-b-features-chain,route-c-transformer,final-ensemble}.md`.
- Conda env required: **`aicup-tt`** (built by P1 Task 1). Never use base/system Python.
- GPU available: NVIDIA 3090. Only Route C (P4) actually needs it; everything else is CPU.

## Plan execution status

| Plan | Tasks | Status |
|---|---|---|
| P1 cv-foundation | 1–10 | **DONE** (commits `fd2cf58`..`d8b46fc`) |
| P2 route-a-postprocess-stack | 1–12 | **next up** |
| P3 route-b-features-chain | 1–10 | pending — independent of P2 |
| P4 route-c-transformer | 1–6 | pending — needs 3090; independent of P2/P3 |
| P5 final-ensemble | 1–7 | pending — blocked by P2 + P3 + P4 |

## P1 results (baseline locked in)

`scripts/diagnose_cv_gap.py` re-ran LGBM sqrt/leaves=15 (180 estimators) on
the new private-safe CV. Every future experiment is measured against these
numbers; any lift smaller than the across-fold std is treated as noise.

| Target | New CV | Old CV (`lgbm_baseline_cv.json`) | Δ |
|---|---:|---:|---:|
| action macro-F1 | 0.2550 | 0.2948 | -0.040 |
| point  macro-F1 | 0.1711 | 0.2154 | -0.044 |
| server AUC      | 0.6541 | 0.6016 | +0.052 |
| **overall**     | **0.3013** | 0.3244 | -0.023 |
| std overall     | 0.0063 | — | — |

Key takeaways:
- **Noise floor: 0.0063 overall.** Reject any A/B/C/Ensemble improvement smaller than this on the new CV.
- pointId is still the weakest target — confirms the design spec's claim. Route B target encoding is the most plausible lift here.
- Server AUC went UP under the new CV: mid-rally cuts expose phase 0 / phase 1 where there is more signal than at end-of-rally; suggests the phase-aware blend in Route A Task 8 could compound the gain.
- Clean public LB (no smoothing) was 0.3264 vs old-CV-overall 0.3244, almost matching. New-CV-overall 0.3013 is therefore the realistic "private-safe" baseline.

## Public LB probes (2026-05-27)

Three clean variants uploaded; see HANDOFF.md "Clean-base public LB probes"
section for the table. Highlights:
- **leaves=31 clean = 0.3364, beats leaves=15 clean 0.3263 by +0.010.**
- **phase_lgbm clean = 0.3349, beats leaves=15 clean by +0.009** (local-vs-public disagreement).
- player_stats clean = 0.3115, confirmed weak; drop or near-zero meta weight.

**Decision framework** (revised — public is not a model-quality oracle):
- Local CV (25 folds, std 0.0063) is the primary signal — it's built to
  mimic private. Public LB is one noisy number per submission with no
  variance estimate.
- Public only breaks ties when the local delta is smaller than one local
  std. Never use public to override a clear local verdict.
- Final submission picks the version with the highest **local** CV overall.

**Re-tune for P2 Route A** under that framework:
- `lgbm31`: local-tied with lgbm15 (delta -0.0009 << std 0.0063); public
  breaks tie cleanly (+0.010). Promote to primary base.
- `lgbm15`: keep as diversity base.
- `phase_lgbm`: real local-vs-public conflict (local rejects by 1σ,
  public lifts by ~1σ). Keep in base set but DON'T pre-weight — let the
  meta-learner decide based on OOF correlation/diversity.
- `markov`: keep as cheap extra signal (99.7% agreement with lgbm15,
  near-duplicate but might give the stacker a stable anchor).
- `player_stats`: BOTH local and public agree it's bad. Drop from base set
  or assign zero meta weight. Remove from `MODELS` list in
  `scripts/build_route_a_submission.py` / `scripts/produce_base_oof.py`.

## Known issues to fix when we hit them

Flagged during plan review on 2026-05-27, NOT YET FIXED:

1. **P3 Route B Task 10 in-bag leakage (REAL BUG)** — `scripts/build_route_b_submission.py` as drafted trains stage-A action on full train, then uses **in-bag** predictions as `act_p_<i>` features for training stage-B point. LGBM is overconfident in-bag → stage-B learns to over-trust `act_p_*` columns → at test time real action probs are less confident → severe distribution shift.
   - **Fix when implementing P3 T10**: feed OOF action probabilities (`artifacts/oof/chain_action_action.parquet`) into the stage-B training feature set instead of in-bag preds. Same for stage-C consuming both.

2. **P2 Route A Task 11 incomplete** — only LGBM `predict_test_lgbm` is sketched. `markov`, `phase_lgbm`, `player_stats` test-time variants are described in prose only.
   - **Fix when implementing P2 T11**: each existing `train_*.py` already has internal logic that produces a `submission_*.csv` artifact today; extract that test-prediction block into a `predict_test_*` helper using the same OOF schema (`artifacts/oof/<model>_<target>_test.parquet`).

3. **P2 Task 10 stacker label collapse** — `attached.drop_duplicates("rally_uid")` keeps only one seed's cut-target label per rally; the seed-averaged base features see all 5 seeds. This is a feature/label mismatch but probably benign because LR with L2 is robust to label noise.
   - **Decide when running P2 T9–10**: if stacking lift is < 0.01, revisit by training the meta-learner on the un-averaged (rally, seed) rows so labels match features per row.

4. **P1 Task 5.3 phase-stratified rng state consumption** — the histogram-building `rng.integers` call inside `assign(_cut=...)` consumes rng state before the actual cut sampling. May cause cuts to differ slightly from a Task-4-only path. Should not affect invariants — tests will catch any drift.

## Logged decisions

- Use `git add -f artifacts/cv_splits.parquet` when needed: the existing `.gitignore` excludes `artifacts/*.parquet`, but the plan asks us to commit this specific small file (~1–2 MB).
- Run the diagnostic in P1 Task 9 in the background (10–20 min CPU work, 25 LGBM fits) and only score it once it finishes.
- Skip P5 Task 6 Optuna HPO until P2–P4 land (3–6 h CPU; not worth burning until base models are stable).

## Resuming work

Next concrete steps the next agent should take, in order:
1. **P1 is fully done.** Skip the P1 plan.
2. Open `docs/superpowers/plans/2026-05-27-route-a-postprocess-stack.md` and start at Task 1 (OOF parquet schema lock — `scripts/oof_loader.py`).
3. Before Task 11 (test-time refit), re-read item 2 in the "Known issues" section above; the existing `train_*.py` scripts already have test-prediction logic that should be lifted into `predict_test_*` helpers.

Wall-clock budget for P2:
- Tasks 1–5 (oof_loader + producer + scoring): code only, ~30 min interactive
- Tasks 2.2–2.3 (lgbm15 / lgbm31 OOF generation): 25 folds × ~30 s/fold × 2 leaves = ~25 min each, run in background
- Tasks 3–4 (markov / phase_lgbm / player_stats producers): need legacy script refactor — budget 1 hour each for the refactor, then ~15–30 min per OOF run
- Tasks 6–9 (post-process + stacker + tests): ~1 hour interactive
- Tasks 10–12 (submission assembly + lift comparison): ~1 hour

Realistically P2 takes 4–6 hours of compute + 2–3 hours of agent work. Pace yourself across sessions.

## Notes on conda usage

- All plan commands run inside the env: `conda run -n aicup-tt <command>` (no shell activation needed).
- Do NOT install with `pip install --user`. Do NOT touch `base`.
- For ad-hoc Python sanity checks BEFORE `aicup-tt` exists, the `jitkg` env on this machine has `pandas`. Don't conflate it with project work.
