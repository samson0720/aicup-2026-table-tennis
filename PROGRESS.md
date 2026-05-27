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
| P1 cv-foundation | 1–10 | **DONE** (commits `fd2cf58`..`d8b46fc`, main) |
| P2 route-a-postprocess-stack | 1–12 | **in progress** on `p2-route-a` branch |
| P3 route-b-features-chain | 1–10 | pending — independent of P2 |
| P4 route-c-transformer | 1–6 | pending — needs 3090; independent of P2/P3 |
| P5 final-ensemble | 1–7 | pending — blocked by P2 + P3 + P4 |

## P2 partial results so far (2026-05-28)

LGBM OOFs done. Apples-to-apples NEW CV scores via score_oof:

| Model     | action F1 | point F1 | server AUC | overall |
|-----------|----------:|---------:|-----------:|--------:|
| lgbm15    | 0.2587    | 0.1730   | 0.6499     | **0.3027** |
| lgbm31    | 0.2518    | 0.1704   | 0.6469     | **0.2983** |

**Strongest local-vs-public conflict yet**: lgbm15 beats lgbm31 by **+0.0044**
on new CV (>1σ above noise floor 0.00168). But on clean public LB, leaves=31
WON by +0.010. The two signals genuinely disagree on the same submission set.

Implication: when stacker weights are learned on local OOF (per spec 5.3),
the meta-learner will probably weight lgbm15 higher than lgbm31. That's
correct for private LB — we're explicitly NOT optimizing public — but means
the final submission may have a LOWER public score than RECOMMENDED 0.4102
even if it's BETTER on private. Don't panic if public clean reads below 0.33.

## P2 progress (branch `p2-route-a`)

Commits in order: `8833af7` (T1 oof_loader) → `82eba0c` (T5-T9 helpers + T2
producer code) → `80dd039` (T3+T4 markov/phase_lgbm refactor) → `a6fe428`
(T10+T11 stacker assembly + test-time refit).

Status:
- T1 ✓ scripts/oof_loader.py + tests (4 tests green)
- T2 ✓ producer code; lgbm15 OOF generated (score 0.3027, matches diagnostic baseline within noise)
- T3 ✓ markov_oof helper extracted; smoke-tested on 1 fold
- T4 ✓ phase_lgbm_oof helper extracted (player_stats dropped per F3)
- T5-T9 ✓ score_oof + postprocess (prior/threshold/blend/pair-prior) + stacker (10 tests green total)
- T10+T11 ✓ code written
- T12: pending the lift comparison once full pipeline finishes

Background pipeline queued (`baiax9m3h`):
1. (running) lgbm31 OOF → markov OOF → phase_lgbm OOF
2. (queued) predict_test for lgbm15 / lgbm31 / markov / phase_lgbm
3. (queued) build_route_a_submission → writes artifacts/submission_A_stacked.csv

Total wait ~1.5-2 hours. Resume here when the notification arrives — score
the stacked OOF against the new-CV noise floor (0.00168) and decide on
audit F4 (label-collapse fix needed if lift < 0.005).

## P1 results (baseline locked in)

`scripts/diagnose_cv_gap.py` re-ran LGBM sqrt/leaves=15 (180 estimators) on
the new private-safe CV. Every future experiment is measured against these
numbers; any lift smaller than the across-fold std is treated as noise.

| Target | New CV | Old CV (`lgbm_baseline_cv.json`) | Δ | **Noise floor** (across-seed std) |
|---|---:|---:|---:|---:|
| action macro-F1 | 0.2550 | 0.2948 | -0.040 | 0.00525 |
| point  macro-F1 | 0.1711 | 0.2154 | -0.044 | 0.00506 |
| server AUC      | 0.6541 | 0.6016 | +0.052 | 0.00397 |
| **overall**     | **0.3013** | 0.3244 | -0.023 | **0.00168** |

Key takeaways:
- **Noise floor (spec Section 1.2): 0.00168 overall, across-seed std.**
  Reject any A/B/C/Ensemble improvement smaller than this on the new CV.
- Earlier commits (`f48385b`, `d8b46fc`, `4629e72`) and HANDOFF erroneously
  quoted 0.00635 as the noise floor — that was the across-FOLD std (25 cells
  together). The spec mandates the smaller, stricter across-seed std. The
  diagnostic JSON now records both; downstream tools use `std_across_seed`.
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
- Local CV (5 seeds × 5 folds, **across-seed std 0.00168**) is the primary
  signal — it's built to mimic private. Public LB is one noisy number per
  submission with no variance estimate.
- Public only breaks ties when the local delta is smaller than one local
  std. Never use public to override a clear local verdict.
- Final submission picks the version with the highest **local** CV overall.

**Re-tune for P2 Route A** under that framework:
- `lgbm31`: local-tied with lgbm15 on the OLD CV (delta -0.0009; old-CV
  numbers come from `artifacts/lgbm_baseline_cv.json`, not the new CV).
  Both are within the new CV's across-seed noise floor of 0.00168; we
  have not yet rerun leaves=31 under the new CV. Public LB clean breaks
  the tie cleanly (+0.010). Promote to primary base; re-verify under new
  CV in P2 Task 2.
- `lgbm15`: keep as diversity base.
- `phase_lgbm`: real local-vs-public conflict (local rejects by 1σ,
  public lifts by ~1σ). Keep in base set but DON'T pre-weight — let the
  meta-learner decide based on OOF correlation/diversity.
- `markov`: keep as cheap extra signal (99.7% agreement with lgbm15,
  near-duplicate but might give the stacker a stable anchor).
- `player_stats`: BOTH local and public agree it's bad. Drop from base set
  or assign zero meta weight. Remove from `MODELS` list in
  `scripts/build_route_a_submission.py` / `scripts/produce_base_oof.py`.

## Audit findings (2026-05-28, pre-P2)

Cross-checked spec vs 5 plans vs PROGRESS vs HANDOFF vs memory. Fixed (committed)
findings:

- **F1: Noise floor was wrong** — spec says across-seed std (0.00168), I had
  across-fold std (0.00635) in PROGRESS/HANDOFF/commit messages. Fixed in
  commit `9ea6268`; diagnostic JSON now emits both.
- **F2: LightGBM GPU not available** — spec asked for it, conda-forge has
  CPU-only. Spec Section 6 risk row permits fallback. Documented GPU policy
  (commit `ed202f8`): PyTorch + XGBoost GPU paths verified; LightGBM stays
  CPU because data is below GPU break-even.

Documented-but-deferred findings (acceptable spec deviations or design
ambiguities to resolve during P2 implementation, NOT bugs in P1):

- **F3: Drop player_stats from P2 base set deviates from spec Section 5.1**
  which lists it as one of five bases. Justified by both local CV (-0.037)
  and public clean (-0.015) agreeing it is worse than the leaves=15 baseline.
  When implementing P2 Task 4, decide: (a) skip building its OOF entirely,
  or (b) build it but exclude from MODELS list in the stacker. Option (a)
  saves ~37 min CPU and is the cleaner choice.
- **F4: P2 Task 10 stacker label-collapse design ambiguity** — plan averages
  base OOF over 5 seeds (one row per rally) but each seed has its own cut
  target, so `drop_duplicates("rally_uid")` keeps an arbitrary single label
  per rally. Cleaner design: train meta-learner on un-averaged (rally, seed)
  rows so labels match per-row. Defer decision until we see the lift from
  the plan-as-written; if lift < 0.005 (3 std-across-seed), switch to the
  un-averaged form.
- **F5: Spec Section 2.4 phase-blend weights** — spec gives starting values
  (0.3/0.7, 0.6/0.4, 1.0/0). Plan P2 Task 8 says they are tunable by OOF
  grid search. Use the spec values as the initial point of the grid.
- **F6: P1 Task 5.3 rng state consumption** — provisional-cut sampling and
  fold-assignment rng share state with later real-cut sampling. Mitigated
  in the actual cv_splits.py by re-seeding with `seed + 10_000` for the
  real cuts. Tests pass. No action needed.

## Known issues to fix when we hit them

Flagged during plan review on 2026-05-27, NOT YET FIXED:

1. **P3 Route B Task 10 in-bag leakage (REAL BUG)** — `scripts/build_route_b_submission.py` as drafted trains stage-A action on full train, then uses **in-bag** predictions as `act_p_<i>` features for training stage-B point. LGBM is overconfident in-bag → stage-B learns to over-trust `act_p_*` columns → at test time real action probs are less confident → severe distribution shift.
   - **Fix when implementing P3 T10**: feed OOF action probabilities (`artifacts/oof/chain_action_action.parquet`) into the stage-B training feature set instead of in-bag preds. Same for stage-C consuming both.

2. **P2 Route A Task 11 incomplete** — only LGBM `predict_test_lgbm` is sketched. `markov`, `phase_lgbm`, `player_stats` test-time variants are described in prose only.
   - **Fix when implementing P2 T11**: each existing `train_*.py` already has internal logic that produces a `submission_*.csv` artifact today; extract that test-prediction block into a `predict_test_*` helper using the same OOF schema (`artifacts/oof/<model>_<target>_test.parquet`).

3. **P2 Task 10 stacker label collapse** — `attached.drop_duplicates("rally_uid")` keeps only one seed's cut-target label per rally; the seed-averaged base features see all 5 seeds. This is a feature/label mismatch but probably benign because LR with L2 is robust to label noise.
   - **Decide when running P2 T9–10**: if stacking lift is < 0.01, revisit by training the meta-learner on the un-averaged (rally, seed) rows so labels match features per row.

4. ~~P1 Task 5.3 phase-stratified rng state consumption~~ — RESOLVED. The
   actual cv_splits.py re-seeds with `seed + 10_000` for the real cut sampler,
   isolating it from the provisional-cut rng state used for fold assignment.
   Tests pass; see audit F6.

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

## P2 entry criteria (must hold before starting Route A)

- [x] `aicup-tt` env exists and `torch.cuda.is_available()` True
- [x] `artifacts/cv_splits.parquet` materialized (74975, 6)
- [x] `tests/test_cv_splits.py` all 8 invariants green
- [x] `artifacts/cv_gap_diagnostic.json` written with both std definitions
- [x] Noise floor decision recorded: 0.00168 across-seed overall
- [x] GPU policy recorded: CPU LightGBM, GPU PyTorch + optional GPU XGBoost
- [x] PROGRESS lists known issues to address mid-P2 (F3/F4/F5; P3 in-bag bug for later)
- [ ] **Invoke `superpowers:executing-plans` skill** when opening
      `docs/superpowers/plans/2026-05-27-route-a-postprocess-stack.md`
- [ ] **Invoke `superpowers:verification-before-completion`** before any
      "task done" claim during P2

## Notes on conda usage

- All plan commands run inside the env: `conda run -n aicup-tt <command>` (no shell activation needed).
- Do NOT install with `pip install --user`. Do NOT touch `base`.
- For ad-hoc Python sanity checks BEFORE `aicup-tt` exists, the `jitkg` env on this machine has `pandas`. Don't conflate it with project work.

## GPU policy (verified 2026-05-28)

Spec Section 1.1 asked for "lightgbm built with GPU support". The conda-forge
`lightgbm=4.3.*` package and the PyPI wheel are both **CPU-only**; both
`device='cuda'` and `device='gpu'` raise `LightGBMError` ("Tree Learner was
not enabled in this build"). Spec Section 6 risk row explicitly allows the
fallback to CPU LightGBM.

Decision (per user instruction "能用到gpu的可以盡管用"):

- **PyTorch on 3090**: verified `cuda True`, `NVIDIA GeForce RTX 3090`. Used
  by Route C Transformer (P4) — the only spec-mandated GPU path.
- **XGBoost on 3090**: verified `xgb.XGBClassifier(device='cuda',
  tree_method='hist')` works. xgboost 2.0.3. Available as an optional
  ensemble-diversity base if we want one in P2 (spec Section 1.1 listed
  xgboost for "ensemble diversity, GPU-capable" but the current plans do
  not use it; revisit only if a base is needed beyond the planned five).
- **LightGBM stays CPU**: dataset is ~12k rows per fold which is below
  LightGBM GPU's typical break-even (~100k rows). GPU on this size is
  often slower than CPU due to data-transfer overhead. Building a GPU
  LightGBM from source (requires installing CUDA-dev headers or OpenCL
  ICD into the env) would risk breaking the working PyTorch 12.1 setup
  for negligible-to-negative wall-clock benefit.

Wall-clock budget on CPU LightGBM: per fit ~30 s (180 estimators, 12k rows,
~30 features) → per OOF model 75 fits × 30 s = ~37 min. With 5 base models
that's ~3 hours total. Acceptable.
