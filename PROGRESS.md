# AI Cup 2026 — Score-Improvement Plan Execution Progress

Status as of 2026-05-27. Updated by the active AI agent after every plan task.
Source-of-truth for the next agent: read this BEFORE touching code.

## 2026-05-31: pointId feature engineering — TRIED, REJECTED (below noise)

Branch `feat/pointid-feature-engineering` (NOT merged into the 0.4247 line).

Added prefix-safe pointId joint/dispersion features to `add_prefix_features`
(`bigram_last_point`, `last_point_position`, `bigram_last_position`,
`std_pointId`, `point_switch_rate`). Diagnostic `scripts/diag_point_features.py`
first confirmed real headroom: the (last2,last1) point bigram and
(last_point,last_position) joints carry ~4.8–4.9% mutual information with the
next pointId, more than any single column.

But after rebuilding all base OOF locally and scoring honestly:

| base | point F1 Δ | overall Δ |
|---|---:|---:|
| lgbm15 | +0.0007 | -0.0003 |
| lgbm31 | +0.0009 | +0.0001 |
| phase_lgbm | -0.0006 | -0.0000 |

All far below the point noise floor (0.00506) and overall noise floor (0.00168);
action even dipped slightly. Cause: the existing `point_cnt_*`/`point_rate_*`/
`last1-5_pointId`/`last_action_point`/`phase_last_point` columns already let the
trees reconstruct these joints, so the explicit codes are redundant. Reverted on
`feat/per-target-beta`; preserved here for the record only.

## 2026-05-30: per-target prior-temperature (beta) + new public best 0.4247437

- Added `beta` to `postprocess.prior_correct` (default 1.0, backward compatible)
  and `select_beta()` honest-CV selector. Wired into `train_chain_lgbm.score_chain`
  (writes `route_b_beta_*.json`) and the canonical `build_final_perrow.py`
  (`_select_beta_nested`). actionId picks beta≈0.6, pointId beta≈0.4 (was
  hardcoded 1). Clean same-base ensemble lift = **+0.0028 overall** (~1.7σ;
  action +0.0026, point +0.0044). No downside (server untouched).
- Regenerated all 4 base OOF + test parquets locally on the Mac (~36 min CPU)
  and rebuilt the canonical ensemble. Honest local-CV overall = **0.3231**.
- `submission_FINAL_smooth_perrow.csv` scored **0.4247437** on public LB
  (prev best 0.4102132, +0.0145; both smoothed → real model gain).
- Fixed `build_final_perrow.py` for sklearn≥1.7 (removed dropped `multi_class`
  arg; behavior-preserving). STILL present in stacker.py,
  build_route_a_submission.py, build_final_submissions.py,
  diag_perrow_vs_seedavg.py — fix if those are run on new sklearn.
- Local Mac env recipe (no conda/Homebrew): see README.md "重新產生 submission B".
- Diagnostics added: scripts/diag_per_class_f1.py, scripts/diag_prior_headroom.py.
- Next best ROI (transformer is NOT it — data too small/short): add an XGBoost
  base for stacker diversity, then pointId feature engineering.

## CRITICAL CORRECTION (2026-05-28): seed-averaging inflated the stack scores

The reported final ensemble overall **0.3497 was inflated and unrealizable.**
Root cause: `build_final_submissions.py` / `build_route_a_submission.py` and the
stacker averaged each rally's base predictions over the 5 seeds
(`average_over_seeds`) before stacking AND before scoring. Because every seed
cuts the same rally at a **different** strikeNumber, that average is an ensemble
over cut points. At submission time each test rally has exactly **one** prefix
(one cut, 1845 rows = 1845 rallies), so the averaging cannot be reproduced. It
inflated the local score, dominated by server AUC (per-row **0.6557** vs
seed-averaged **0.7702**).

Decomposition discovered: the "+0.047 ensemble lift" in the old P5 section
compared base scores on the **all-cuts** population (0.3027) against the
ensemble on the **seed-averaged** population (0.3497) — two different
populations, not a valid comparison.

**Honest per-row ensemble (matches test reality, matches public ~0.32):**

| metric | seed-avg (INFLATED) | per-row (HONEST) | best base lgbm15 |
|---|---:|---:|---:|
| action F1 | 0.2540 | 0.2894 | 0.2587 |
| point F1  | 0.2352 | 0.1842 | 0.1730 |
| server AUC| 0.7702 | 0.6557 | 0.6499 |
| overall   | **0.3497** | **0.3206** | 0.3027 |

The ensemble is still genuinely worth it: honest lift over best base =
**+0.0179** (~10x the 0.00168 noise floor). It was just measured wrong.

Second (real) bug fixed: the old submission's meta-learner was trained on
seed-averaged features but applied to single-cut test features — a
train/test distribution mismatch that likely caused the old clean submission to
score *below* leaves31 on public. The fix trains the meta on per-row features.

**Canonical builder is now `scripts/build_final_perrow.py`** (per-row stack +
honest nested-threshold scoring + distribution-matched test meta). Outputs:
`artifacts/submission_FINAL_safe_perrow.csv` (private bet) and
`artifacts/submission_FINAL_smooth_perrow.csv` (public backup, 1236 overlap rows
smoothed). The old `build_final_submissions.py` and its
`submission_FINAL_safe.csv` are **DEPRECATED** (inflated/mismatched).

vs old safe submission: point predictions changed on 1039/1845 rallies (56%),
action on 370, server recalibrated (corr 0.973, mean 0.474->0.516). Honest
score = 0.3206 ≈ public clean ~0.32, so the ruler is now trustworthy.

Evidence scripts (re-runnable): `scripts/diag_threshold_honesty.py` (rules out
in-sample threshold tuning as the cause, only +0.005), `scripts/diag_perrow_vs_seedavg.py`
(isolates the seed-averaging inflation, server 0.6525 per-row vs 0.7635 seed-avg).

NEXT: use the per-row score (0.3206) as the honest ruler for ANY future "make it
stronger" work. Never compare a seed-averaged number against an all-cuts number.

## Where we are

- Spec: `docs/superpowers/specs/2026-05-27-aicup-score-improvements-design.md` (Draft, awaiting user review — but execution has begun per user instruction).
- Plans: `docs/superpowers/plans/2026-05-27-{cv-foundation,route-a-postprocess-stack,route-b-features-chain,route-c-transformer,final-ensemble}.md`.
- Conda env required: **`aicup-tt`** (built by P1 Task 1). Never use base/system Python.
- GPU available: NVIDIA 3090. Only Route C (P4) actually needs it; everything else is CPU.

## Plan execution status

| Plan | Tasks | Status |
|---|---|---|
| P1 cv-foundation | 1–10 | **DONE** (commits `fd2cf58`..`d8b46fc`, main) |
| P2 route-a-postprocess-stack | 1–12 | **DONE** on `p2-route-a` branch |
| P3 route-b-features-chain | 1–10 | **DONE** on `p2-route-a` branch |
| P4 route-c-transformer | 1–6 | DONE as rejected weak base — GPU path works, 2-epoch OOF below noise gate |
| P5 final-ensemble | 1–7 | DONE — final A+B stack built, Route C excluded |

## P2 Route A results (2026-05-28)

All 24 base parquet artifacts are present under `artifacts/oof/`:
4 models × 3 targets × `{OOF, test}`. Final builder writes
`artifacts/submission_A_stacked.csv` with 1,845 test rallies.

Apples-to-apples NEW CV base scores via `score_oof`:

| Model     | action F1 | point F1 | server AUC | overall |
|-----------|----------:|---------:|-----------:|--------:|
| lgbm15    | 0.2587    | 0.1730   | 0.6499     | **0.3027** |
| lgbm31    | 0.2518    | 0.1704   | 0.6469     | **0.2983** |
| markov    | 0.1592    | 0.0911   | 0.5841     | **0.2170** |
| phase_lgbm| 0.2426    | 0.1584   | 0.6444     | **0.2893** |

Route A stacked OOF (`artifacts/route_a_scores.json`):

| Variant | action F1 | point F1 | server AUC | overall | lift vs lgbm15 |
|---|---:|---:|---:|---:|---:|
| raw stack | 0.2412 | 0.2324 | 0.7643 | **0.3423** | **+0.0396** |
| final selected | 0.2412 | 0.2324 | 0.7643 | **0.3423** | **+0.0396** |

Lift is far above the overall noise floor (`0.00168`) and the 3σ health
threshold (`0.005`), so audit F4's un-averaged `(rally, seed)` stacker rewrite
is NOT needed for P2.

The initial phase-prior server blend `{0: 0.7, 1: 0.4, 2: 0.0}` was rejected
by local OOF: server AUC fell from `0.7643` to `0.7468`, costing `0.0035`
overall (> noise floor). Route A therefore uses pure stacked server
probabilities (`SERVER_BLEND_WEIGHTS = {0: 0.0, 1: 0.0, 2: 0.0}`).

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
- T10+T11 ✓ submission assembly + test-time refit written and executed
- T12 ✓ lift quantified: `0.3423 - 0.3027 = +0.0396`; P2 accepted

## P3 Route B results (2026-05-28)

Implemented OOF-safe feature engineering and chain LightGBM:

- `scripts/target_encoding.py`: smoothed encoders with unseen-key global prior fallback.
- `scripts/feature_ngrams.py`: prefix n-gram/streak features.
- `scripts/feature_semisupervised.py`: train+test observable player feature distributions.
- `scripts/build_features_v2.py`: writes `artifacts/prefix_train_v2.parquet` `(74975, 322)`.
- `scripts/train_chain_lgbm.py`: action → point → server chain using prior-stage OOF probabilities.
- `scripts/build_route_b_submission.py`: full-train test inference without in-bag leakage in downstream training features.

Route B OOF scores (`artifacts/route_b_chain_scores.json`):

| Variant | action F1 | point F1 | server AUC | overall | lift vs lgbm15 |
|---|---:|---:|---:|---:|---:|
| raw argmax | 0.2689 | 0.1627 | 0.6498 | 0.3026 | -0.0000 |
| prior+threshold selected | 0.2880 | 0.1817 | 0.6498 | **0.3178** | **+0.0152** |

The selected Route B score beats the best base (`lgbm15 overall=0.3027`) by
`+0.0152`, well above the `0.00168` noise floor and within the planned
`+0.015` to `+0.030` target range.

Important implementation note: the known P3 Task 10 in-bag leakage issue was
fixed. Point/server training rows consume `chain_action`/`chain_point` OOF
probabilities merged by `(rally_uid, seed, fold)`. Test rows consume full-train
upstream model probabilities, which matches the intended inference path without
letting downstream stages train on in-bag upstream probabilities.

Generated submission:

- `artifacts/submission_B_chain.csv` `(1845, 4)`.
- Test probability parquets:
  `artifacts/oof/chain_action_action_test.parquet`,
  `artifacts/oof/chain_point_point_test.parquet`,
  `artifacts/oof/chain_server_server_test.parquet`.

## P4 Route C progress (2026-05-28)

GPU gate details:

- Inside the default sandbox, PyTorch cannot see CUDA and `nvidia-smi` cannot
  communicate with the driver.
- Outside the sandbox via approved escalated commands, `nvidia-smi` sees:
  GTX 1080 and RTX 3090.
- CUDA device ordering differs from `nvidia-smi`: use
  `env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt ...` to expose the RTX
  3090 as PyTorch device 0. `CUDA_VISIBLE_DEVICES=1` maps to the GTX 1080.

Implemented P4 foundation:

- `scripts/seq_dataset.py`: `RallyPrefixDataset`, masks, padding, and collate.
- `scripts/seq_model.py`: small multi-task Transformer with action/point/server heads.
- `scripts/train_seq_transformer.py`: fold-level training/smoke script with AMP support.
- `tests/test_seq_dataset.py` and `tests/test_seq_model.py`: 5 tests for shape,
  label-stroke exclusion, collate, forward pass, and parameter sanity.

Validation:

- Full `pytest -q` passes with 32 tests.
- 3090 smoke command passed:
  `env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt python -m scripts.train_seq_transformer --seeds 11 --fold 0 --epochs 1 --batch-size 64 --d-model 64 --layers 1 --ffn 128 --max-train 256 --max-valid 128`
  printed `device=cuda`, `NVIDIA GeForce RTX 3090`, one train loss, and
  `valid_pred_rows=128`.

Full 25-fold OOF run (5 seeds × 5 folds, 2 epochs) completed on RTX 3090:

- Command:
  `env CUDA_VISIBLE_DEVICES=0 conda run -n aicup-tt python -u -m scripts.train_seq_transformer --seeds 11 22 33 44 55 --fold -1 --epochs 2 --batch-size 256 --model-name seq --write-oof --write-partial`
- Outputs:
  `artifacts/oof/seq_action.parquet`,
  `artifacts/oof/seq_point.parquet`,
  `artifacts/oof/seq_server.parquet`, each with 74,975 rows.
- OOF score (`artifacts/route_c_seq_scores.json`), selected raw because
  prior+threshold hurt this model:
  action F1 `0.1550`, point F1 `0.1386`, server AUC `0.6129`,
  overall **`0.2400`**.

Decision: reject current Route C from final stack. It is far below lgbm15
(`0.3027`) and Route B (`0.3178`), so it is not a useful base. A single
10-epoch pilot on seed 11 / fold 0 improved that fold from `0.2328` to
`0.2915`, but still did not beat lgbm15 locally; full long training is deferred.

## P5 Final ensemble results (2026-05-28)

Implemented `scripts/build_final_submissions.py`.

Final stack inputs:

- Route A raw bases: `lgbm15`, `lgbm31`, `markov`, `phase_lgbm`.
- Route B chain bases: `chain_action`, `chain_point`, `chain_server`.
- Route C `seq` excluded because its OOF overall is `0.2400`.

Final OOF score (`artifacts/final_scores.json`):

| action F1 | point F1 | server AUC | overall |
|---:|---:|---:|---:|
| 0.2540 | 0.2352 | 0.7702 | **0.3497** |

Lift comparisons:

- vs best base `lgbm15`: `+0.0471`.
- vs Route A standalone `0.3423`: `+0.0075`.
- vs Route B standalone `0.3178`: `+0.0319`.

Generated submissions:

- `artifacts/submission_FINAL_safe.csv`: private-safe model ensemble, no old-test smoothing.
- `artifacts/submission_FINAL_smooth.csv`: same predictions with old-test server smoothing
  on 1,236 overlap rows; use only as the public-leaderboard backup.

Submission guardrail passed for both files: 1,845 unique test rallies, valid
columns, valid action/point class ranges, server probabilities in `[0, 1]`.

## P1 results (baseline locked in)

`scripts/diagnose_cv_gap.py` re-ran LGBM sqrt/leaves=15 (180 estimators) on
the new private-safe CV. Every future experiment is measured against these
numbers; any lift smaller than the across-seed std is treated as noise.

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
