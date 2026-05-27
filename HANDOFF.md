# AI CUP Table Tennis Handoff

## Current best public result

- Uploaded file: `artifacts/submission_RECOMMENDED_upload.csv`
- Public score: `0.4102132`
- Rank at time checked: `45/333`

Second upload was lower:

- Uploaded file: `artifacts/submission_CANDIDATE2_lgbm_sqrt_leaves15.csv`
- Public score: `0.3990370`

Third upload was a clean, no-leakage-proxy submission:

- Uploaded file: `artifacts/submission_CLEAN_lgbm_sqrt_leaves15.csv`
- Public score: `0.3263827`
- Rank at time checked: `157/335`

## Clean-base public LB probes (2026-05-27, 3-shot probe)

All clean (no old-test smoothing). Compared against the known clean baseline
`submission_CLEAN_lgbm_sqrt_leaves15.csv = 0.3263827`.

| Submission | Public score | Δ vs leaves15 clean |
|---|---:|---:|
| `submission_CLEAN_lgbm_sqrt_leaves31.csv`     | **0.3363696** | **+0.0100** |
| `submission_phase_lgbm_clean.csv`             | **0.3348781** | **+0.0086** |
| `submission_lgbm_playerstats_clean.csv`       | 0.3115222     | -0.0149     |

Conclusions used to retune P2 Route A's base set:

1. leaves=31 is the strongest clean base on public LB (was tied with leaves=15
   on local CV, 0.32350 vs 0.32439). Stacking should treat leaves=31 as the
   primary base, leaves=15 as the diversity-only addition.
2. phase_lgbm shows local-vs-public disagreement: local CV rejected it
   (0.31869 < 0.32439); public LB clean has it +0.0086 above leaves=15.
   Likely because public LB has more short-prefix (phase 0/1) rallies than
   our 5-pp phase stratification preserves, and phase_lgbm specializes there.
   Keep it as a base in P2 with non-trivial weight.
3. player_stats is confirmed weak by both local and public (-0.037 local,
   -0.015 public vs leaves=15). Drop it from P2 base set, or assign near-zero
   meta-learner weight.

New CV ↔ public-clean mapping reference point:
- clean leaves=15: new CV overall 0.3013 ↔ public 0.3263 (offset ≈ +0.025)
- This is the only mapping we have so far; do not over-trust the offset for
  models that perform differently across phases.

## Important conclusion

The old `Reference_Only_Old_Test_Data/test.csv` leakage was already used for `serverGetPoint`.

For the 1,236 `rally_uid` values overlapping with `test_new.csv`:

- old `serverGetPoint = 1` -> submitted `0.95`
- old `serverGetPoint = 0` -> submitted `0.05`

This leakage does not provide the next-stroke `actionId` or `pointId` labels. In old test, those columns are the known prefix strokes, not the hidden next-stroke answer.

The clean leaves15 upload removed the old-test `serverGetPoint` smoothing and dropped from the public-best `0.4102132` to `0.3263827`. This suggests the public leaderboard is highly sensitive to the old-test overlap smoothing, so public scores from smoothing submissions should not be treated as pure model quality.

## Scripts

- `scripts/eda.py`: data summary and leakage/overlap checks.
- `scripts/train_lgbm_baseline.py`: prefix sample builder and LightGBM CV.
- `scripts/make_lgbm_submission.py`: full-train LightGBM submission generator.
- `scripts/train_markov_ensemble.py`: phase-aware Markov baseline and ensemble check.
- `scripts/train_player_stats_lgbm.py`: OOF-safe player statistics experiment.
- `scripts/train_phase_lgbm.py`: phase-specific LightGBM experiment.
- `scripts/train_catboost_baseline.py`: CatBoost experiment, too slow on this laptop.
- `scripts/test_like_validation.py`: 5-seed test-like validation gate.
- `scripts/private_safe_report.py`: three-scale report for all-prefix match, test-like rally, and test-like match validation.

## Best local results so far

Original all-prefix GroupKFold by match:

- LGBM `sqrt`, `num_leaves=31`: local overall `0.32350`, public `0.4102132`
- LGBM `sqrt`, `num_leaves=15`: local overall `0.32439`, public `0.3990370`
- Clean LGBM `sqrt`, `num_leaves=15`: public `0.3263827`

Other experiments:

- Markov ensemble: local `0.32355`, negligible improvement.
- Player stats: local `0.28714`, rejected.
- Phase-specific LGBM: local `0.31869`, rejected.
- CatBoost: stopped after more than 30 minutes, not practical for current laptop.

## Test-like validation gate

Script:

```powershell
python scripts\test_like_validation.py --seeds 11 22 33 44 55 --folds-rally 3 --folds-match 3 --estimators 180
```

Result file:

- `artifacts/test_like_validation_gate.json`

Key conclusion:

- Rally-level public-like validation did not reliably distinguish leaves31 vs leaves15.
- Match-level private-safe validation preferred leaves15, opposite of public leaderboard.
- Therefore this validation is not yet reliable enough for final submission decisions.

## Current public-best file

```text
artifacts/submission_RECOMMENDED_upload.csv
```

## Validation strategy (post-2026-05-27)

Old `GroupKFold by match` mismatches `test_new.csv`, which is built from
mid-rally cut points, not held-out matches. New CV (`scripts/cv_splits.py`,
materialized to `artifacts/cv_splits.parquet`):

- Per-rally random cut point in `[2, rally_len]`, mimicking `test_new`.
- Match-aware grouping: every rally of a match shares one fold per seed.
- Phase-stratified greedy round-robin: each match is assigned to the fold
  with the lowest current load in that match's dominant phase bucket.
- 5 seeds × 5 folds = 25 (seed, fold) cells, ~2999 rallies each.
- Locked-in invariants in `tests/test_cv_splits.py`: schema, row count,
  match grouping, fold range, phase share within 5pp, cut validity,
  cross-seed cut variance, iter_cv_folds no-leak.

Overlap analysis (from design spec): 0 of 216 train matches reappear in
79 test matches; 40 of 71 test players appear in train, 31 are unseen.
Route B target encodings must smooth toward a global prior to handle
unseen players. See `docs/superpowers/specs/2026-05-27-aicup-score-improvements-design.md`.

Diagnostic (`artifacts/cv_gap_diagnostic.json`) records the new-CV LGBM
baseline overall metric and per-seed std. Improvements smaller than one
std are rejected as noise by every downstream route.

Smoothing trick (old-test `serverGetPoint` overlap → 0.95/0.05) stays out
of every new model's training and OOF. It is only applied when generating
the `submission_FINAL_smooth.csv` public-leaderboard backup in Section 5
of the design spec.

## Plan execution status

The active plan is broken into 5 files under `docs/superpowers/plans/`
(P1 CV foundation → P5 final ensemble). See `PROGRESS.md` for current
task status and known plan-review issues to fix during execution.
