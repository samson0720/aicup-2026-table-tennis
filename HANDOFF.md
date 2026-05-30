# AI CUP Table Tennis Handoff

> **⚡ NEXT SESSION START HERE (2026-05-30) — v5 Track B complete.**
> Everything below the divider is older history; read PROGRESS.md first.

## ⛔ Levers already TRIED and REJECTED — do NOT re-litigate (read before proposing ideas)

Production = honest 8-base ensemble, overall **0.327081** (unchanged). The ensemble is at
the **information ceiling**: the next stroke is genuinely unobserved and rallies are short
(median 5 strokes, ~50% ≤4). Every lever below landed sub-floor (overall floor 0.00168;
action 0.00525; point 0.00506). New post-processing / re-encoding / variance-reduction does
NOT move it — only genuinely-new INFORMATION would. All scripts/OOF kept for reproducibility,
none in `build_final_perrow.py:BASES`.

| lever | what | verdict |
|---|---|---|
| seq2 Transformer | full-sequence neural | REJECT +0.00098 |
| TabPFN v2 | foundation tabular (point/server) | REJECT −0.00146 |
| XGBoost / FT-Transformer | more bases | REJECT (sub-floor / pilot ~0.24) |
| focal-loss GBDT, depth-8 cat, rare-class rebalance | objective/tuning | REJECT |
| markov2 (player×2-gram) | deeper n-gram | REJECT −0.00103 (overfit) |
| joint P(point\|action) | structured point | REJECT −0.00071 |
| hill-climb stacker, calibrated/wide decision rules | combiner/rule | REJECT |
| feature pruning | data-centric | REJECT (neutral) |
| **displacement / "pressure"** | pointId→XY run distance | REJECT at pilot — even with the **VERIFIED real geometry** ([[aicup-pointid-geometry]]); point +0.00121 (¼ floor), action negative |
| **markovz (player×landing-zone)** | P(action\|player,zone) | REJECT −0.00008 (redundant w/ markovp + GBDT raw player+zone) |
| pseudo-labeling | self-train on test | NOT done — amplifies majority bias vs macro-F1; covariate shift is minimal |
| serve-class rule zeroing | force P(15-18)=0 at strike≥2 | NO-OP (model never predicts them; fixed-19-label macro-F1) |
| multi-seed blend | avg model-init seeds at test | NOT done — can't gate on local CV (OOF seed = cut seed); expected sub-floor on a stacked ensemble |

**Only thing that moved real score = the serverGetPoint LEAK** (Track A, deployed in leakmax).

**OPEN public A/B (user action):** upload `submission_FINAL_leakmax_catonly.csv` to isolate
whether the ensemble-leakmax −0.00166 public drop (0.4207827→0.4191248) is Track A or the
shuttle 8-base. catonly = cat_sgp-alone point override on the same 8-base (differs from the
shipped ensemble leakmax on 392/1845 rallies' point; action/server identical).

## Next-session handoff: v5 Track B

Read first: `PROGRESS.md` (top "v5"/"v4" sections),
`docs/superpowers/specs/2026-05-30-aicup-v5-final-rank-design.md`, and the memory
(`MEMORY.md` + `aicup-*` files).

**Current state (do NOT redo):**
- repo `/home/tom1030507/ai_cup_table/aicup-2026-table-tennis`, branch `p2-route-a`, conda env `aicup-tt`.
- Final rank = PRIVATE, but the private scoring set INCLUDES the public rows, and the 1236
  leaked-serverGetPoint rallies ARE counted → **the file to upload is
  `submission_FINAL_leakmax.csv`** (NOT `safe_perrow`).
- Honest 8-base ensemble (incl. `shuttle`) = honest overall **0.327081** = `submission_FINAL_safe_perrow.csv`.
- **Track A is DONE** (leak-point ensemble cat_sgp+lgbm_sgp, nested point-F1 lift **+0.0116**,
  deployed into leakmax). Don't redo Track A.
- **Track B is DONE.** B1 position mirror rejected at pilot (action **−0.00569**).
  B2 fold-safe transition pretraining made ShuttleNet standalone much stronger
  (action **+0.01419**, point **+0.00655**) but production swap/add lifts were only
  **+0.000455 / +0.000231**, below the **0.00168** floor. Kept scripts/artifacts;
  reverted BASES to shipped `shuttle`; rebuilt safe/smooth/leakmax. Don't redo Track B.

**Track-B result detail:** `scripts/train_shuttle.py` now has opt-in
`--mirror-position-augment` and `--pretrain-transition-epochs`; default behavior is
unchanged. `scripts/score_shuttle_pilot.py` reproduces pilot/full standalone gates.
There are 69,712 usable next-stroke transitions from 84,707 strokes (rally-first strokes
have no preceding prefix). See `PROGRESS.md` top v5 sections and
`artifacts/shuttle_pretrain_integration_gate.json`.

**Process rules (lessons already paid for):**
- Everything `conda run -n aicup-tt`; GPU `env CUDA_VISIBLE_DEVICES=0` (the 3090 shows as
  nvidia-smi **index 1**, ~900 MiB = actually training; index 0 is the idle GTX 1080).
- `scripts/train_shuttle.py` needs `--num-workers 6` or it's data-loading-bound (GPU 0%).
- OOF/training runs take 10 min–hours: launch with `run_in_background: true`; do NOT hand
  them to subagents (they time out). **One command per background Bash call** — a newline in
  a single call gets space-joined and corrupts the command (this bit us twice).
- Honest per-row scoring only, never seed-average. **Every gate number must come from a real
  run you verified before writing it to PROGRESS/commit** (an interim commit this session
  recorded numbers that weren't from a real run — caught and corrected).
- Final upload = `submission_FINAL_leakmax.csv`; after changing any honest base, rebuild
  `build_final_perrow` (safe+smooth) then `build_leakmax_submission`.
- Update PROGRESS + memory and commit each step.

(If on a different machine: the gitignored `artifacts/oof/*.parquet` are NOT in git —
regenerate them first.)

---

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

Conclusions used to retune P2 Route A's base set.

Decision framework (revised after careful thought about variance):

- **Local CV is the primary go/no-go signal.** It is built to mimic private
  LB (per-rally cuts + match-grouped) and has 25 independent folds giving
  std=0.0063. Public LB is one number per submission with no variance
  estimate -- a single noisy reading.
- **Public LB acts as a tie-breaker only** when the local delta is smaller
  than one local std (i.e., local cannot tell which of two models wins).
- Never use public to override a clear local-CV verdict, because that
  optimizes for the public sample, not the private one.

Applied to the 3 probes above:

1. **leaves=31 vs leaves=15: local says tied** (delta -0.0009 << std 0.0063).
   Public breaks the tie cleanly in favor of leaves=31 (+0.010). Promote
   leaves=31 to the primary stacking base; keep leaves=15 for diversity.
2. **phase_lgbm vs leaves=15 baseline: real disagreement.** Local rejects
   it (-0.0057, about 1 std worse). Public clean says +0.0086 (about 1
   std better). With unknown public variance, we cannot tell whether the
   public lift is a sample fluke or a genuine phase-distribution effect.
   Keep phase_lgbm in the base set but do not pre-weight it -- let the
   meta-learner decide. If its OOF probabilities are diverse and
   informative, meta will pick it up; if not, weight goes to ~0.
3. **player_stats: both signals agree it is bad.** Drop from the base set
   or zero its meta weight.

Final submission selection still follows the design spec's rule: pick the
version with the highest **local-CV** overall, not the highest public LB.
Public is a sanity check (if local and public diverge by more than ~0.05,
suspect a training-pipeline bug) and a tie-breaker, nothing more.

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
baseline (mean overall 0.3013) plus **two** std numbers:

- `std_across_seed.overall = 0.00168` — std of the 5 per-seed mean overalls.
  **This is the noise floor** per spec Section 1.2. Reject any downstream
  improvement smaller than this.
- `std_across_fold.overall = 0.00635` — std of all 25 (seed, fold) cells.
  Larger because it includes within-seed fold-to-fold variance. Useful for
  spotting unstable folds, NOT the noise floor.

Per-target noise floors (across-seed std):
- action macro-F1: 0.00525
- point macro-F1:  0.00506
- server AUC:      0.00397

Smoothing trick (old-test `serverGetPoint` overlap → 0.95/0.05) stays out
of every new model's training and OOF. It is only applied when generating
the `submission_FINAL_smooth.csv` public-leaderboard backup in Section 5
of the design spec.

## Plan execution status

The active plan is broken into 5 files under `docs/superpowers/plans/`
(P1 CV foundation → P5 final ensemble). See `PROGRESS.md` for current
task status and known plan-review issues to fix during execution.
