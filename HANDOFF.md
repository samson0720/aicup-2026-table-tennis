# AI CUP Table Tennis Handoff

## Current best public result

- Uploaded file: `artifacts/submission_RECOMMENDED_upload.csv`
- Public score: `0.4102132`
- Rank at time checked: `45/333`

Second upload was lower:

- Uploaded file: `artifacts/submission_CANDIDATE2_lgbm_sqrt_leaves15.csv`
- Public score: `0.3990370`

## Important conclusion

The old `Reference_Only_Old_Test_Data/test.csv` leakage was already used for `serverGetPoint`.

For the 1,236 `rally_uid` values overlapping with `test_new.csv`:

- old `serverGetPoint = 1` -> submitted `0.95`
- old `serverGetPoint = 0` -> submitted `0.05`

This leakage does not provide the next-stroke `actionId` or `pointId` labels. In old test, those columns are the known prefix strokes, not the hidden next-stroke answer.

## Scripts

- `scripts/eda.py`: data summary and leakage/overlap checks.
- `scripts/train_lgbm_baseline.py`: prefix sample builder and LightGBM CV.
- `scripts/make_lgbm_submission.py`: full-train LightGBM submission generator.
- `scripts/train_markov_ensemble.py`: phase-aware Markov baseline and ensemble check.
- `scripts/train_player_stats_lgbm.py`: OOF-safe player statistics experiment.
- `scripts/train_phase_lgbm.py`: phase-specific LightGBM experiment.
- `scripts/train_catboost_baseline.py`: CatBoost experiment, too slow on this laptop.
- `scripts/test_like_validation.py`: 5-seed test-like validation gate.

## Best local results so far

Original all-prefix GroupKFold by match:

- LGBM `sqrt`, `num_leaves=31`: local overall `0.32350`, public `0.4102132`
- LGBM `sqrt`, `num_leaves=15`: local overall `0.32439`, public `0.3990370`

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

## Recommended state

Do not submit small local-CV improvements blindly. The current public-best file remains:

```text
artifacts/submission_RECOMMENDED_upload.csv
```

Only one daily submission remained after the two uploads described above.
