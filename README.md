# AI CUP 2026 Table Tennis Rally Forecasting

This repository contains the training, validation, ensembling, and submission
pipeline for the AI CUP 2026 table-tennis time-series prediction task.

The task is to predict the next stroke and rally outcome from an observed rally
prefix:

- `actionId`: next-stroke action, 19-class macro-F1
- `pointId`: next-stroke landing/result point, 10-class macro-F1
- `serverGetPoint`: whether the server wins the rally, binary AUC

The official overall score is weighted:

```text
score = 0.4 * action_macro_f1 + 0.4 * point_macro_f1 + 0.2 * server_auc
```

## Repository Layout

```text
AI CUP競賽資料集/
  train.csv
  test_new.csv
  sample_submission.csv
  Reference_Only_Old_Test_Data/

scripts/
  Feature building, CV split generation, model training, OOF production,
  stacking, post-processing, diagnostics, and submission builders.

tests/
  Unit tests for CV splits, target encoding, scoring, post-processing,
  sequence datasets/models, and final-rule helpers.

artifacts/
  Checked-in reports, logs, scores, CV metadata, and final submission CSVs.
```

Large reproducible intermediates are intentionally ignored by git:

- `artifacts/*.parquet`
- `artifacts/oof/*.parquet`
- `artifacts/*.pkl`
- `artifacts/*.joblib`
- `another_data/`

This means the final checked-in submission files are present, but a full
from-scratch rebuild requires regenerating OOF/test probability parquet files
and restoring any optional external data under `another_data/`.

## Environment

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate aicup-tt
```

The environment file currently specifies Python 3.11 and pinned major package
families for pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, PyTorch,
Optuna, and pytest.

## Data

The main data directory is expected to match `AI CUP*` from the repository root.
The current project uses:

- `AI CUP競賽資料集/train.csv`
- `AI CUP競賽資料集/test_new.csv`
- `AI CUP競賽資料集/Reference_Only_Old_Test_Data/test.csv`

The old reference test file contains leakage-like information and is not the
formal private evaluation set. The pipeline therefore keeps two final submission
styles:

- `submission_FINAL_safe_perrow.csv`: model-only, private-safe submission.
- `submission_FINAL_leakmax.csv`: public-oriented variant using old-test overlap
  information.

For the current checked-in data:

- train rows: 84,707
- train rallies: 14,995
- test_new rows: 5,668
- test_new rallies: 1,845
- old-test overlap with test_new: 1,236 rallies

## Modeling Approach

The project converts each rally into prefix-to-next-stroke samples. For a target
stroke at `cut_strikeNumber`, only rows with smaller `strikeNumber` are used as
features.

Key feature groups include:

- phase features: receive, third-ball, later rally
- score context: score difference, score sum, close/deuce-like indicators
- first observed strokes: `obs1_*`, `obs2_*`, `obs3_*`
- recent strokes: `last1_*` through `last5_*`
- prefix count/rate/entropy features
- inferred next-player features
- OOF-safe target encoders
- n-gram and transition features
- optional displacement/pressure features

The final system is a two-level ensemble:

1. Base models produce OOF and test probabilities for each target.
2. A per-target LogisticRegression stacker learns from concatenated base
   probabilities.

The current production base list is defined in:

```text
scripts/build_final_perrow.py
```

The final stack uses 21 action bases, 21 point bases, and 18 server bases.

## Cross-Validation

Private-safe CV is implemented in:

```text
scripts/cv_splits.py
```

The split design:

- 5 seeds x 5 folds
- group by `match`
- one random cut point per rally per seed
- phase-stratified fold assignment

The fixed split metadata is stored as:

```text
artifacts/cv_splits.parquet
```

This file is ignored by git, so regenerate it when rebuilding from scratch:

```bash
python -m scripts.cv_splits
```

## Common Commands

Run tests:

```bash
pytest
```

Generate CV splits:

```bash
python -m scripts.cv_splits
```

Run the baseline LightGBM CV:

```bash
python -m scripts.train_lgbm_baseline
```

Produce base OOF files:

```bash
python -m scripts.produce_base_oof --model lgbm15
python -m scripts.produce_base_oof --model lgbm31
python -m scripts.produce_base_oof --model markov
python -m scripts.produce_base_oof --model phase_lgbm
```

Train full-data base models and predict `test_new.csv`:

```bash
python -m scripts.predict_test_base --model lgbm15
python -m scripts.predict_test_base --model lgbm31
python -m scripts.predict_test_base --model markov
python -m scripts.predict_test_base --model phase_lgbm
```

Build the final per-row stack and safe submission:

```bash
python -m scripts.build_final_perrow
```

Build the public-oriented leakmax submission:

```bash
python -m scripts.build_leakmax_submission
```

## Final Outputs

The checked-in final submissions are:

```text
artifacts/submission_FINAL_safe_perrow.csv
artifacts/submission_FINAL_leakmax.csv
```

`submission_FINAL_safe_perrow.csv` uses model predictions only.

`submission_FINAL_leakmax.csv` starts from the public-oriented smoothed variant
and additionally overrides `pointId` on the 1,236 old-test-overlap rallies using
leak-feature point models. It also uses hard `serverGetPoint` values of `1.0`
or `0.0` for the overlap rallies.

Use `safe_perrow` when reporting private-safe modeling performance. Use
`leakmax` only when intentionally optimizing for public-overlap behavior and
clearly disclose the overlap handling.

## Current Local Scores

Current final per-row ensemble score:

```json
{
  "action_macro_f1": 0.3723943337345462,
  "point_macro_f1": 0.2704900381136824,
  "server_auc": 0.7052797049334245,
  "overall": 0.39820968972597637
}
```

The score is stored in:

```text
artifacts/final_perrow_scores.json
```

## Notes On Reproducibility

Several scripts depend on large ignored intermediates and optional external data.
For a clean rebuild, generate artifacts in this order:

1. Generate `artifacts/cv_splits.parquet`.
2. Produce all required base OOF files under `artifacts/oof/`.
3. Produce corresponding test probability parquets under `artifacts/oof/`.
4. Run `scripts.build_final_perrow`.
5. Optionally run `scripts.build_leakmax_submission`.

The `*_extra` model families require `another_data/train.csv`, which is excluded
from version control.

## Important Caveats

- Do not average different seed cut points for the same rally before stacking.
  The final builder uses per-row stacking because test time has only one prefix
  per rally.
- Keep `safe_perrow` and `leakmax` results separate. They answer different
  evaluation questions.
- The old reference test data contains leakage-like labels. Treat any submission
  using old-test overlap logic as public-oriented rather than private-safe.
