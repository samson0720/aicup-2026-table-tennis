# AI Cup 2026 — Private-Score Push (Design)

Status: Draft for user review (2026-05-28).
Builds on the per-row honest ruler (`build_final_perrow.py`) and the rejected
sequence-model pilot (`2026-05-28-aicup-sequence-model-pilot-design.md`).

## 1. Goal and honest feasibility

Push the **private/honest** score, not the public-smooth number. Current honest
state (per-row ruler, the only trustworthy one):

- Honest ensemble overall = **0.3206** (action macro-F1 0.2894, point macro-F1
  0.1842, server AUC 0.6557). Best single base lgbm15 = 0.3027.
- Metric: `overall = 0.4·action_macroF1 + 0.4·point_macroF1 + 0.2·server_AUC`.
- Noise floor (across-seed std on overall) = **0.00168**. Any base/ensemble
  change below this is treated as noise and rejected.

The weakest target with the most headroom is **point** (0.1842, 0.4 weight).
action (0.2894) is already heavily worked (seq added nothing above noise).
server (0.2 weight) has limited honest headroom. So point and action are the
levers; the plan attacks them with genuinely *different, strong model classes*
rather than refinements to the existing LGBM family (the seq2 lesson: refining a
model class that's already represented yields sub-noise diversity).

**Honest framing:** like the seq pilot, any single prong may end below the noise
floor. That is an acceptable, documented negative — the goal is information +
any above-noise gain, not a guaranteed score jump.

## 2. Strategy: three independently-gated prongs

Each prong produces the standard artifacts and plugs into the existing ensemble
with no architectural change:

- OOF: `artifacts/oof/<model>_<target>.parquet` (schema from `oof_loader.py`:
  `rally_uid, seed, fold, cut_strikeNumber` + `p_*`), 74,975 rows over the 25
  `cv_splits.parquet` cells.
- Test: `artifacts/oof/<model>_<target>_test.parquet` (1,845 rows, `rally_uid` +
  `p_*`), single-cut per rally, distribution-matched to test.
- Integration: add the model to `BASES` in `build_final_perrow.py`, rebuild,
  score honest per-row.

**Gate (hard):** a base ships only if adding it lifts the honest ensemble
`overall` by **> 0.00168** vs the current 0.3206. Expensive prongs run a
**pilot slice first** (seed 11 × folds 0–2), exactly like the seq pilot, before
committing the full 25-fold run.

**Discipline:** no seed-averaging anywhere; `cv_splits.parquet` drives every OOF
cut; test inference is single-cut and distribution-matched (same scale of
training data as the OOF models, per the seq2 fix).

**Sequencing:** A (CatBoost) → B (TabPFN) → C (refinements). Stop early if a
prong clears the floor by a wide margin and the next is redundant; continue if
gains are marginal.

## 3. Prong A — CatBoost base (first; lowest risk, best EV/hour)

**Why:** the data is heavily categorical (strikeId, handId, strengthId, spinId,
positionId, prefix actionId/pointId, gamePlayerId/Other, sex). CatBoost's native
ordered-target-statistics encoding handles categoricals without the manual
target-encoding leak Route B fought, and it is a different booster family from
LGBM → real ensemble diversity. conda-forge + GPU on the 3090.

**Install:** `conda install -n aicup-tt -c conda-forge catboost`. Verify it does
not disturb the working PyTorch/CUDA (`torch.cuda.is_available()` still True).

**Featurization:** reuse the *existing* tabular prefix-feature pipeline that
feeds the LGBM bases (resolve the exact producer in planning by reading
`scripts/produce_base_oof.py` / the lgbm trainer), but pass categorical columns
natively via CatBoost `cat_features` instead of pre-encoding. This keeps the
comparison apples-to-apples (same features, different learner).

**Models:** three CatBoostClassifiers — action (`classes_count=19`), point
(`classes_count=10`), server (binary). `task_type='GPU'`, devices set so the
3090 is used. Verify GPU multiclass predicts all classes (historical CatBoost
GPU-multiclass bug); fall back to CPU multiclass for action/point if any class
is dropped.

**OOF + test:** OOF over the 25 cv cells; full-train single-seed test inference
(distribution-matched, like seq2). Write standard parquets as `cat_*`.

**Gate:** pilot slice first; then full 25-fold; integrate and require ensemble
lift > noise floor.

## 4. Prong B — TabPFN v2 base (second; constrained, point + server only)

**Why:** a tabular foundation model (in-context transformer) benchmarked to beat
tuned GBDT ensembles on small data — the highest-diversity novel base. point
(10 classes, the highest-headroom target) is squarely in its wheelhouse.

**Constraints (drive the scope):**
- Only **TabPFN v2** is cleanly open (Apache-2.0). v2.5/2.6/3 require PriorLabs
  browser login + non-commercial license → out of scope for automated runs.
- Install is **pip-only** (`pip install tabpfn` *into the aicup-tt env*, never
  `--user`/system). This deviates from "conda-forge only"; **env safety first**:
  smoke-import TabPFN and re-verify `torch.cuda.is_available()` before any real
  run. If the pip install threatens the PyTorch/CUDA stack, clone the env
  (`conda create --clone aicup-tt -n aicup-tt-tabpfn`) and run Prong B there.
- v2 classification ≈ **10-class cap** → use TabPFN for **point (10) and server
  (2)** only; CatBoost (Prong A) owns the 19-class action.
- v2 sample sweet-spot ≈ 10k. Per-fold OOF training context (~12k) exceeds it:
  **subsample the in-context training set to TabPFN's supported size per fold**
  (optionally average a few subsample seeds). Resolve exact cap in planning.

**Featurization:** reuse `artifacts/prefix_train_v2.parquet` (74,975 × 322),
capped to TabPFN's feature limit (select top features if needed). GPU.

**OOF + test:** standard parquets as `tabpfn_point` / `tabpfn_server`.

**Gate:** pilot slice on point first (cheapest signal on the weakest target);
proceed only if it shows above-noise ensemble lift.

## 5. Prong C — loss / threshold refinements (third; cheapest, incremental)

Targeted, bounded refinements on existing model classes — no new architecture:
- **Loss:** focal / class-balanced cross-entropy (and/or tuned class weights)
  for the seq trainer and any new neural base, to lift minority-class macro-F1.
- **Decision layer:** revisit threshold tuning / probability calibration
  (isotonic/Platt) for the GBDT bases feeding the stack.

Keep only components that clear the floor on honest CV. Expect incremental
results; this prong is a tail, not the headline.

## 6. Integration and submission policy

- Add each *surviving* base to `BASES` in `build_final_perrow.py`; rebuild;
  record the honest ensemble overall and the lift vs 0.3206 in PROGRESS.md.
- Final submission = the variant with the highest **honest local** overall.
- Public LB is daily-limited and teammate-shared: **at most one upload**, and
  only if the honest local lift exceeds the 0.00168 noise floor. Public is a
  tie-breaker, never a model-selection oracle.
- Smoothing stays an end-only public-sprint step, untouched by this work.

## 7. Success criteria

- **Per-prong (hard):** a base ships iff ensemble overall lifts by > 0.00168.
- **Aspirational (soft):** if 1–2 prongs land, push honest overall from 0.3206
  toward ~0.33–0.34. Not guaranteed; a fully sub-noise outcome (like seq2) is a
  valid, documented result.
- **Process:** every prong scored on the honest per-row ruler; pilot-before-full
  on expensive prongs; PROGRESS.md updated per task.

## 8. Risks and constraints

- **Env breakage:** new installs must not break PyTorch+CUDA. Mitigation:
  CatBoost via conda-forge first (clean); TabPFN pip-import smoke + cuda re-check;
  clone the env for TabPFN if risky. Never touch base/system Python.
- **CatBoost GPU multiclass bug (historical):** verify all classes predicted;
  CPU multiclass fallback for action/point.
- **TabPFN caps/license:** open v2 only, point+server only, subsample context.
- **Sub-noise outcomes:** acceptable; reject from production, keep artifacts +
  code committed for reproducibility (as done for seq2).
- **GPU:** RTX 3090 is PyTorch/CatBoost device 0 via `CUDA_VISIBLE_DEVICES=0`
  (CUDA fastest-first ordering; `=1` is the GTX 1080 — do not use).
- **conda only:** all runs via `conda run -n aicup-tt ...`.

## 9. Out of scope (YAGNI)

- TabPFN v2.5/3 (license/login friction) and TabPFN on the 19-class action.
- New raw features beyond the existing prefix representation (coverage is
  complete; this is a model-class diversity play, not a feature play).
- Large Optuna/HPO sweeps (small targeted tuning only).
- Replacing LGBM/chain bases — new bases are *added*, not substitutes.
- Public-smooth optimization (separate end-step, not this private push).
