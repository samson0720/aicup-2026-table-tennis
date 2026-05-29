# AI Cup 2026 — Private-Score Push v2 (Design)

Status: Draft for user review (2026-05-29).
Follows `2026-05-28-aicup-private-score-push-design.md` (CatBoost shipped +0.0032;
TabPFN/seq/depth-tuning rejected). Production honest overall = **0.3238**.

## 1. Goal and honest feasibility

Maximize the **honest per-row private** score above 0.3238. Public-smooth (the
`serverGetPoint` old-test leak, ~0.41 public) is **deliberately deferred to the
final submission** per the user's strategy — out of scope here.

**Hard reality (verified this session):** the next stroke is genuinely unobserved
(old test data does NOT reveal the target action/point — 0/500 overlap rallies
have it; it only leaks the rally's `serverGetPoint`, already exploited). Six model
families all plateau at action 0.24–0.29 / point 0.16–0.19 → we are near the
**information ceiling**, not the modeling ceiling. So **no large private jump is
expected**; the aim is to squeeze every honest point. Realistic target: cumulative
**+0.005 … +0.02** if multiple levers land; each independently gated, stop when
returns die (as with CatBoost/TabPFN).

Metric: `overall = 0.4·action_macroF1 + 0.4·point_macroF1 + 0.2·server_AUC`.
Macro-F1 weights every class equally → **rare action/point classes are the single
biggest untapped lever** (we currently do prior-correction + threshold tuning, but
no resampling / rare-class loss).

## 2. Methodology and overfit guardrails

- **Honest per-row ruler only** (no seed-averaging). cv_splits drives all OOF cuts.
- **Adding a base** (Phases 1–2): gate on the existing nested-CV ensemble overall
  (`build_final_perrow`) — ship only if it lifts overall by **> 0.00168**. This is
  already honest (nested GroupKFold), so no extra holdout needed for bases.
- **Learning ensemble weights** (Phase 3) is the overfit-prone step. Guard it with
  a **seed-55 meta-holdout**: fit/select the stacker on seeds 11/22/33/44 only,
  report the final overall on seed 55 (untouched by selection). Adopt the learned
  stacker only if it beats the current LR meta on the seed-55 holdout *and* on the
  4-seed selection score. (All bases already have 5-seed OOF, so no recompute.)
- Every surviving base plugs into `build_final_perrow` via the standard
  `oof/<model>_<target>.parquet` + `_test.parquet` contract.

## 3. Phase 1 — Macro-F1 rare-class lever (do first; most on-metric)

The metric rewards rare-class recall; we've never resampled. Produce **rebalanced
variants of the strongest tree bases** (CatBoost and lgbm) that up-weight rare
action/point classes, then gate them as additional bases.

- **3.1 Adaptive / static resampling**: oversample rare action/point classes in the
  per-fold training set (ART-style class-wise emphasis on underperforming classes,
  or a simpler fixed rare-class oversample). CatBoost/LGBM already support
  `class_weights`; resampling is the new lever.
- **3.2 Loss**: where supported, a focal / class-balanced objective for the GBDTs;
  for any neural base, focal loss.
- **3.3 Gate**: add the rebalanced base (`cat_bal`, `lgbm_bal`) to BASES; require
  ensemble lift > noise floor. Keep only what clears.

Rationale: directly attacks point (0.187, weakest, 0.4 weight) and action (0.4
weight) where rare classes drag macro-F1 down.

## 4. Phase 2 — Diverse GPU bases

CatBoost (+0.0032) proved a genuinely different model class helps the stack. Add:

- **4.1 XGBoost (GPU)** — already installed and GPU-verified; a different tree
  algorithm from LGBM/CatBoost. Mirror the CatBoost producer/test-inference path
  (`produce_*_oof` + `predict_test_*`), per-row OOF + full-train test.
- **4.2 FT-Transformer (GPU neural-tabular)** — runs on the engineered prefix
  tabular features (distinct from the seq model's raw sequences). Train with class
  weighting / focal loss (the lesson from TabPFN's majority-bias failure). Implement
  via a small in-repo FT-Transformer (or `rtdl`/`pytorch-tabular` if it installs
  cleanly into a cloned env, same isolation pattern as TabPFN).
- **4.3 Gate**: each added independently; keep only above-floor contributors.

## 5. Phase 3 — Smarter stacking (holdout-gated)

Replace/augment the fixed LR meta with **hill-climbing / greedy weighted ensemble
selection** over all surviving bases (research: extracts more than a fixed meta).
Fit/select on seeds 11/22/33/44; **validate on the seed-55 holdout**. Adopt only if
it beats the current LR stacker on the holdout. If it doesn't generalize, keep LR.

## 6. Success criteria

- Per lever (hard): ships only if it clears the 0.00168 floor (Phases 1–2 on
  nested-CV; Phase 3 on the seed-55 holdout).
- Soft: cumulative honest overall meaningfully above 0.3238 (target +0.005…+0.02).
- A fully sub-noise outcome is a valid documented result (the ceiling is real).

## 7. Risks and constraints

- **Information ceiling** — the dominant risk; gains may be small or nil. Accepted.
- **Overfitting the ruler** — mitigated by the nested-CV floor (bases) and seed-55
  holdout (stacker). No tuning grids scored on the same CV that selects.
- **Env safety** — any pip-installed neural lib goes in a cloned env
  (`aicup-tt-*`), never the main `aicup-tt` (the TabPFN torch-2.12/GLIBCXX lesson).
- **GPU** — RTX 3090 via `CUDA_VISIBLE_DEVICES=0`. Compute is unconstrained.
- **conda only**; honest per-row scoring; no seed-averaging.

## 8. Out of scope (YAGNI)

- Public-smooth `serverGetPoint` leak — reserved for the final submission, not here.
- Pseudo-labeling — research shows it's weak for tabular (features ≈ label).
- Recovering action/point from old test — verified impossible (next stroke unobserved).
- Massive 70-model zoos — overfit risk outweighs ceiling-limited upside; add bases
  deliberately and gate each.
