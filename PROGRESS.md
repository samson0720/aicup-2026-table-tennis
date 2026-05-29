# AI Cup 2026 — Score-Improvement Plan Execution Progress

Status as of 2026-05-28. Updated by the active AI agent after every plan task.
Source-of-truth for the next agent: read this BEFORE touching code.

## Private-push v4 (2026-05-30) — parallel multi-bet + public leak expansion

Spec `docs/superpowers/specs/2026-05-30-aicup-private-push-v4-design.md`, plan
`docs/superpowers/plans/2026-05-30-aicup-private-push-v4.md`. Five levers run in
parallel, each gated honest nested-CV vs production **0.325678** (> 0.00168 floor):
L1 ShuttleNet neural base, L2 focal-loss GBDT, L3 higher-order Markov, L4 (action,
point) joint, L5 public leak. Baseline reproduced exactly (0.32567846) after the P1
refactor — pipeline intact.

### v4 L3 — higher-order player×context Markov (`markov2`) — REJECTED (2026-05-30)

`scripts/produce_markov2_oof.py`: Dirichlet backoff global → last1 →
(last1,last2) → (player,last1) → (player,last1,last2), α=8. `last2_*` features
already existed in `build_one_sample_per_rally` (no feature-builder change). OOF
74975 rows/target, test 1845.

| config | action | point | server | overall | lift |
|---|---:|---:|---:|---:|---:|
| production (markovp) | 0.29633 | 0.18954 | 0.65667 | **0.325678** | — |
| + markov2 | 0.29730 | 0.18598 | 0.65667 | 0.324644 | **−0.00103** |

**VERDICT: REJECT.** Net negative. action +0.00097 (deeper context helps a bit) but
point −0.00356 (the (player,2-gram) cells are too sparse → overfit/noise on point).
Exactly the spec's flagged redundancy-with-markovp risk: markovp already captures the
(player,last1) signal; the extra 2-gram depth adds noise, not signal. BASES reverted
to the 7-base markovp set; production stays **0.325678**. `produce_markov2_oof.py` +
OOF/test parquets kept for reproducibility (NOT in BASES).

### v4 L2 — focal-loss GBDT (`lgbm_focal`) — REJECTED (2026-05-30)

`scripts/focal_loss.py` (multiclass focal custom objective, LightGBM 4.x params-
objective) + `scripts/produce_lgbm_focal_oof.py`. Two implementation bugs found and
fixed during bring-up: (1) scaling the hessian by the focal factor → underfit
(trees won't split, standalone macro-F1 ~0.04); (2) bare p*(1-p) hessian → 0 at
saturation → runaway leaf values (raw margins ~5e5, one-hot probs, ~0.14). Fixed by
flooring the standard softmax hessian at 0.1 (commits 5fddd71, dfaf8c6). γ=2.

Standalone (full OOF, floored hessian): action **0.2191**, point **0.0967** — below
every existing GBDT base (cat 0.2716/0.1739, lgbm15 0.2587/0.1730).

| config | action | point | server | overall | lift |
|---|---:|---:|---:|---:|---:|
| production | 0.29633 | 0.18954 | 0.65667 | **0.325678** | — |
| + lgbm_focal (action+point) | 0.29489 | 0.18949 | 0.65667 | 0.325087 | **−0.00059** |

**VERDICT: REJECT.** Net negative (action −0.00144, point flat). focal loss did not
deliver the literature's macro-F1 gain on this 19-class / ~12k-rows-per-fold /
heavily-engineered-feature setup — it underfits/destabilizes relative to the sqrt
class-weighted GBDTs, and as a same-class (LightGBM) base adds no ensemble diversity.
γ=1 not pursued: action+point both clearly weaker standalone, so a second γ cannot
flip the diversity verdict (the depth-8/redundancy lesson). BASES reverted; production
stays **0.325678**. Scripts/parquets kept for reproducibility (NOT in BASES).

### v4 L1 — ShuttleNet-style neural base — PILOT GREEN, PROCEED (2026-05-30)

`scripts/shuttle_model.py` (`ShuttleForecaster`: rally-progress enc + player-style
enc + position-aware gated fusion, action+point heads only) + `scripts/train_shuttle.py`
(class-weighted CE, AMP, early stop). Pilot seed11×folds0-2 (8960 rows), seq winning
config (d256/l4/h4/ffn768/do0.2/lr3e-4, 50ep patience12, early-stopped ~ep23).

| model | action | point | (slice) |
|---|---:|---:|---|
| lgbm15 (same slice) | 0.2551 | 0.1619 | — |
| seq pilot (prior, rejected) | 0.2425 | 0.1618 | — |
| **shuttle pilot** | **0.2681** | **0.1828** | beats both GBDT + seq |

**VERDICT: PROCEED.** First neural model COMPETITIVE on this task — beats lgbm15 on
both targets on the pilot slice (action +0.0130, point +0.0209), unlike the generic
seq Transformer (rejected). The ShuttleNet inductive bias (dual encoder + player-style
embeddings + position-aware gated fusion) is the difference. Proceeding to L1.4: full
25-fold OOF + test + integration A/B gate vs 0.325678. NOTE: data-loading-bound
(GPU idle, RallyPrefixDataset single-thread); add DataLoader num_workers for the full run.

### v4 L4 — structured (action, point) joint base (`joint`) — REJECTED (2026-05-30)

`scripts/produce_joint_oof.py`: OOF-safe smoothed P(point|action) (Dirichlet α=4,
fit per fold-train) marginalized over cat's action OOF, P(point)=Σ_a P(point|a)·P̂(a).
point-only base. OOF 74975, test 1845. Evaluated through the full downstream pipeline.

| config | action | point | server | overall | lift |
|---|---:|---:|---:|---:|---:|
| production | 0.29633 | 0.18954 | 0.65667 | **0.325678** | — |
| + joint (point) | 0.29633 | 0.18775 | 0.65667 | 0.324966 | **−0.00071** |

**VERDICT: REJECT.** point −0.00179. The marginalized joint smears the point
distribution (averaging conditionals weighted by an imperfect P̂(a)); `chain_point`
and `cat` already model the action→point dependence more sharply, so the joint adds a
weaker, redundant point signal that the stacker can only down-weight or be hurt by.
BASES reverted; production stays **0.325678**. Scripts/parquets kept (NOT in BASES).

## Private-push v3 — P1 (macro-F1 decision rule) — REJECTED (2026-05-29)

Spec `docs/superpowers/specs/2026-05-29-aicup-private-push-v3-design.md`, plan
`docs/superpowers/plans/2026-05-29-aicup-p1-decision-rule.md`. Pluggable decision
rules in `scripts/decision_rule.py`; gate harness `scripts/eval_decision_rule.py`
(nested-CV ruler + seed-55 holdout). Honest per-row throughout.

**The gate harness FLAGGED a false positive that the production A/B caught.**

Stage 1 — gate harness (`additive_baseline` used honest per-fold priors):

| rule | action nested | action holdout | point nested | overall | lift |
|---|---:|---:|---:|---:|---:|
| additive_baseline | 0.2863 | 0.2821 | 0.1866 | 0.32050 | — |
| additive_wide | 0.2859 | 0.2677 | 0.1900 | 0.32173 | +0.00123 |
| calibrated | 0.2927 | 0.2865 | 0.1881 | 0.32367 | +0.00317 |
| weighted | 0.2859 | 0.2830 | 0.1866 | 0.32034 | -0.00017 |

The harness picked `calibrated` (+0.00317, holdout-robust). BUT its `additive_baseline`
scored only 0.32050 because it uses **honest per-fold priors**, whereas the REAL
production rule uses the **full-data prior** (a stable, stronger baseline).

Stage 2 — production A/B (the real `build_final_perrow`, full-y prior; the
`AICUP_PROD_RULE` env var swaps the rule):

| rule | action | point | server | overall |
|---|---:|---:|---:|---:|
| **additive_baseline (= old prod)** | 0.2963 | 0.1895 | 0.6567 | **0.32568** |
| calibrated | 0.2927 | 0.1881 | 0.6567 | 0.32366 (**-0.00202**) |

**VERDICT: REJECT.** `calibrated` ignores the prior (isotonic), so it scores ~0.3237
under both prior conventions — it beat the weak per-fold-prior harness baseline but
LOSES to the real full-data-prior production rule by -0.00202. Production stays
`additive_baseline`, overall **0.32568** (unchanged). `additive_wide` also failed
(overfit: action holdout 0.2821→0.2677); `weighted` was sub-noise.

**LESSON (for P2-P4 and future gates):** a gate harness's baseline MUST reproduce the
exact production rule, including the prior convention. The per-fold-prior baseline
under-measured production by ~0.005 and produced a false SHIP. Always confirm a SHIP
in the real `build_final_perrow` A/B before trusting a standalone harness. The
`additive_baseline` rule in the refactored builder is asserted byte-equivalent to the
legacy pipeline (`tests/test_build_final_rule.py`) and the A/B reproduced 0.32568
exactly, so the refactor itself is safe/behaviour-preserving.

The pluggable decision-rule refactor (`decision_rule.py`, `eval_decision_rule.py`,
the rule-factory `_nested_f1`) is KEPT as reproducible tooling for later phases, with
`PROD_RULE` defaulting to `additive_baseline`. Untried-but-plausible follow-up (not
pursued — ruler-overfit risk + diminishing returns): a `calibrated` variant that
prior-corrects BEFORE isotonic, to combine the prior boost with calibration.
Artifacts: `artifacts/decision_rule_scores.json`, `artifacts/p1_build_{baseline,calibrated}.log`.


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

**VALIDATED on public LB (2026-05-28):** `submission_FINAL_safe_perrow.csv`
scored **0.3492438** clean — beats the previous best clean single model
leaves31 (0.3364) by +0.013 and the old mismatched ensemble (~0.32) by +0.029.
First time the ensemble actually wins on public clean. Confirms the honest
ruler: local 0.3206 -> public 0.3492 (offset +0.029) matches the base offset
(lgbm15 local 0.3027 -> public 0.3263, +0.024). Local and public now agree on
direction AND magnitude. The smooth_perrow variant (untested) should be >= the
old 0.4102 smooth record if public-LB maximization is wanted.

Evidence scripts (re-runnable): `scripts/diag_threshold_honesty.py` (rules out
in-sample threshold tuning as the cause, only +0.005), `scripts/diag_perrow_vs_seedavg.py`
(isolates the seed-averaging inflation, server 0.6525 per-row vs 0.7635 seed-avg).

NEXT: use the per-row score (0.3206) as the honest ruler for ANY future "make it
stronger" work. Never compare a seed-averaged number against an all-cuts number.

## Sequence-model pilot results (2026-05-28)

Plan: `docs/superpowers/plans/2026-05-28-aicup-sequence-model-pilot.md`,
Phase 1 Tasks 1-6. Slice: seed 11 x folds 0-2, 8,960 per-row examples. All
numbers below use honest per-row scoring; no seed averaging.

Winning pilot config: `--d-model 256 --layers 4 --nhead 4 --ffn 768
--dropout 0.2 --batch-size 256 --lr 3e-4`. Best monitor epochs were fold0=26,
fold1=34, fold2=48 (median 34).

| model      | action F1 | point F1 | server AUC | overall |
|------------|----------:|---------:|-----------:|--------:|
| lgbm15     | 0.2551    | 0.1619   | 0.6486     | 0.2965  |
| seq best   | 0.2425    | 0.1618   | 0.6166     | 0.2850  |

Standalone seq is not better than lgbm15, but it adds diversity to the stack:
existing 5 bases overall **0.3115** vs existing+seq best **0.3168**, lift
**+0.0054**. This is above the 0.00168 noise floor (~3.2x).

VERDICT: **GREEN** by the planned ensemble-lift gate. Decision: Phase 2 is
eligible, but execution is paused here pending human approval per the handoff
instruction. If Phase 2 is approved, start with Task 8's 15-minute design
confirmation before coding test-time inference.

## Sequence-model PHASE 2 result — seq2 REJECTED (2026-05-28)

Phase 2 (plan Tasks 7-10) executed after the GREEN pilot. Honest per-row
ruler throughout; no seed-averaging.

- **Task 7 — full 25-fold seq2 OOF** (`seq2_{action,point,server}.parquet`,
  74,975 rows each; winning config d-model 256, ffn 768, dropout 0.2; 25/25
  cells early-stopped, median best epoch 33). Standalone honest (argmax):
  action 0.2394, point 0.1708, server AUC 0.6262, overall **0.2893** — below
  lgbm15 0.3027. Commit `ee3e639`.
- **Task 8 — test inference**: added `--predict-test` to
  `scripts/train_seq_pilot.py` (full-train single-seed model, fixed 33 epochs,
  no held-out; synthetic-cut test dataset reusing `RallyPrefixDataset`
  unchanged — cut = max observed strikeNumber + 1, prefix = all observed
  strokes; `test_new.csv` has no `serverGetPoint`, so a placeholder label is
  added and ignored). Writes `seq2_{target}_test.parquet` (1845 rows, base
  schema). `tests/test_seq_test_inference.py` green. Commit `9af9db5`.
- **Task 9 — integration**: added seq2 to `build_final_perrow.py` BASES and
  rebuilt. seq2 OOF keys verified identical to the other bases (inner-join
  keeps all 74,975 rows, so the comparison is apples-to-apples).

| config | action | point | server | overall | lift vs 0.3206 |
|---|---:|---:|---:|---:|---:|
| 5-base (production) | 0.28938 | 0.18416 | 0.65569 | **0.32056** | — |
| +seq2 (all 3 targets) | 0.29186 | 0.18361 | 0.65674 | 0.32154 | +0.00098 |
| +seq2 (action only)   | 0.29186 | 0.18416 | 0.65569 | 0.32155 | +0.00099 |

**VERDICT: seq2 REJECTED from the production ensemble.** Honest overall lift is
+0.00098 (all-3) / +0.00099 (action-only), both **below the 0.00168 noise
floor**. seq2's only real effect is action +0.00248, itself below the
action-specific noise floor (0.00525). The pilot slice GREEN (+0.0054 on
seed11×folds0-2) did NOT generalize to the full population — exactly what the
across-seed noise floor is designed to catch. Per the project rule (reject
sub-noise changes; upload only if local lift > noise floor), the **5-base
ensemble (overall 0.3206) remains the production submission**; no public upload.

`build_final_perrow.py` BASES reverted to the 5-base set and the production
submissions/scores restored to the 5-base versions. The seq2 OOF + `_test`
parquets and the `--predict-test` code are kept (committed) for
reproducibility; `scripts/train_seq_transformer.py` was left untouched.

## CatBoost base result — Prong A SHIPPED (2026-05-29)

Private-score push, prong A of `2026-05-28-aicup-private-score-push-design.md`
(plan `2026-05-28-aicup-catboost-base.md`). Honest per-row ruler throughout.

- **New base `cat`**: CatBoost over the same prefix-feature pipeline as the LGBM
  bases (native categoricals via `cat_features`), per-row OOF on the 25 cv cells
  + full-train test inference. Files: `scripts/produce_catboost_oof.py`,
  `scripts/predict_test_catboost.py`, GPU params added to
  `scripts/train_catboost_baseline.py` (now importable as a module),
  `catboost=1.2.*` added to `environment.yml`.
- **GPU pivot**: the plan specified CPU (historical CatBoost GPU-multiclass
  class-dropping bug), but CPU was impractically slow (~hours for 19-class
  multiclass × 25 cells). Verified catboost **1.2.10 has no such bug** (19-class
  GPU fit keeps all 19 classes, 4.9s), so the prong runs on the 3090
  (`--gpu`, `CUDA_VISIBLE_DEVICES=0`). ~minutes instead of hours.
- **Standalone honest (argmax)**: action 0.2716, point 0.1739, server AUC
  0.6500, overall **0.3082** — beats lgbm15 0.3027 (action +0.0129); now the
  strongest single base.

| config | action | point | server | overall | lift vs 0.32056 |
|---|---:|---:|---:|---:|---:|
| 5-base (prev production) | 0.28938 | 0.18416 | 0.65569 | 0.32056 | — |
| +cat (6-base) | 0.29388 | 0.18727 | 0.65667 | **0.32379** | **+0.00324** |

**VERDICT: SHIP.** Ensemble lift +0.00324 is ~1.9x the 0.00168 noise floor
(action +0.0045, point +0.0031, server +0.0010) — the first base to clear the
gate since the per-row fix, and a real contrast to seq2 (+0.00098, rejected).
`cat` is now in `build_final_perrow.py` BASES; **production honest overall =
0.3238**. `submission_FINAL_safe_perrow.csv` regenerated (6 bases).

**Public upload: NOT done (left to user).** Honest local lift > noise floor, so
uploading `submission_FINAL_safe_perrow.csv` to confirm is justified per the
rules — but uploads are daily-limited/teammate-shared, so the decision/timing is
the user's.

NEXT (private-score push): Prong B (TabPFN v2, point+server) and Prong C
(loss/threshold refinements), each gated the same way.

## TabPFN base result — Prong B REJECTED (2026-05-29)

Private-score push, prong B. Honest per-row ruler. Autonomous overnight run.

- **Env isolation**: TabPFN installed in a CLONED env `aicup-tt-tabpfn` (NOT main).
  First attempt `pip install tabpfn` pulled **tabpfn 8.0.3**, which force-upgraded
  torch 2.2→2.12 + CUDA 13 and broke the clone (`GLIBCXX_3.4.29 not found`). Fixed
  by pinning the open Apache v2 line: `tabpfn==2.2.1` + `tabpfn-common-utils==0.1.9`
  (with the `[interactive]` telemetry extra) via `--no-deps`, keeping torch 2.2.2.
  **Main `aicup-tt` env was never touched** (clone isolation did its job).
- **Limits confirmed empirically**: TabPFN v2 caps at **10 classes** (action's 19
  → excluded; point=10 and server=2 OK) and **~10k context samples** (per-fold
  ~12k → subsampled to 10k). Files: `scripts/produce_tabpfn_oof.py`,
  `scripts/predict_test_tabpfn.py` (run in the clone env; write `tabpfn_point`,
  `tabpfn_server` OOF + `_test`).
- **Standalone (argmax)**: point macro-F1 **0.0742** (vs lgbm15 0.1730, cat 0.1739),
  server AUC 0.6488. TabPFN applies no class weighting → majority-biased → poor
  macro-F1 on the imbalanced targets.

| config | point | server | overall | Δ vs cat-prod |
|---|---:|---:|---:|---:|
| cat-only production | 0.18727 | 0.65667 | 0.32379 | — |
| +tabpfn (point+server) | 0.18358 | 0.65675 | 0.32233 | **-0.00146** |

**VERDICT: REJECTED.** Adding TabPFN *hurts* (overall -0.00146; point -0.00369) —
its noisy/majority-biased point signal degrades the stack. Reverted; production
stays **cat-only 0.32379**. OOF/test parquets + scripts kept for reproducibility;
not in production BASES. The `aicup-tt-tabpfn` clone env can be removed
(`conda env remove -n aicup-tt-tabpfn`) if disk is needed.

## Prong C (CatBoost tuning) + private-push summary (2026-05-29)

Prong C = refinements. Since the two new-model-class bets (seq, TabPFN) failed and
CatBoost was the proven lever, the single pre-specified A/B was a **depth-8**
CatBoost (`catd8`) vs the shipped depth-6 `cat` (one honest comparison, not a grid
— no ruler-overfitting). Parametrized `depth`/`--model-name` in
`train_catboost_baseline.py` / `produce_catboost_oof.py` / `predict_test_catboost.py`
(defaults preserve the shipped `cat` behavior).

| base | action | point | server | overall |
|---|---:|---:|---:|---:|
| cat (depth 6, shipped) | 0.2716 | 0.1739 | 0.6500 | **0.3082** |
| catd8 (depth 8) | 0.2682 | 0.1748 | 0.6459 | 0.3064 |

**VERDICT: no gain.** Depth 8 overfits (worse standalone, -0.0018) and is highly
correlated with `cat`, so it would not add ensemble diversity. The depth-6 `cat`
config is already well-chosen. catd8 parquets discarded (reproducible via
`--depth 8`); no production change. No further tuning pursued (diminishing
returns + ruler-overfitting risk per the project philosophy).

### Private-score push — final state

Goal: lift the honest per-row ensemble above 0.3206. Result: **0.3238** (+0.0032).

| prong | model class | verdict | honest ensemble effect |
|---|---|---|---|
| (pre) seq2 | Transformer | REJECTED | +0.00098 (sub-noise) |
| A | CatBoost (depth 6, GPU) | **SHIPPED** | **+0.00324 (>0.00168 floor)** |
| B | TabPFN v2 (point+server) | REJECTED | -0.00146 (hurts) |
| C | CatBoost depth-8 tune | no gain | worse standalone |

Production = **6-base per-row ensemble incl. `cat`, honest overall 0.3238**
(action 0.2939, point 0.1873, server 0.6567). `submission_FINAL_safe_perrow.csv`
regenerated. **Public upload still pending the user** (honest lift > noise floor
justifies one confirmation upload, but uploads are daily-limited/teammate-shared).
The only above-noise lever found was CatBoost; the ensemble now looks saturated to
the new-model-class bets tried. The `aicup-tt-tabpfn` clone env can be removed.

## Private-push v2 — Phase 1 (rare-class macro-F1) REJECTED (2026-05-29)

Plan `2026-05-29-aicup-rareclass-macrof1.md`. Two rebalanced CatBoost variants vs
the shipped `cat` (sqrt weights). Honest per-row, GPU.

- **`cat_bal`** (full `balanced` class weights): standalone overall **0.299** < cat
  0.3082 (balanced over-weights ultra-rare classes, hurts argmax).
- **`cat_os`** (rare-class oversampling, `min_frac=0.3`, `scripts/resample.py`):
  standalone action **0.2733** (>cat 0.2716 — helps rare action classes) but point
  0.1642 (worse), overall 0.305.
- **GPU fix discovered**: CatBoost GPU default `gpu_ram_part=0.95` caused a hard
  hang at ~cell 19 (sequential fits colliding at 95% RAM). Fixed with
  `gpu_ram_part=0.45` on GPU fits — benefits all future GPU CatBoost runs.

Ensemble gate (OOF-only nested-CV, baseline 0.32379):

| config | overall | lift |
|---|---:|---:|
| production (cat) | 0.32379 | — |
| cat + cat_bal | 0.32300 | -0.00079 |
| cat + cat_os | 0.32381 | +0.00001 |
| cat_os replaces cat | 0.32351 | -0.00029 |
| cat + cat_os (action-only) | 0.32402 | +0.00022 |

**VERDICT: REJECTED.** Best config (+0.00022) is far below the 0.00168 floor. The
downstream prior-correction + nested threshold tuning already captures the
rare-class macro-F1 signal, so training-time rebalancing is redundant. Production
unchanged (cat-only, 0.32379); `build_final_perrow.py` untouched. cat_bal/cat_os
scripts + OOF/test parquets kept for reproducibility (not in BASES). Proceeding to
Phase 2 (XGBoost-GPU + FT-Transformer diverse bases).

## Private-push v2 — Phase 2 results (2026-05-29)

### 2-A: XGBoost (GPU) base — REJECTED

`scripts/produce_xgb_oof.py` + `predict_test_xgb.py` (numeric features, sqrt
sample-weights, GPU; LabelEncoder so XGBoost accepts folds with missing classes).
Standalone honest: action 0.2499, point 0.1705, server 0.6339, overall **0.2949**
— weaker than cat 0.3082 and lgbm15 0.3027 (a third, more-redundant GBDT). Gate
(OOF-only): cat+xgb overall **0.32408**, lift **+0.00028** ≪ 0.00168 floor.
**REJECTED.** Test inference skipped (won't ship). Scripts + OOF kept for
reproducibility; not in BASES.

### 2-B: FT-Transformer (GPU neural-tabular) — REJECTED at pilot

`scripts/ft_transformer.py` (numerical feature tokenizer → transformer → CLS →
multi-task heads, class-weighted CE; pure PyTorch, no new deps) +
`scripts/produce_ft_oof.py`. Pilot (seed 11 × folds 0-2) monitor (argmax)
plateaued at overall **~0.24** (best fold-0 0.2401; action ~0.13-0.20, point
~0.09-0.10) — far below the GBDT bases (~0.30) and no better than the already-
rejected seq Transformer (0.24, which added nothing to the stack). Neural models
on the engineered tabular features don't beat the GBDTs that already exploit
those features. **REJECTED at the pilot gate** — full 25-fold + test skipped to
avoid wasting compute (consistent with seq/TabPFN). ft_pilot parquets discarded;
`ft_transformer.py`/`produce_ft_oof.py` kept for reproducibility.

### Phase 3: hill-climbing stacker — REJECTED

Greedy weighted blend vs the LR meta, gated on a seed-55 holdout (fit/select on
seeds 11/22/33/44). Two-stage check:
- **Argmax-only** comparison first looked great: holdout overall LR 0.2893 vs
  hillclimb 0.3157 (+0.0264). **But that was an artifact** — LR-argmax is
  handicapped for macro-F1 (it favors majority classes); the production pipeline
  fixes this with prior-correction + threshold tuning.
- **Fair** comparison (prior + nested threshold tuning applied to BOTH, the actual
  production method): holdout overall **LR 0.32650 vs hillclimb 0.22383
  (-0.10267)**. The blend's probabilities collapse under threshold tuning (action
  F1 0.067). LR dominates.

**REJECTED.** The LR meta-stacker is the right combiner; hill-climbing is far worse
once the production scoring is applied. (Lesson: always compare stackers with the
full downstream pipeline, not argmax.) Production unchanged.

## Private-push v2 — FINAL SUMMARY (2026-05-29)

Goal: lift honest private overall above 0.3238 (CatBoost from the prior campaign).
Result: **no lever cleared the 0.00168 noise floor. Production stays 0.3238.**

| phase | lever | verdict | honest signal |
|---|---|---|---|
| 1 | rare-class rebalancing (cat_bal/cat_os) | REJECTED | best +0.00022 |
| 2-A | XGBoost (GPU) | REJECTED | +0.00028 |
| 2-B | FT-Transformer (GPU) | REJECTED | pilot ~0.24 ≪ GBDT |
| 3 | hill-climbing stacker | REJECTED | -0.103 on holdout (fair) |

Combined with the prior campaign (seq2, TabPFN, depth-8 all rejected; only
CatBoost shipped +0.00324), the honest conclusion is firm: **the ensemble is
saturated and the data is at its information ceiling** (next stroke genuinely
unobserved; short rallies). Every model-class / rebalancing / stacker idea tried
lands sub-noise. The only remaining big lever is the **public-smooth
serverGetPoint trick (deferred to the final submission)**. New scripts/variants
kept for reproducibility; production = 6-base per-row ensemble (incl. cat),
honest overall **0.3238**, `submission_FINAL_safe_perrow.csv`. Public upload still
pending the user.

## Player-conditional transition features (markovp) — SHIPPED (2026-05-29)

Idea 1 from the user/consult: explicit smoothed **P(next | last_stroke,
next_player)** as features — genuinely untried (the `markov` base is
player-agnostic; conditions on phase + last-stroke only). `scripts/produce_markovp_oof.py`:
OOF-safe hierarchical backoff (global → last-stroke → (player,last-stroke),
Dirichlet alpha=8), action+point only (player-conditional server unclear).

Gate (OOF nested-CV vs 0.32379): **+markovp → overall 0.32568, lift +0.00188**
(action 0.2939→0.2963, point 0.1873→0.1895 — both improve consistently).

**VERDICT: SHIP.** +0.00188 clears the 0.00168 floor (marginal, ~1.1x, vs
CatBoost's 1.9x — a borderline-but-real, mechanistically-sound gain on both
macro-F1 targets). The tree has player ID + last-action as raw features but
can't estimate the smoothed interaction for sparse player×prev-action combos;
the explicit backoff prob supplies it. Added to `build_final_perrow.py` BASES
(action+point; server unchanged). **Production honest overall now 0.32568.**
Second lever to ship after CatBoost; the only v2-era idea that cleared the gate.
(Still nowhere near 0.4 — that remains leak/public-only territory; this is an
honest private gain.)

## Leak-max (public) submission — outcome-as-feature (2026-05-29)

User directive: maximize the competition score by every permitted means (COI
caution overridden; see [[user-role-aicup]]). The one permitted leak is
serverGetPoint (known for 1236/1845 test rallies via old-test; organizer-allowed
as training data).

- **Measured leak-as-feature**: feed true serverGetPoint as a FEATURE to the
  action/point models (`--leak-sgp` in produce_catboost_oof / predict_test_catboost).
  OOF (full leak): point macro-F1 **0.1739 → 0.2142 (+0.0403)**, action +0.0008
  (the rally outcome predicts LANDING, not stroke type). cat_sgp overall 0.3082→0.3246.
- **Deployment** (`scripts/build_leakmax_submission.py`): keep the honest
  submission untouched; build a separate `submission_FINAL_leakmax.csv` =
  smooth submission with **point overridden by cat_sgp on the 1236 overlap
  rallies** (true serverGetPoint known there) + server smoothing. Changed point
  on 749 rallies; action/server unchanged.
- **Realistic gain**: leak covers ~67% of test (likely the public portion);
  point boost applies there → modest bump over the ~0.41 smooth baseline. Exact
  public number needs an upload. **Not alone enough to reach the #1 ~0.55** —
  action has no outcome-leak lever, so toppers likely also use LB-probing or
  other tricks. Within rules (programmatic, organizer-permitted data).
- **Honest/private submission unchanged**: `submission_FINAL_safe_perrow.csv`
  (cat+markovp, 0.32568) remains the clean private bet; leak only helps public.

## Feature pruning (data-centric) — REJECTED (2026-05-29)

User direction: prune useless / high-variance features to improve prediction.
`scripts/cat_feature_importance.py` (per-fold CatBoost importance mean+std on
seed11 x folds0-2) found **78 of 195 features have ~0 importance for both
action & point** (e.g. obs*_strikeId, handId_cnt_*, prefix_is_long); the
high-std features (next_gamePlayerId_inferred, first_gamePlayerId) are also the
most IMPORTANT, so kept. Built `cat_pruned` (top 117 features, via
`--keep-features artifacts/cat_keep_features.json`).

| | action | point | server | overall |
|---|---:|---:|---:|---:|
| cat (195 feats) standalone | 0.2716 | 0.1739 | 0.6500 | 0.3082 |
| cat_pruned (117) standalone | 0.2716 | 0.1733 | 0.6501 | 0.3080 |
| ensemble w/ cat | — | — | — | 0.32568 |
| ensemble w/ cat_pruned | — | — | — | 0.32537 |

**VERDICT: REJECTED.** Pruning is neutral-to-slightly-worse (-0.0003) — CatBoost
already ignores useless features, so removing them changes nothing. Confirms the
issue is the **information ceiling, not feature noise**; data-CLEANING can't help,
only NEW signal does (as markovp/CatBoost did). Production unchanged (cat full,
0.32568). `cat_feature_importance.py` + `--keep-features` kept as reusable tooling.

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
