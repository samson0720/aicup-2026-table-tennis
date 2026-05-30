# AI Cup 2026 — Table Tennis "Next Stroke" Prediction

Predict, for each rally given its first few strokes (the prefix), the **next
stroke's** three labels:

| Target | Classes | Metric |
|---|---|---|
| `actionId` (stroke type) | 19 | macro-F1 |
| `pointId` (landing zone) | 10 | macro-F1 |
| `serverGetPoint` (does the server win the point) | 2 (binary) | AUC |

**Overall score** = `0.4 × action_F1 + 0.4 × point_F1 + 0.2 × server_AUC`

Data: `AI CUP競賽資料集/` (train: 14,995 rallies / 84,707 strokes; plus
test_new and sample_submission). Average rally length is 5.65 strokes
(median 5, p90 10) — sequences are short and the dataset is small/medium, so
**gradient-boosted trees are the workhorse, not deep sequence models**.

---

## Current best result

| Submission | Public LB | Notes |
|---|---|---|
| `artifacts/submission_FINAL_smooth_perrow.csv` | **0.4247437** | New A+B ensemble + per-target β + old-test smoothing (current best) |
| `artifacts/submission_FINAL_safe_perrow.csv` | (clean, private bet) | Same model, no smoothing trick |
| old `submission_RECOMMENDED_upload.csv` | 0.4102132 | Previous best (also smoothed) |

Honest local-CV ensemble overall = **0.3231** (action 0.2921 / point 0.1880 /
server 0.6557), **+0.0204** over the best single base lgbm15 (0.3027).

> **Methodology core**: local CV is the primary signal (self-built to mimic the
> private cut-point distribution); public LB is only a tie-breaker. The
> smoothing trick (1236 rallies overlapping the old test → server 0.95/0.05)
> only helps public, not private. See [HANDOFF.md](HANDOFF.md) and
> [PROGRESS.md](PROGRESS.md).

---

## Approach overview

1. **Self-built CV** (`scripts/cv_splits.py`): each rally is cut at a random
   stroke to mimic the test set, match-aware grouping, phase-stratified,
   5 seeds × 5 folds. Noise floor = **0.00168** (across-seed std); any lift
   below this is treated as noise.
2. **Route A** (post-process + stacking, `build_route_a_submission.py`): four
   bases — `lgbm15`, `lgbm31`, `markov`, `phase_lgbm` — produce OOF
   probabilities, then prior-correction + threshold tuning, then an LR
   meta-learner stacks them.
3. **Route B** (features + chain, `train_chain_lgbm.py`): target encoding and
   n-grams; an action→point→server chain where each downstream stage consumes
   the upstream stage's OOF probabilities (avoids in-bag leakage).
4. **Route C** (Transformer, `train_seq_transformer.py`): **rejected** — the
   data size / sequence length are unfavorable for deep models, its OOF is far
   below the tree models.
5. **Final ensemble** (**canonical = `build_final_perrow.py`**): per-row
   stacking of A+B (avoids the seed-averaging inflation), honest
   nested-threshold scoring, emits both a safe and a smooth submission.

### Per-target prior-temperature (β) — latest change
`postprocess.prior_correct(probs, prior, beta=1.0)` gained a `beta` argument,
and `select_beta()` picks it with honest CV. actionId prefers β≈0.6 and pointId
β≈0.4 (previously both hardcoded to β=1). Clean same-base lift on the canonical
ensemble = **+0.0028** (~1.7σ, no downside).

---

## Reproducing a submission

### A. Original environment (for the formal submission)
Linux + conda `aicup-tt` (GPU 3090). Prefix every command with
`conda run -n aicup-tt`.

```bash
for m in lgbm15 lgbm31 markov phase_lgbm; do
  conda run -n aicup-tt python -m scripts.produce_base_oof  --model $m
  conda run -n aicup-tt python -m scripts.predict_test_base --model $m
done
conda run -n aicup-tt python -m scripts.build_final_perrow
```

### B. Local macOS (for fast experiments; no conda / no Homebrew)
Use the project `.venv` and satisfy lightgbm's libomp from scikit-learn's
bundled copy (~36 min, all CPU):

```bash
uv venv .venv --python 3.11
uv pip install --python .venv lightgbm pandas scikit-learn pyarrow
bash scripts/run_full_b_local.sh   # chains OOF + test + final; DYLD libomp preset
```

For post-processing only (no retraining), rebuild directly from the
materialized parquets — no lightgbm needed at all:

```bash
uv run --with pandas --with scikit-learn --with pyarrow \
  python -m scripts.build_route_b_submission_from_parquet
```

---

## Layout

| Path | Contents |
|---|---|
| `AI CUP競賽資料集/` | Raw data (train / test_new / sample_submission) |
| `scripts/` | All pipeline and diagnostic scripts |
| `artifacts/` | OOF / test probability parquets, submissions, score JSONs (parquets are gitignored) |
| `tests/` | pytest (CV invariants, postprocess, stacker, scoring, etc.) |
| `docs/superpowers/specs/` | Design spec |
| `PROGRESS.md` | **Per-session source of truth — read this first** |
| `HANDOFF.md` / `HANDOVER_TO_CODEX.md` | Competition progress and handoff notes |

Tests: `pytest -q` (locally:
`uv run --with pandas --with scikit-learn --with pyarrow --with pytest python -m pytest`).

---

## Next optimization directions (by ROI)

1. **Add an XGBoost base** — in the tree-model sweet spot, a different mechanism
   from LGBM, gives the stacker real diversity; runs locally on CPU. Most
   practical.
2. **Stacking refinement / HPO** — per-target meta-learner, probability
   calibration, Optuna over the bases.
3. ~~pointId feature engineering~~ — TRIED 2026-05-31, REJECTED. Joint
   point/position bigram features had real mutual information but the existing
   `point_cnt_*`/`last1-5_pointId` columns already reconstruct them; lift was
   below the noise floor. Kept on branch `feat/pointid-feature-engineering` for
   the record. See PROGRESS.md.
4. ~~Transformer / sequence models~~ — data size and sequence length are
   unfavorable; lowest ROI, do last or skip.
