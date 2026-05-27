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
| P1 cv-foundation | 1–10 | **in progress** (executing now) |
| P2 route-a-postprocess-stack | 1–12 | pending — blocked by P1 |
| P3 route-b-features-chain | 1–10 | pending — blocked by P1 |
| P4 route-c-transformer | 1–6 | pending — blocked by P1, needs 3090 |
| P5 final-ensemble | 1–7 | pending — blocked by P2 + P3 + P4 |

## Known issues to fix when we hit them

Flagged during plan review on 2026-05-27, NOT YET FIXED:

1. **P3 Route B Task 10 in-bag leakage (REAL BUG)** — `scripts/build_route_b_submission.py` as drafted trains stage-A action on full train, then uses **in-bag** predictions as `act_p_<i>` features for training stage-B point. LGBM is overconfident in-bag → stage-B learns to over-trust `act_p_*` columns → at test time real action probs are less confident → severe distribution shift.
   - **Fix when implementing P3 T10**: feed OOF action probabilities (`artifacts/oof/chain_action_action.parquet`) into the stage-B training feature set instead of in-bag preds. Same for stage-C consuming both.

2. **P2 Route A Task 11 incomplete** — only LGBM `predict_test_lgbm` is sketched. `markov`, `phase_lgbm`, `player_stats` test-time variants are described in prose only.
   - **Fix when implementing P2 T11**: each existing `train_*.py` already has internal logic that produces a `submission_*.csv` artifact today; extract that test-prediction block into a `predict_test_*` helper using the same OOF schema (`artifacts/oof/<model>_<target>_test.parquet`).

3. **P2 Task 10 stacker label collapse** — `attached.drop_duplicates("rally_uid")` keeps only one seed's cut-target label per rally; the seed-averaged base features see all 5 seeds. This is a feature/label mismatch but probably benign because LR with L2 is robust to label noise.
   - **Decide when running P2 T9–10**: if stacking lift is < 0.01, revisit by training the meta-learner on the un-averaged (rally, seed) rows so labels match features per row.

4. **P1 Task 5.3 phase-stratified rng state consumption** — the histogram-building `rng.integers` call inside `assign(_cut=...)` consumes rng state before the actual cut sampling. May cause cuts to differ slightly from a Task-4-only path. Should not affect invariants — tests will catch any drift.

## Logged decisions

- Use `git add -f artifacts/cv_splits.parquet` when needed: the existing `.gitignore` excludes `artifacts/*.parquet`, but the plan asks us to commit this specific small file (~1–2 MB).
- Run the diagnostic in P1 Task 9 in the background (10–20 min CPU work, 25 LGBM fits) and only score it once it finishes.
- Skip P5 Task 6 Optuna HPO until P2–P4 land (3–6 h CPU; not worth burning until base models are stable).

## Resuming work

Next concrete steps the next agent should take, in order:
1. Check `git log` to see how many P1 tasks were committed.
2. Run `conda env list | grep aicup-tt` to see if env exists.
3. Read the plan task that matches the lowest unchecked `- [ ]` checkbox in `docs/superpowers/plans/2026-05-27-cv-foundation.md`.
4. Resume from there.

## Notes on conda usage

- All plan commands run inside the env: `conda run -n aicup-tt <command>` (no shell activation needed).
- Do NOT install with `pip install --user`. Do NOT touch `base`.
- For ad-hoc Python sanity checks BEFORE `aicup-tt` exists, the `jitkg` env on this machine has `pandas`. Don't conflate it with project work.
