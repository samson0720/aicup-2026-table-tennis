#!/bin/bash
# Full Route-A-base regeneration + canonical final ensemble, run locally on macOS
# using the project .venv (lightgbm satisfied by sklearn's bundled libomp via
# DYLD_FALLBACK_LIBRARY_PATH). Produces the canonical A+B submission with the new
# per-target prior-temperature (beta) selection wired in.
set -euo pipefail
cd "$(dirname "$0")/.."

export DYLD_FALLBACK_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/sklearn/.dylibs"
PY=".venv/bin/python"

echo "=== START $(date) ==="
for m in lgbm15 lgbm31 markov phase_lgbm; do
  echo ">>> OOF $m $(date)"
  $PY -u -m scripts.produce_base_oof --model "$m"
done
for m in lgbm15 lgbm31 markov phase_lgbm; do
  echo ">>> TEST $m $(date)"
  $PY -u -m scripts.predict_test_base --model "$m"
done
echo ">>> FINAL build_final_perrow $(date)"
$PY -u -m scripts.build_final_perrow
echo "=== DONE $(date) ==="
