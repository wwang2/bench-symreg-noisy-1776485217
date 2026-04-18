#!/usr/bin/env bash
# Reproduce orbit 01-gp-regression: fit the GP, evaluate for seeds 1..3,
# regenerate figures. Run from the repo root.

set -euo pipefail

ORBIT_DIR="orbits/01-gp-regression"

# 1. Sanity check: solution.py loads and prints a fitted kernel.
python3 "${ORBIT_DIR}/solution.py"

# 2. Evaluate on 3 seeds (evaluator is deterministic — test grid is fixed).
for SEED in 1 2 3; do
  python3 research/eval/evaluator.py \
      --solution "${ORBIT_DIR}/solution.py" \
      --seed "${SEED}"
done

# 3. Regenerate all figures.
python3 "${ORBIT_DIR}/make_figures.py"
