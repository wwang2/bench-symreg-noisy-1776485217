# Symbolic Regression (Noisy)

## Problem Statement
Given 50 noisy data points (x, y) in `research/eval/train_data.csv` with x in [-5, 5],
find a function f(x) that best approximates the underlying generator. The training
data contains additive noise; the held-out test set (used by the evaluator) is clean.

## Solution Interface
Solution must be a Python file `orbits/<name>/solution.py` exposing either:
- `f(x)` — a callable taking a numpy array of x values and returning predicted y values, OR
- `solve(seed=42)` — returning a callable with the same signature

The evaluator loads the module and calls `module.f` (preferred) or `module.solve(seed)`.

## Success Metric
MSE on a held-out clean test set of 500 points across x in [-5, 5] (minimize).
Target: 0.01.

## Budget
Max 3 orbits.
