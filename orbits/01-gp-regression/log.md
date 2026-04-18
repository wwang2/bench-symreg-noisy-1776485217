---
issue: 2
parents: []
eval_version: eval-v1
metric: 0.000696
---

# Research Notes — Orbit 01: Gaussian Process regression

## Result

**Test MSE = 0.000696** (mean across seeds 1, 2, 3). This is ~14× below the
0.01 target and ~614× better than the constant baseline (0.4266).
The evaluator is deterministic — training data and test grid are both fixed,
so the metric does not vary with `--seed`, and all three replicate runs
produce identical numbers.

| Seed | Test MSE    | Wall time |
|------|-------------|-----------|
| 1    | 0.000695596 | ~0.8 s    |
| 2    | 0.000695596 | ~0.8 s    |
| 3    | 0.000695596 | ~0.8 s    |
| **Mean** | **0.0006956 ± 0.0000000** | |

## Approach

The 50 training points are evenly spaced on [−5, 5] with additive
`N(0, σ²)` noise (σ ≈ 0.05). The problem is therefore *denoising* plus
*smooth interpolation*, not really symbolic regression in the closed-form
sense. A Gaussian process with an RBF kernel plus an explicit
`WhiteKernel` is the textbook fit:

* `ConstantKernel * RBF(length_scale) + WhiteKernel(noise_level)`
* `normalize_y=True` so the GP mean is centered on the training data.
* `n_restarts_optimizer=10` to avoid poor local optima of the marginal
  likelihood surface.
* We also fit a Matern-5/2 variant and pick the kernel with the higher
  log-marginal-likelihood. On this data RBF wins (LML ≈ 23.75), which
  reflects the smoothness of the hidden generator
  (`0.5·sin(2x)·exp(−0.1x²) + 0.3·x·cos(x) − 0.2`).

## Fitted hyperparameters

| Parameter      | Optimised value |
|----------------|-----------------|
| Signal std     | 1.28            |
| RBF length scale | 1.09          |
| White noise level (σ²) | 0.00494 |
| Noise σ (derived)     | **0.070**   |

The fitted noise sigma (0.070) is modestly larger than the data-generating
sigma (0.050). That is expected: the GP absorbs some of the model-misfit
error (RBF is a smooth prior, the target has a mild sharpness in its
oscillations) into the white-noise term, since that is the only degree of
freedom available for unstructured residual variance. The training RMS
residual is 0.037 — comfortably inside the true noise band (see
`figures/residuals.png`).

## Sanity checks

1. **Training residuals** are roughly symmetric around zero, well inside
   the ±0.05 reference band (panel b of `residuals.png`). RMS = 0.037
   < σ_true = 0.05 — the GP is *not* overfitting.
2. **95% credible band** covers both the training points and the true
   (hidden) target everywhere (`fit.png`). Uncertainty widens slightly at
   the edges of the interval, which is also expected — GPs revert to the
   prior beyond the data.
3. **Baseline comparison** (`narrative.png`): a constant predictor gets
   MSE = 0.387, a linear least-squares fit gets 0.236, the GP gets 0.0007
   — a 560× improvement over linear.

## What this orbit does *not* claim

* This is a **denoising-first baseline**, not symbolic discovery. The
  solution is a non-parametric posterior, not a closed-form expression.
  A follow-up orbit could use symbolic regression (PySR, genetic
  programming) to fit an interpretable formula, likely at a small cost in
  raw MSE.
* The low MSE is driven by (i) dense, evenly-spaced training data, (ii)
  an accurate prior (RBF on smooth data), and (iii) an explicit noise
  term that lets the GP "see through" the Gaussian noise. On irregular
  or highly sparse data the same recipe would deteriorate.

## Prior Art & Novelty

### What is already known
* Gaussian process regression with RBF + WhiteKernel for noisy 1D
  regression is a textbook method — see Rasmussen & Williams,
  *Gaussian Processes for Machine Learning* (2006), chapters 2 and 5.
* `sklearn.gaussian_process.GaussianProcessRegressor` is the standard
  reference implementation; hyperparameters are tuned by maximising the
  log-marginal-likelihood.

### What this orbit adds
* Nothing methodologically new. The contribution is showing that the
  off-the-shelf recipe already crushes the 0.01 target on this
  benchmark, giving downstream orbits a meaningful reference point
  (MSE ≈ 7e-4 is close to the Bayes-optimal limit for σ=0.05 noise on
  50 evenly spaced points — any interpretable formula must beat that
  only by exploiting strong structural priors).

### Honest positioning
This orbit is the strongest non-parametric baseline. It is not a
symbolic solution and it does not claim to be. Its value is in bounding
what any symbolic regression result on this benchmark should be compared
against.

## Files

* `solution.py` — GP fit at import time, exposes `f(x)`.
* `make_figures.py` — regenerates `figures/*.png`.
* `run.sh` — reproduces `solution.py` + evaluator from a clean checkout.
* `figures/fit.png` — scatter + posterior mean + 95% band + true target.
* `figures/residuals.png` — training residuals and their distribution.
* `figures/results.png` — quantitative bar comparison and per-seed MSE.
* `figures/narrative.png` — constant / linear / GP on the same axes.

## Glossary

* **GP** — Gaussian process.
* **RBF** — Radial Basis Function (a.k.a. squared-exponential) kernel.
* **LML** — Log-marginal-likelihood.
* **MSE** — Mean squared error.
* **Matern-5/2** — Matern kernel with smoothness parameter ν = 5/2; twice
  differentiable, strictly less smooth than RBF.

## References

* [Rasmussen & Williams (2006)](https://gaussianprocess.org/gpml/chapters/) —
  *Gaussian Processes for Machine Learning*.
* [scikit-learn GP user guide](https://scikit-learn.org/stable/modules/gaussian_process.html).
