"""Orbit 01: Gaussian Process regression for noisy symbolic regression.

Fits a GP with an RBF kernel plus a WhiteKernel (for the ~sigma=0.05 training
noise) on the 50 observed points once at import time. Exposes ``f(x)`` that
returns the posterior mean, which acts as a smooth denoised interpolant of the
hidden target.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    WhiteKernel,
)


# ---------------------------------------------------------------------------
# Load training data (fixed path — the frozen evaluator lives next to it)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_TRAIN_CSV = os.path.join(_REPO_ROOT, "research", "eval", "train_data.csv")

_data = np.loadtxt(_TRAIN_CSV, delimiter=",", skiprows=1)
_X_TRAIN = _data[:, 0].reshape(-1, 1)
_Y_TRAIN = _data[:, 1]


# ---------------------------------------------------------------------------
# Kernel: constant * RBF + white noise.
# - ConstantKernel: the signal variance.
# - RBF: smooth interpolation; length_scale is optimised.
# - WhiteKernel: explicitly models the observation noise (sigma ~ 0.05).
# We also fit a Matern-3/2 alternative and keep the one with the higher
# log-marginal-likelihood. Matern is a touch less smooth than RBF, which can
# help capture sharper features of the target.
# ---------------------------------------------------------------------------
def _fit_gp() -> GaussianProcessRegressor:
    kernels = [
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2))
        + WhiteKernel(noise_level=0.05**2, noise_level_bounds=(1e-6, 1e-1)),
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 1e2), nu=2.5)
        + WhiteKernel(noise_level=0.05**2, noise_level_bounds=(1e-6, 1e-1)),
    ]

    best_gp = None
    best_lml = -np.inf
    for kernel in kernels:
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=10,
            alpha=0.0,           # noise is handled by the WhiteKernel
            random_state=0,
        )
        gp.fit(_X_TRAIN, _Y_TRAIN)
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        if lml > best_lml:
            best_lml = lml
            best_gp = gp
    return best_gp


_GP = _fit_gp()


def f(x: np.ndarray) -> np.ndarray:
    """Posterior mean of the GP evaluated at ``x``.

    Accepts either a 1D array of query points or a 2D column vector and
    returns a 1D array of predictions.
    """
    x = np.asarray(x, dtype=float)
    flat = x.ravel().reshape(-1, 1)
    mean = _GP.predict(flat)
    return mean.reshape(x.shape) if x.ndim > 1 else mean


if __name__ == "__main__":
    # Quick sanity check when run directly.
    x = np.linspace(-5, 5, 11)
    print("f(x) sample:", f(x))
    print("fitted kernel:", _GP.kernel_)
    print("log-marginal-likelihood:", _GP.log_marginal_likelihood(_GP.kernel_.theta))
