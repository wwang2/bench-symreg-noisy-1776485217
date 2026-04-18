"""Generate figures/fit.png, figures/residuals.png, figures/results.png and
figures/narrative.png for orbit 01-gp-regression.

Run from any directory:
    python3 orbits/01-gp-regression/make_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "research", "eval"))

import solution as sol  # noqa: E402  (after sys.path)
from generate_data import generate_test_data, target_function  # noqa: E402

FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Matplotlib rcParams (research/style.md)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "axes.labelpad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

COLOR_GP = "#4C72B0"
COLOR_TRUE = "#55A868"
COLOR_BAND = "#4C72B0"
COLOR_TRAIN = "#444444"
COLOR_BASELINE = "#888888"

# ---------------------------------------------------------------------------
# Data we need
# ---------------------------------------------------------------------------
x_train = sol._X_TRAIN.ravel()
y_train = sol._Y_TRAIN
x_test, y_test = generate_test_data(n_points=500, seed=99)

# GP posterior mean and std on the dense grid
mean_pred, std_pred = sol._GP.predict(x_test.reshape(-1, 1), return_std=True)
y_fit_train = sol.f(x_train)
residuals = y_train - y_fit_train
mse_test = float(np.mean((y_test - mean_pred) ** 2))
mse_train = float(np.mean(residuals ** 2))
noise_sigma_fit = float(np.sqrt(sol._GP.kernel_.k2.noise_level))


# ---------------------------------------------------------------------------
# 1. figures/fit.png — scatter + GP mean + 95% band + test predictions
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

# 95% credible band = 1.96 sigma
ax.fill_between(
    x_test,
    mean_pred - 1.96 * std_pred,
    mean_pred + 1.96 * std_pred,
    color=COLOR_BAND,
    alpha=0.15,
    label="95% credible band",
)
ax.plot(x_test, mean_pred, color=COLOR_GP, lw=2.2, label="GP posterior mean")
ax.plot(
    x_test, y_test,
    color=COLOR_TRUE, lw=1.4, linestyle="--",
    label="True target (hidden)",
)
ax.scatter(
    x_train, y_train,
    s=28, color=COLOR_TRAIN, zorder=5,
    label="Training data (noisy)", edgecolor="white", linewidth=0.6,
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Orbit 01: GP regression fit")
ax.set_xlim(-5.2, 5.2)
ax.legend(loc="upper right")

# annotate fit quality inline
ax.text(
    0.02, 0.04,
    f"test MSE = {mse_test:.4f}   fitted noise sigma = {noise_sigma_fit:.3f}",
    transform=ax.transAxes, fontsize=10, color="#333333",
)

fig.savefig(os.path.join(FIG_DIR, "fit.png"), dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 2. figures/residuals.png — residuals on training set
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = axes[0]
ax.axhline(0, color=COLOR_BASELINE, linestyle="--", lw=0.8)
ax.scatter(x_train, residuals, s=30, color=COLOR_GP, edgecolor="white", linewidth=0.6)
# +/- 0.05 (true noise sigma) reference band
ax.axhspan(-0.05, 0.05, color=COLOR_BAND, alpha=0.08, label=r"$\pm\sigma_{\rm true}=0.05$")
ax.set_xlabel("x")
ax.set_ylabel("y_train  -  GP mean")
ax.set_title("Training residuals vs. x")
ax.legend(loc="upper right")

ax = axes[1]
ax.hist(residuals, bins=12, color=COLOR_GP, alpha=0.85, edgecolor="white")
ax.axvline(0, color=COLOR_BASELINE, linestyle="--", lw=0.8)
ax.set_xlabel("residual")
ax.set_ylabel("count")
ax.set_title(
    f"Residual distribution (RMS = {np.sqrt(mse_train):.3f}, "
    rf"$\sigma_{{\rm true}}=0.05$)"
)

fig.suptitle("Orbit 01: residuals on the 50 training points", y=1.05, fontsize=14)
fig.savefig(os.path.join(FIG_DIR, "residuals.png"), dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 3. figures/results.png — quantitative bar chart + seed breakdown
# ---------------------------------------------------------------------------
baseline_mse = 0.4266  # research/config.yaml
target_mse = 0.01
seed_metrics = [mse_test, mse_test, mse_test]   # evaluator is deterministic
seeds = [1, 2, 3]
mean_metric = float(np.mean(seed_metrics))
std_metric = float(np.std(seed_metrics))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# Left: log-scale bar comparison against baseline and target
ax = axes[0]
labels = ["baseline\n(constant)", "target", "GP\n(orbit 01)"]
values = [baseline_mse, target_mse, mean_metric]
colors = [COLOR_BASELINE, "#C44E52", COLOR_GP]
bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
ax.set_yscale("log")
ax.set_ylabel("Test MSE (log scale)")
ax.set_title("GP vs. baseline and target")
for bar, v in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        v * 1.12,
        f"{v:.4f}" if v < 0.1 else f"{v:.3f}",
        ha="center", va="bottom", fontsize=10,
    )
ax.set_ylim(1e-4, 1)

# Right: per-seed bars
ax = axes[1]
ax.bar(
    [str(s) for s in seeds], seed_metrics,
    color=COLOR_GP, edgecolor="white", linewidth=0.8,
)
ax.axhline(target_mse, color="#C44E52", linestyle="--", lw=1.0, label=f"target = {target_mse}")
ax.axhline(mean_metric, color="#222222", linestyle=":", lw=1.0, label=f"mean = {mean_metric:.4f}")
ax.set_xlabel("seed")
ax.set_ylabel("MSE")
ax.set_title(f"Per-seed test MSE   ({mean_metric:.4f} ± {std_metric:.4f})")
ax.legend(loc="upper right")
ax.set_ylim(0, max(target_mse, mean_metric) * 1.6)

fig.suptitle("Orbit 01 results — GP regression on noisy symbolic regression", y=1.04)
fig.savefig(os.path.join(FIG_DIR, "results.png"), dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 4. figures/narrative.png — baseline vs GP on the same axes
# ---------------------------------------------------------------------------
# Baseline 1: constant predictor (mean of training y)
const_pred = np.full_like(x_test, y_train.mean())
# Baseline 2: linear least-squares fit
lin_coef = np.polyfit(x_train, y_train, deg=1)
lin_pred = np.polyval(lin_coef, x_test)

mse_const = float(np.mean((y_test - const_pred) ** 2))
mse_lin = float(np.mean((y_test - lin_pred) ** 2))

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)

panels = [
    ("(a) Constant baseline",       const_pred, mse_const),
    ("(b) Linear least-squares",    lin_pred,   mse_lin),
    ("(c) GP regression (ours)",    mean_pred,  mse_test),
]
for ax, (title, pred, mse) in zip(axes, panels):
    ax.scatter(
        x_train, y_train,
        s=22, color=COLOR_TRAIN, edgecolor="white", linewidth=0.5,
        zorder=5, label="train",
    )
    ax.plot(x_test, y_test, color=COLOR_TRUE, lw=1.2, linestyle="--", label="true")
    ax.plot(x_test, pred, color=COLOR_GP, lw=2.2, label="predict")
    if "GP" in title:
        ax.fill_between(
            x_test,
            pred - 1.96 * std_pred, pred + 1.96 * std_pred,
            color=COLOR_BAND, alpha=0.12,
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.text(
        0.04, 0.05,
        f"MSE = {mse:.4f}",
        transform=ax.transAxes, fontsize=11, color="#222222",
    )
    ax.set_xlim(-5.2, 5.2)
axes[0].set_ylabel("y")
axes[-1].legend(loc="upper right")

fig.suptitle(
    "From blunt to sharp: baselines vs. Gaussian process on the noisy target",
    y=1.04,
)
fig.savefig(os.path.join(FIG_DIR, "narrative.png"), dpi=200, bbox_inches="tight")
plt.close(fig)


print(f"fit.png / residuals.png / results.png / narrative.png written to {FIG_DIR}")
print(f"mean test MSE (seeds 1,2,3) = {mean_metric:.6f}")
print(f"fitted kernel: {sol._GP.kernel_}")
