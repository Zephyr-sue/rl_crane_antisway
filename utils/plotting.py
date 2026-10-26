"""Consistent matplotlib styling and figures for the report, README and dashboard."""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots in fixed order (validated reference palette); colour follows the controller
COLORS = {
    "GRPO": "#2a78d6",
    "RBF": "#eb6834",
    "PPO": "#1baf7a",
}
# GRPO variants in the learning-curve figure (slot 1 = main run; RBF and PPO keep slots 2 and 3)
VARIANT_COLORS = {"rtg (main)": "#2a78d6", "outcome": "#eda100", "process": "#e87ba4", "no params": "#6250d6"}
VARIANT_RUNS = {"rtg (main)": "grpo_seed*", "outcome": "grpo_outcome_seed*", "process": "grpo_process_seed*",
                "no params": "grpo_noparams_seed*"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ORDER = ["GRPO", "PPO", "RBF"]


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "legend.frameon": False,
    })


def read_progress(path: str) -> pd.DataFrame:
    """Validation rows of a run's progress.csv (empty for a run that has not logged yet)."""
    try:
        return pd.read_csv(path).dropna(subset=["val_return"])
    except (pd.errors.EmptyDataError, KeyError):
        return pd.DataFrame()


def _mean_curve(files, xcol, metric, scale, log=False):
    curves = [df for f in files if len(df := read_progress(f)) > 1]
    if not curves:
        return None
    hi = min(c[xcol].max() for c in curves)
    lo = max(5e4, max(c[xcol][c[xcol] > 0].min() for c in curves))
    grid = np.geomspace(lo, hi, 100) if log else np.linspace(0, hi, 100)
    ys = np.stack([np.interp(grid, c[xcol], c[metric]) for c in curves])
    return grid / scale, ys.mean(0), ys.std(0), len(curves)


def plot_learning_curves(runs_dir: str, out_path: str):
    """Left: GRPO vs PPO validation success vs environment steps (log scale, mean +/- std over seeds).
    Right: GRPO advantage variants and the no-parameter ablation vs environment steps."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.9))
    for label, pattern in [("GRPO", "grpo_seed*"), ("PPO", "ppo_seed*")]:
        res = _mean_curve(sorted(glob.glob(os.path.join(runs_dir, pattern, "progress.csv"))), "env_steps", "val_success", 1e6, log=True)
        if res:
            x, m, sd, n = res
            axes[0].plot(x, m, color=COLORS[label], label=f"{label} ({n} seeds)" if n > 1 else label)
            axes[0].fill_between(x, m - sd, m + sd, color=COLORS[label], alpha=0.2, linewidth=0)
    for label, pattern in VARIANT_RUNS.items():
        res = _mean_curve(sorted(glob.glob(os.path.join(runs_dir, pattern, "progress.csv"))), "env_steps", "val_success", 1e6)
        if res:
            x, m, sd, n = res
            axes[1].plot(x, m, color=VARIANT_COLORS[label], label=label)
            if n > 1:
                axes[1].fill_between(x, m - sd, m + sd, color=VARIANT_COLORS[label], alpha=0.2, linewidth=0)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Env. steps (M, log)", fontsize=7)
    axes[1].set_xlabel("GRPO env. steps (M)", fontsize=7)
    for ax in axes:
        ax.set_ylabel("Validation success", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0.5, 1.0)
        ax.legend(loc="lower right", fontsize=5.5)
    axes[0].legend(loc="upper left", fontsize=5.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_trajectories(trajs: dict, out_path: str, title: str = ""):
    """trajs: {controller_name: rollout dict}. Panels: trolley x, hook sway, payload sway, force."""
    setup_style()
    fig, axes = plt.subplots(4, 1, figsize=(3.5, 4.6), sharex=True)
    any_traj = next(iter(trajs.values()))
    for name, tr in trajs.items():
        kw = dict(color=COLORS.get(name, INK_2), label=name, linewidth=1.2)
        axes[0].plot(tr["t"], tr["x"], **kw)
        axes[1].plot(tr["t"], np.rad2deg(tr["th1"]), **kw)
        axes[2].plot(tr["t"], np.rad2deg(tr["th2"]), **kw)
        axes[3].plot(tr["t"], tr["u"] * 30.0, **kw)
    axes[0].axhline(any_traj["target_x"], color=INK_2, linestyle="--", linewidth=0.8)
    gust = any_traj["gust"]
    for ax in axes:
        for t0, t1 in _intervals(any_traj["t"], gust != 0):
            ax.axvspan(t0, t1, color=GRID, alpha=0.8, linewidth=0)
    axes[0].set_ylabel("Trolley x (m)")
    axes[1].set_ylabel(r"Hook $\theta_1$ (°)")
    axes[2].set_ylabel(r"Payload $\theta_2$ (°)")
    axes[3].set_ylabel("Force F (N)")
    axes[3].set_xlabel("Time (s)  (shaded: wind gust)")
    axes[0].legend(ncol=3, loc="lower right")
    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _intervals(t, mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = t[i]
        if not m and start is not None:
            out.append((start, t[i]))
            start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


def plot_heatmaps(grids: dict, m2_vals, l_vals, out_path: str, train_box=None):
    """grids: {controller: 2D success-rate array (len(m2_vals), len(l_vals))}. Single-hue sequential map."""
    setup_style()
    names = list(grids)
    fig, axes = plt.subplots(1, len(names), figsize=(3.5 * len(names) / 2 + 0.6, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        im = ax.imshow(grids[name], origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto",
                       extent=[l_vals[0], l_vals[-1], m2_vals[0], m2_vals[-1]])
        if train_box is not None:
            (l0, l1), (m0, m1) = train_box
            ax.add_patch(plt.Rectangle((l0, m0), l1 - l0, m1 - m0, fill=False, edgecolor=INK, linestyle="--", linewidth=1.0))
        ax.set_title(name)
        ax.set_xlabel(r"$L_1+L_2$ (m)")
        ax.grid(False)
    axes[0].set_ylabel(r"Payload mass $m_2$ (kg)")
    fig.colorbar(im, ax=axes, shrink=0.9, label="Success rate")
    fig.savefig(out_path)
    plt.close(fig)


def plot_condition_bars(summary: pd.DataFrame, metric: str, ylabel: str, out_path: str):
    """Grouped bars: conditions on x, one bar per controller (fixed colour per controller)."""
    setup_style()
    conds = list(dict.fromkeys(summary["condition"]))
    ctrls = [c for c in ORDER if c in set(summary["controller"])]
    width = 0.8 / len(ctrls)
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    for j, ctrl in enumerate(ctrls):
        vals = [summary[(summary.condition == c) & (summary.controller == ctrl)][metric].mean() for c in conds]
        xs = np.arange(len(conds)) + (j - (len(ctrls) - 1) / 2) * width
        ax.bar(xs, vals, width=width * 0.9, color=COLORS[ctrl], label=ctrl)
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels(conds)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=len(ctrls), loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
