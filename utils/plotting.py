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
    "PPO": "#eb6834",
    "SAC": "#1baf7a",
    "LQR": "#eda100",
    "SMC": "#e87ba4",
    "Zero": "#8a8984",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ORDER = ["GRPO", "PPO", "SAC", "LQR", "SMC", "Zero"]


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "legend.frameon": False,
    })


def plot_learning_curves(runs_dir: str, out_path: str, x: str = "env_steps"):
    """Validation return vs env steps (or wall time), mean +/- std over seeds, one panel per algo."""
    setup_style()
    algos = [("grpo", "GRPO"), ("ppo", "PPO"), ("sac", "SAC")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for algo, label in algos:
        files = sorted(glob.glob(os.path.join(runs_dir, f"{algo}_seed*", "progress.csv")))
        curves = []
        for f in files:
            df = pd.read_csv(f).dropna(subset=["val_return"])
            if len(df):
                curves.append(df)
        if not curves:
            continue
        for ax, xcol, metric in [(axes[0], x, "val_return"), (axes[1], x, "val_success")]:
            grid = np.linspace(0, min(c[xcol].max() for c in curves), 100)
            ys = np.stack([np.interp(grid, c[xcol], c[metric]) for c in curves])
            scale = 1e6 if xcol == "env_steps" else 60.0
            m, s = ys.mean(0), ys.std(0)
            ax.plot(grid / scale, m, color=COLORS[label], label=f"{label} ({len(curves)} seed{'s' if len(curves) > 1 else ''})")
            if len(curves) > 1:
                ax.fill_between(grid / scale, m - s, m + s, color=COLORS[label], alpha=0.2, linewidth=0)
    xlabel = "Environment steps (millions, log scale)" if x == "env_steps" else "Wall-clock time (min)"
    for ax, ylabel in [(axes[0], "Validation return"), (axes[1], "Validation success rate")]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if x == "env_steps":
            ax.set_xscale("log")
    axes[0].set_ylim(bottom=max(axes[0].get_ylim()[0], -2500))
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="upper left")
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
        ax.set_title(f"{name}: success rate")
        ax.set_xlabel(r"Total cable length $L_1+L_2$ (m)")
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
