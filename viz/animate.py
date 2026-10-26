"""Animated crane GIF/MP4 (matplotlib): one controller, or several side by side on the same scenario.

    uv run viz/animate.py                                 # all controllers -> results/compare.gif
    uv run viz/animate.py --controllers GRPO --out results/grpo.gif
    uv run viz/animate.py --condition "Strong gusts" --index 2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter  # noqa: E402

from envs.double_pendulum_crane_env import DoublePendulumCraneEnv  # noqa: E402
from evaluate import CONDITIONS, CONTROLLERS, make_controller, make_scenarios  # noqa: E402
from utils.evaluation import rollout  # noqa: E402
from utils.plotting import COLORS, GRID, INK, INK_2, setup_style  # noqa: E402


def animate(trajs: dict, out_path: str, stride: int = 3, fps: int = 17, title: str = ""):
    setup_style()
    names = list(trajs)
    n = len(names)
    cols = min(n, 3)  # controllers side by side
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(4.2 * cols, 2.3 * rows + 2.2))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [0.9])
    any_tr = trajs[names[0]]
    m2, L1, L2 = any_tr["params"]
    target = any_tr["target_x"]
    T = len(any_tr["t"])
    frames = range(0, T, stride)

    scenes = []
    for k, name in enumerate(names):
        ax = fig.add_subplot(gs[k // cols, k % cols])
        tr = trajs[name]
        lo = min(tr["x"].min(), target, 0) - 1.2
        hi = max(tr["x"].max(), target, 0) + 1.2
        ax.set_xlim(lo, hi)
        ax.set_ylim(-(L1 + L2) - 0.4, 0.4)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_yticks([])
        ax.axhline(0, color=INK_2, linewidth=2)
        ax.axvline(target, color=INK_2, linestyle="--", linewidth=1)
        ax.set_title(name, color=INK, fontweight="bold")
        trolley, = ax.plot([], [], "s", color=COLORS.get(name, INK_2), markersize=12)
        cables, = ax.plot([], [], "-", color=INK_2, linewidth=1.2)
        hook, = ax.plot([], [], "o", color=INK_2, markersize=5)
        load, = ax.plot([], [], "s", color=COLORS.get(name, INK_2), markersize=6 + 2 * np.sqrt(m2),
                        markeredgecolor=INK, markeredgewidth=0.6)
        gust_txt = ax.text(0.02, 0.05, "", transform=ax.transAxes, color="#6250d6", fontsize=8)
        info = ax.text(0.98, 0.05, "", transform=ax.transAxes, ha="right", fontsize=8, color=INK)
        scenes.append((tr, trolley, cables, hook, load, gust_txt, info))

    ax_s = fig.add_subplot(gs[rows, :])
    cursors = []
    for name in names:
        tr = trajs[name]
        ax_s.plot(tr["t"], np.rad2deg(tr["th2"]), color=COLORS.get(name, INK_2), label=name, linewidth=1.2)
    for t0, t1 in _gust_intervals(any_tr):
        ax_s.axvspan(t0, t1, color=GRID, linewidth=0)
    cursor = ax_s.axvline(0, color=INK, linewidth=0.8)
    cursors.append(cursor)
    ax_s.set_xlabel("Time (s)   (shaded: wind gust)")
    ax_s.set_ylabel(r"Payload sway $\theta_2$ (°)")
    ax_s.legend(ncol=n, loc="upper right")
    fig.suptitle(title or f"$m_2$ = {m2:.1f} kg,  $L_1$ = {L1:.2f} m,  $L_2$ = {L2:.2f} m", fontsize=10)
    fig.tight_layout()

    def update(i):
        artists = []
        for tr, trolley, cables, hook, load, gust_txt, info in scenes:
            x, th1, th2 = tr["x"][i], tr["th1"][i], tr["th2"][i]
            hx, hy = x + L1 * np.sin(th1), -L1 * np.cos(th1)
            lx, ly = hx + L2 * np.sin(th2), hy - L2 * np.cos(th2)
            trolley.set_data([x], [0])
            cables.set_data([x, hx, lx], [0, hy, ly])
            hook.set_data([hx], [hy])
            load.set_data([lx], [ly])
            g = tr["gust"][i]
            gust_txt.set_text(f"gust {g:+.1f} N" if g else "")
            status = "settled" if tr["settled"][i] else ""
            info.set_text(f"t={tr['t'][i]:4.1f}s  F={30 * tr['u'][i]:+5.1f}N  {status}")
            artists += [trolley, cables, hook, load, gust_txt, info]
        cursor.set_xdata([any_tr["t"][i]])
        return artists + [cursor]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.endswith(".mp4"):
        anim.save(out_path, writer=FFMpegWriter(fps=fps), dpi=110)
    else:
        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"Saved {out_path}")


def _gust_intervals(tr):
    mask = tr["gust"] != 0
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = tr["t"][i]
        if not m and start is not None:
            out.append((start, tr["t"][i]))
            start = None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controllers", nargs="+", default=None)
    p.add_argument("--condition", default="Random", choices=list(CONDITIONS))
    p.add_argument("--index", type=int, default=3, help="scenario index within the condition's test set")
    p.add_argument("--out", default=None)
    p.add_argument("--stride", type=int, default=3)
    args = p.parse_args()

    names = [c.upper() for c in args.controllers] if args.controllers else CONTROLLERS
    out = args.out or os.path.join("results", "compare.gif" if len(names) > 1 else f"{names[0].lower()}.gif")

    sc = make_scenarios(args.condition, args.index + 1)[args.index]
    env = DoublePendulumCraneEnv()
    trajs = {n: rollout(env, make_controller(n), sc) for n in names}
    animate(trajs, out, stride=args.stride)


if __name__ == "__main__":
    main()
