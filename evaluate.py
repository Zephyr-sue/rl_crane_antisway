"""Benchmark trained RL agents against classical baselines and produce all figures/tables.

Examples:
    uv run evaluate.py                   # full benchmark (100 episodes per condition)
    uv run evaluate.py --episodes 20     # quicker
    uv run evaluate.py --only-plots      # regenerate figures from results/*.csv
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from envs.scenario import sample_scenario, sample_gusts
from models import default_path
from utils import plotting

RESULTS = "results"
EVAL_SEED = 7

# Test conditions: name -> (scenario overrides, gust intensity multiplier)
CONDITIONS = {
    "Nominal": ({"m2": 2.0, "L1": 1.0, "L2": 0.8}, 1.0),
    "Random": ({}, 1.0),
    "Heavy": ({"m2": 4.0}, 1.0),
    "Light": ({"m2": 0.5}, 1.0),
    "Long cable": ({"L1": 1.4, "L2": 1.2}, 1.0),
    "Short cable": ({"L1": 0.6, "L2": 0.4}, 1.0),
    "Strong gusts": ({}, 2.0),
    "OOD": ({"m2": 5.0, "L1": 1.5, "L2": 1.3}, 1.0),
}


def make_scenarios(condition: str, n: int, seed: int = EVAL_SEED):
    overrides, gust_scale = CONDITIONS[condition]
    rng = np.random.default_rng([seed, list(CONDITIONS).index(condition)])
    scenarios = []
    for _ in range(n):
        sc = sample_scenario(rng, overrides=overrides)
        if gust_scale != 1.0:
            sc.gusts = sample_gusts(rng, len(sc.gusts), 0.02, level=gust_scale)
        scenarios.append(sc)
    return scenarios


def available_controllers(seed: int = 0):
    names = [n for n, algo in [("GRPO", "grpo"), ("PPO", "ppo"), ("SAC", "sac")] if os.path.exists(default_path(algo, seed))]
    return names + ["LQR", "SMC", "Zero"]


def make_controller(name: str, seed: int = 0):
    import torch
    torch.set_num_threads(1)
    from baselines import LQRController, SlidingModeController, ZeroController
    from models import load_agent
    if name in ("GRPO", "PPO", "SAC"):
        return load_agent(name.lower(), default_path(name.lower(), seed), device="cpu")
    return {"LQR": LQRController, "SMC": SlidingModeController, "Zero": ZeroController}[name]()


def _eval_job(job):
    name, seed, scenarios, tag = job
    from envs.double_pendulum_crane_env import DoublePendulumCraneEnv
    from utils.evaluation import episode_metrics, rollout
    ctrl = make_controller(name, seed)
    env = DoublePendulumCraneEnv()
    rows = []
    for i, sc in enumerate(scenarios):
        m = episode_metrics(rollout(env, ctrl, sc))
        rows.append({"controller": name, "seed": seed, **tag, "episode": i, **m})
    return rows


def run_benchmark(controllers, episodes, workers, seed=0):
    jobs = []
    for cond in CONDITIONS:
        scenarios = make_scenarios(cond, episodes)
        chunks = np.array_split(np.arange(episodes), max(1, episodes // 25))
        for name in controllers:
            for chunk in chunks:
                jobs.append((name, seed, [scenarios[i] for i in chunk], {"condition": cond}))
    with Pool(workers) as pool:
        rows = [r for res in pool.imap_unordered(_eval_job, jobs) for r in res]
    return pd.DataFrame(rows)


def run_heatmap(controllers, per_cell, workers, n=8):
    m2_vals = np.linspace(0.5, 6.0, n)
    l_vals = np.linspace(0.8, 3.2, n)
    jobs = []
    for i, m2 in enumerate(m2_vals):
        for j, lt in enumerate(l_vals):
            rng = np.random.default_rng([EVAL_SEED, i, j])
            scs = [sample_scenario(rng, overrides={"m2": m2, "L1": 0.55 * lt, "L2": 0.45 * lt}) for _ in range(per_cell)]
            for name in controllers:
                jobs.append((name, 0, scs, {"i": i, "j": j}))
    with Pool(workers) as pool:
        rows = [r for res in pool.imap_unordered(_eval_job, jobs) for r in res]
    df = pd.DataFrame(rows)
    grids = {}
    for name in controllers:
        g = np.zeros((n, n))
        sub = df[df.controller == name].groupby(["i", "j"])["success"].mean()
        for (i, j), v in sub.items():
            g[i, j] = v
        grids[name] = g
    return grids, m2_vals, l_vals


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["condition", "controller"], sort=False).agg(
        success=("success", "mean"),
        settle_time=("settle_time", "mean"),
        max_sway_deg=("max_sway_deg", "mean"),
        residual_sway_deg=("residual_sway_deg", "mean"),
        final_error_cm=("final_error_cm", "mean"),
        overshoot_cm=("overshoot_cm", "mean"),
        control_energy=("control_energy", "mean"),
        ret=("return", "mean"),
    ).reset_index()
    return agg


def write_latex_table(summary: pd.DataFrame, path: str):
    """Compact table averaged over all conditions (settling time over successful episodes)."""
    overall = summary.groupby("controller").mean(numeric_only=True)
    overall = overall.reindex([c for c in plotting.ORDER if c in overall.index])
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Ctrl. & Succ. (\%) & $t_s$ (s) & Max sway ($^\circ$) & Resid. ($^\circ$) \\", r"\midrule"]
    for name, r in overall.iterrows():
        ts = "--" if np.isnan(r.settle_time) else f"{r.settle_time:.2f}"
        lines.append(f"{name} & {100 * r.success:.1f} & {ts} & {r.max_sway_deg:.1f} & {r.residual_sway_deg:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_trajectory_figure(controllers, out_path):
    from envs.double_pendulum_crane_env import DoublePendulumCraneEnv
    from utils.evaluation import rollout
    sc = make_scenarios("Random", 6)[3]
    env = DoublePendulumCraneEnv()
    trajs = {n: rollout(env, make_controller(n), sc) for n in controllers if n != "Zero"}
    plotting.plot_trajectories(
        trajs, out_path,
        title=f"$m_2$={sc.m2:.1f} kg, $L_1$={sc.L1:.2f} m, $L_2$={sc.L2:.2f} m")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--heatmap-episodes", type=int, default=10)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--only-plots", action="store_true")
    p.add_argument("--skip-heatmap", action="store_true")
    args = p.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    controllers = available_controllers()
    print("Controllers:", controllers)

    if not args.only_plots:
        df = run_benchmark(controllers, args.episodes, args.workers)
        df.to_csv(os.path.join(RESULTS, "metrics.csv"), index=False)
        summary = summarize(df)
        summary.to_csv(os.path.join(RESULTS, "summary.csv"), index=False)
    summary = pd.read_csv(os.path.join(RESULTS, "summary.csv"))
    write_latex_table(summary, os.path.join(RESULTS, "summary_table.tex"))

    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.precision", 2):
        print(summary.pivot(index="controller", columns="condition", values="success").reindex(
            [c for c in plotting.ORDER if c in controllers]))
        print(summary.groupby("controller").mean(numeric_only=True))

    plotting.plot_learning_curves("runs", os.path.join(RESULTS, "learning_curves.png"))
    plotting.plot_learning_curves("runs", os.path.join(RESULTS, "learning_curves_walltime.png"), x="wall_time")
    plotting.plot_condition_bars(summary, "success", "Success rate", os.path.join(RESULTS, "success_by_condition.png"))
    plotting.plot_condition_bars(summary, "max_sway_deg", "Max sway (°)", os.path.join(RESULTS, "sway_by_condition.png"))
    make_trajectory_figure(controllers, os.path.join(RESULTS, "trajectories.png"))

    if not args.skip_heatmap:
        heat_path = os.path.join(RESULTS, "heatmap.npz")
        if not args.only_plots or not os.path.exists(heat_path):
            rl = [c for c in controllers if c in ("GRPO", "PPO", "SAC")]
            best_rl = max(rl, key=lambda c: summary[summary.controller == c]["success"].mean()) if rl else None
            names = ([best_rl] if best_rl else []) + ["LQR"]
            grids, m2_vals, l_vals = run_heatmap(names, args.heatmap_episodes, args.workers)
            np.savez(heat_path, names=names, m2=m2_vals, l=l_vals, **{f"g_{n}": grids[n] for n in names})
        data = np.load(heat_path)
        grids = {str(n): data[f"g_{n}"] for n in data["names"]}
        plotting.plot_heatmaps(grids, data["m2"], data["l"], os.path.join(RESULTS, "robustness_heatmap.png"),
                               train_box=((1.0, 2.6), (0.5, 4.0)))
    print(f"Saved results to {RESULTS}/")


if __name__ == "__main__":
    main()
