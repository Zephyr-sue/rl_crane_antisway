"""Benchmark GRPO and residual GRPO against PPO, LQR and the RBF adaptive controller; produce all figures/tables.

Examples:
    uv run evaluate.py                   # full benchmark (100 episodes per condition) + heatmap + ablations
    uv run evaluate.py --episodes 20     # quicker
    uv run evaluate.py --only-plots      # regenerate figures from results/*.csv / *.npz
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from envs.scenario import sample_scenario, sample_gusts
from models import default_path, TRAINED_DIR
from utils import plotting

RESULTS = "results"
EVAL_SEED = 7
CONTROLLERS = ["Res-GRPO", "GRPO", "PPO", "LQR", "RBF"]

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


def canonical_name(name: str) -> str:
    """Case-insensitive controller lookup: 'res-grpo' -> 'Res-GRPO'."""
    for c in CONTROLLERS:
        if c.lower() == name.lower():
            return c
    raise ValueError(f"unknown controller {name!r}; choose from {', '.join(CONTROLLERS)}")


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


def make_controller(name: str):
    """'Res-GRPO', 'GRPO', 'PPO', 'LQR', 'RBF', or a path to any trained checkpoint (used for ablations)."""
    import torch
    torch.set_num_threads(1)
    from baselines import LQRController, RBFAdaptiveController
    from models import load_agent
    if name == "RBF":
        return RBFAdaptiveController()
    if name == "LQR":
        return LQRController()
    if name == "Res-GRPO":
        return load_agent(default_path(0, "resgrpo"), device="cpu")
    if name in ("GRPO", "PPO"):
        return load_agent(default_path(0, name.lower()), device="cpu")
    return load_agent(name, device="cpu")


def _eval_job(job):
    name, scenarios, tag = job
    from envs.double_pendulum_crane_env import DoublePendulumCraneEnv
    from utils.evaluation import episode_metrics, rollout
    ctrl = make_controller(name)
    env = DoublePendulumCraneEnv(observe_params=getattr(ctrl, "observe_params", True))
    return [{"controller": name, **tag, "episode": i, **episode_metrics(rollout(env, ctrl, sc))}
            for i, sc in enumerate(scenarios)]


def _run_jobs(jobs, workers):
    with Pool(workers) as pool:
        return pd.DataFrame([r for res in pool.imap_unordered(_eval_job, jobs) for r in res])


def run_benchmark(episodes, workers):
    jobs = []
    for cond in CONDITIONS:
        scenarios = make_scenarios(cond, episodes)
        for chunk in np.array_split(np.arange(episodes), max(1, episodes // 25)):
            for name in CONTROLLERS:
                jobs.append((name, [scenarios[i] for i in chunk], {"condition": cond}))
    return _run_jobs(jobs, workers)


def run_heatmap(per_cell, workers, n=8):
    m2_vals = np.linspace(0.5, 6.0, n)
    l_vals = np.linspace(0.8, 3.2, n)
    jobs = []
    for i, m2 in enumerate(m2_vals):
        for j, lt in enumerate(l_vals):
            rng = np.random.default_rng([EVAL_SEED, i, j])
            scs = [sample_scenario(rng, overrides={"m2": m2, "L1": 0.55 * lt, "L2": 0.45 * lt}) for _ in range(per_cell)]
            for name in CONTROLLERS:
                jobs.append((name, scs, {"i": i, "j": j}))
    df = _run_jobs(jobs, workers)
    grids = {}
    for name in CONTROLLERS:
        g = np.zeros((n, n))
        for (i, j), v in df[df.controller == name].groupby(["i", "j"])["success"].mean().items():
            g[i, j] = v
        grids[name] = g
    return grids, m2_vals, l_vals


def run_ablations(episodes, workers):
    """Evaluate every GRPO checkpoint in models/trained/ablations (plus the main one) on random scenarios."""
    paths = {"rtg (main)": default_path()}
    for p in sorted(glob.glob(os.path.join(TRAINED_DIR, "ablations", "*.pt"))):
        paths[os.path.basename(p).split("_grpo_seed")[0].replace("noparams", "no params")] = p
    scenarios = make_scenarios("Random", episodes)
    jobs = [(p, scenarios, {"variant": v}) for v, p in paths.items()]
    df = _run_jobs(jobs, workers)
    return df.groupby("variant", sort=False).agg(
        success=("success", "mean"), settle_time=("settle_time", "mean"),
        max_sway_deg=("max_sway_deg", "mean"), ret=("return", "mean")).reset_index()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["condition", "controller"], sort=False).agg(
        success=("success", "mean"),
        settle_time=("settle_time", "mean"),
        max_sway_deg=("max_sway_deg", "mean"),
        residual_sway_deg=("residual_sway_deg", "mean"),
        final_error_cm=("final_error_cm", "mean"),
        overshoot_cm=("overshoot_cm", "mean"),
        control_energy=("control_energy", "mean"),
        ret=("return", "mean"),
    ).reset_index()


def write_latex_table(summary: pd.DataFrame, path: str):
    """Per-condition success and settling time for every controller (best in bold)."""
    n = len(CONTROLLERS)
    head = " & ".join(CONTROLLERS)
    lines = [r"\begin{tabular}{l " + "c" * n + " " + "c" * n + "}", r"\toprule",
             r"& \multicolumn{%d}{c}{Success (\%%)} & \multicolumn{%d}{c}{$t_s$ (s)} \\" % (n, n),
             r"Condition & %s & %s \\" % (head, head), r"\midrule"]
    for cond in CONDITIONS:
        rows = [summary[(summary.condition == cond) & (summary.controller == c)].iloc[0] for c in CONTROLLERS]
        best_s = max(r.success for r in rows)
        best_t = np.nanmin([r.settle_time for r in rows])
        succ = [(r"\textbf{%.0f}" if r.success == best_s else "%.0f") % (100 * r.success) for r in rows]
        ts = ["--" if np.isnan(r.settle_time) else
              ((r"\textbf{%.1f}" if r.settle_time == best_t else "%.1f") % r.settle_time) for r in rows]
        lines.append(cond + " & " + " & ".join(succ + ts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_trajectory_figure(out_path, condition="Random", index=3):
    from envs.double_pendulum_crane_env import DoublePendulumCraneEnv
    from utils.evaluation import rollout
    sc = make_scenarios(condition, index + 1)[index]
    env = DoublePendulumCraneEnv()
    trajs = {n: rollout(env, make_controller(n), sc) for n in CONTROLLERS}
    plotting.plot_trajectories(trajs, out_path,
                               title=f"$m_2$={sc.m2:.1f} kg, $L_1$={sc.L1:.2f} m, $L_2$={sc.L2:.2f} m")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--heatmap-episodes", type=int, default=10)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--only-plots", action="store_true")
    p.add_argument("--skip-heatmap", action="store_true")
    p.add_argument("--skip-ablations", action="store_true")
    args = p.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    if not args.only_plots:
        df = run_benchmark(args.episodes, args.workers)
        df.to_csv(os.path.join(RESULTS, "metrics.csv"), index=False)
        summarize(df).to_csv(os.path.join(RESULTS, "summary.csv"), index=False)
    summary = pd.read_csv(os.path.join(RESULTS, "summary.csv"))
    write_latex_table(summary, os.path.join(RESULTS, "summary_table.tex"))
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.precision", 2):
        print(summary.set_index(["condition", "controller"])[["success", "settle_time", "max_sway_deg", "residual_sway_deg", "control_energy"]])

    plotting.plot_learning_curves("runs", os.path.join(RESULTS, "learning_curves.png"))
    plotting.plot_condition_bars(summary, "success", "Success rate", os.path.join(RESULTS, "success_by_condition.png"))
    plotting.plot_condition_bars(summary, "max_sway_deg", "Max sway (°)", os.path.join(RESULTS, "sway_by_condition.png"))
    make_trajectory_figure(os.path.join(RESULTS, "trajectories.png"))

    if not args.skip_heatmap:
        heat_path = os.path.join(RESULTS, "heatmap.npz")
        if not args.only_plots or not os.path.exists(heat_path):
            grids, m2_vals, l_vals = run_heatmap(args.heatmap_episodes, args.workers)
            np.savez(heat_path, names=CONTROLLERS, m2=m2_vals, l=l_vals, **{f"g_{n}": grids[n] for n in CONTROLLERS})
        data = np.load(heat_path)
        plotting.plot_heatmaps({str(n): data[f"g_{n}"] for n in data["names"]}, data["m2"], data["l"],
                               os.path.join(RESULTS, "robustness_heatmap.png"), train_box=((1.0, 2.6), (0.5, 4.0)))

    if not args.skip_ablations and not args.only_plots:
        abl = run_ablations(args.episodes, args.workers)
        abl.to_csv(os.path.join(RESULTS, "ablations.csv"), index=False)
        print(abl)
    print(f"Saved results to {RESULTS}/")


if __name__ == "__main__":
    main()
