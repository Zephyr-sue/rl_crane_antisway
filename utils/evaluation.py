"""Episode rollout and performance metrics shared by evaluate.py, viz and training callbacks."""
from typing import Dict, List

import numpy as np

from envs.double_pendulum_crane_env import DoublePendulumCraneEnv
from envs.scenario import Scenario


def rollout(env: DoublePendulumCraneEnv, controller, scenario: Scenario) -> Dict[str, np.ndarray]:
    """Run one full episode on a fixed scenario and record the trajectory."""
    obs, info = env.reset(options={"scenario": scenario})
    traj = {k: [] for k in ("t", "x", "x_dot", "th1", "th2", "u", "gust", "reward", "settled", "payload_x")}
    done = False
    while not done:
        action = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        x, x_dot, th1, _, th2, _ = env.state
        traj["t"].append(env.current_step * env.dt)
        traj["x"].append(x)
        traj["x_dot"].append(x_dot)
        traj["th1"].append(th1)
        traj["th2"].append(th2)
        traj["u"].append(float(np.clip(action[0], -1, 1)))
        traj["gust"].append(info["gust"])
        traj["reward"].append(reward)
        traj["settled"].append(info["settled"])
        traj["payload_x"].append(env.dynamics.payload_x(env.state))
    out = {k: np.asarray(v) for k, v in traj.items()}
    out["target_x"] = scenario.target_x
    out["params"] = (scenario.m2, scenario.L1, scenario.L2)
    return out


def episode_metrics(traj: Dict[str, np.ndarray], dt: float = 0.02, max_force: float = 30.0) -> Dict[str, float]:
    settled = traj["settled"]
    # Settling time: first instant after which the crane stays settled until the episode ends
    unsettled_idx = np.flatnonzero(~settled)
    if len(unsettled_idx) == 0:
        settle_time = 0.0
    elif unsettled_idx[-1] == len(settled) - 1:
        settle_time = np.nan
    else:
        settle_time = (unsettled_idx[-1] + 1) * dt
    # Success: settled during the final second
    success = float(settled[-int(1.0 / dt):].all())

    target = traj["target_x"]
    direction = np.sign(target) if target != 0 else 1.0
    overshoot = max(0.0, float(np.max(direction * (traj["x"] - target))))
    sway = np.maximum(np.abs(traj["th1"]), np.abs(traj["th2"]))
    last = slice(-int(2.0 / dt), None)
    return {
        "success": success,
        "settle_time": settle_time,
        "max_sway_deg": float(np.rad2deg(sway.max())),
        "residual_sway_deg": float(np.rad2deg(np.sqrt(np.mean(traj["th1"][last] ** 2 + traj["th2"][last] ** 2)))),
        "final_error_cm": float(100 * abs(traj["x"][-1] - target)),
        "overshoot_cm": 100 * overshoot,
        "control_energy": float(np.sum((traj["u"] * max_force) ** 2) * dt),
        "return": float(np.sum(traj["reward"])),
    }


def evaluate_controller(controller, scenarios: List[Scenario], env: DoublePendulumCraneEnv = None) -> List[Dict]:
    env = env or DoublePendulumCraneEnv()
    return [episode_metrics(rollout(env, controller, sc), env.dt, env.max_force) for sc in scenarios]


def summarize(metrics: List[Dict]) -> Dict[str, float]:
    keys = metrics[0].keys()
    return {k: float(np.nanmean([m[k] for m in metrics])) if not all(np.isnan(m[k]) for m in metrics) else np.nan
            for k in keys}
