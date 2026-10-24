"""Interactive real-time crane demo (pygame).

    uv run viz/live.py [--agent grpo]

Keys:
    1 GRPO   2 PPO   3 SAC   4 LQR   5 SMC   (switch controller live)
    mouse click    set a new target position
    G              wind gust on the payload (hold for longer)
    + / -          change payload mass mid-run
    R              new random scenario
    Esc / Q        quit
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from envs.double_pendulum_crane_env import DoublePendulumCraneEnv  # noqa: E402
from evaluate import make_controller, available_controllers  # noqa: E402
from viz.renderer import CraneRenderer, SCALE, WIDTH  # noqa: E402

KEYMAP = {"1": "GRPO", "2": "PPO", "3": "SAC", "4": "LQR", "5": "SMC"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="grpo")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    available = available_controllers()
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = make_controller(name)
        return cache[name]

    name = args.agent.upper() if args.agent.upper() in available else available[0]
    env = DoublePendulumCraneEnv()  # truncation flag is ignored: the demo runs until quit
    renderer = CraneRenderer(headless=False, title="RL anti-sway crane")
    pg = renderer.pg
    rng = np.random.default_rng(args.seed)
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    settled_since = None

    running = True
    while running:
        manual_gust = 0.0
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                key = event.unicode.lower()
                if event.key in (pg.K_ESCAPE, pg.K_q):
                    running = False
                elif key in KEYMAP and KEYMAP[key] in available:
                    name = KEYMAP[key]
                elif key == "r":
                    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
                    settled_since = None
                elif key in ("+", "="):
                    env.dynamics.m2 = env.scenario.m2 = min(6.0, env.dynamics.m2 + 0.5)
                elif key == "-":
                    env.dynamics.m2 = env.scenario.m2 = max(0.5, env.dynamics.m2 - 0.5)
            elif event.type == pg.MOUSEBUTTONDOWN:
                env.scenario.target_x = float(np.clip((event.pos[0] - WIDTH / 2) / SCALE, -5.5, 5.5))
                settled_since = None
        if pg.key.get_pressed()[pg.K_g]:
            manual_gust = 6.0

        # Inject manual gust through the scenario's gust schedule for this step
        step = env.current_step
        if step >= len(env.scenario.gusts):
            env.scenario.gusts = np.concatenate([env.scenario.gusts, np.zeros(10_000)])
        env.scenario.gusts[step] += manual_gust
        obs = env._get_obs()

        obs, _, _, _, info = env.step(get(name).act(obs))
        if info["settled"]:
            settled_since = settled_since if settled_since is not None else env.current_step * env.dt
        else:
            settled_since = None
        status = f"SETTLED since {settled_since:.1f} s" if settled_since is not None else "moving / swaying"
        renderer.draw(env.state, env.dynamics, env.target_x, info["force"] / env.max_force, info["gust"],
                      env.current_step * env.dt,
                      hud_lines=[f"controller: {name}   [{status}]",
                                 "1-5 ctrl  click target  G gust  +/- mass  R reset"],
                      title=f"Controller: {name}")
    renderer.close()


if __name__ == "__main__":
    main()
