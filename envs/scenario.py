"""Replayable crane task scenarios: payload/cable parameters, target, initial sway, gust schedule.

A scenario is fully pre-sampled, so the same task can be replayed exactly. GRPO relies on
this: every trajectory in a group faces the identical scenario, so differences in return
come only from the policy's own action sampling.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Nominal plant and domain-randomization ranges (full-difficulty, level = 1.0)
NOMINAL = {"m2": 2.0, "L1": 1.0, "L2": 0.8}
RANGES = {"m2": (0.5, 4.0), "L1": (0.6, 1.4), "L2": (0.4, 1.2)}
TARGET_RANGE = (1.0, 4.0)
MAX_INIT_ANGLE_DEG = 30.0
MAX_INIT_RATE = 0.3
GUST_FORCE_RANGE = (2.0, 6.0)
GUST_DURATION_RANGE = (0.2, 0.5)
MAX_GUSTS = 3


@dataclass
class Scenario:
    m2: float
    L1: float
    L2: float
    target_x: float
    init_state: np.ndarray            # (6,) [x, x_dot, th1, th1_dot, th2, th2_dot]
    gusts: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (max_steps,) N on payload

    def gust_at(self, step: int) -> float:
        return float(self.gusts[step]) if step < len(self.gusts) else 0.0


def _lerp_range(lo: float, hi: float, nominal: float, level: float):
    return nominal + level * (lo - nominal), nominal + level * (hi - nominal)


def sample_gusts(rng: np.random.Generator, max_steps: int, dt: float, level: float = 1.0) -> np.ndarray:
    """Piecewise-constant gust profile with up to MAX_GUSTS gusts of random sign."""
    gusts = np.zeros(max_steps, dtype=np.float64)
    if max_steps <= int(1.0 / dt):
        return gusts
    n_gusts = int(rng.integers(0, MAX_GUSTS + 1))
    for _ in range(n_gusts):
        start = int(rng.integers(int(1.0 / dt), max_steps))
        duration = int(rng.uniform(*GUST_DURATION_RANGE) / dt)
        magnitude = level * rng.uniform(*GUST_FORCE_RANGE) * rng.choice([-1.0, 1.0])
        gusts[start:start + duration] += magnitude
    return gusts


def sample_scenario(
    rng: np.random.Generator,
    max_steps: int = 600,
    dt: float = 0.02,
    level: float = 1.0,
    randomize_params: bool = True,
    overrides: Optional[dict] = None,
) -> Scenario:
    """Sample a scenario. `level` in [0, 1] widens parameter ranges, initial sway and gusts."""
    params = dict(NOMINAL)
    if randomize_params:
        for key, (lo, hi) in RANGES.items():
            params[key] = float(rng.uniform(*_lerp_range(lo, hi, NOMINAL[key], level)))

    target = float(rng.uniform(*TARGET_RANGE) * rng.choice([-1.0, 1.0]))
    max_angle = np.deg2rad(max(5.0, MAX_INIT_ANGLE_DEG * level))
    init_state = np.array([
        0.0,
        0.0,
        rng.uniform(-max_angle, max_angle),
        rng.uniform(-MAX_INIT_RATE, MAX_INIT_RATE) * level,
        rng.uniform(-max_angle, max_angle),
        rng.uniform(-MAX_INIT_RATE, MAX_INIT_RATE) * level,
    ], dtype=np.float64)
    gusts = sample_gusts(rng, max_steps, dt, level)

    scenario = Scenario(target_x=target, init_state=init_state, gusts=gusts, **params)
    for key, value in (overrides or {}).items():
        setattr(scenario, key, value)
    return scenario
