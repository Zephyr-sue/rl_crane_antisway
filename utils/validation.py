"""Fast policy validation on a fixed scenario set using the batched (GPU) environment.

Used for learning curves of every algorithm, so GRPO, PPO and SAC are scored identically.
"""
from typing import Callable, Tuple

import numpy as np
import torch

from envs.batched_crane import BatchedCraneEnv
from envs.scenario import sample_scenario

VAL_SEED = 2026


class Validator:
    def __init__(self, device: torch.device, n_scenarios: int = 64, observe_params: bool = True):
        rng = np.random.default_rng(VAL_SEED)
        self.env = BatchedCraneEnv(device, observe_params=observe_params)
        self.scenarios = [sample_scenario(rng, self.env.max_steps, self.env.dt, level=1.0) for _ in range(n_scenarios)]

    @torch.no_grad()
    def __call__(self, act_fn: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[float, float]:
        """act_fn maps a (B, D) observation tensor to (B,) normalized forces."""
        obs = self.env.reset(self.scenarios)
        total = torch.zeros(len(self.scenarios), device=self.env.device)
        settled_tail = torch.ones(len(self.scenarios), dtype=torch.bool, device=self.env.device)
        tail_start = self.env.max_steps - int(1.0 / self.env.dt)
        for t in range(self.env.max_steps):
            obs, r, settled = self.env.step(act_fn(obs))
            total += r
            if t >= tail_start:
                settled_tail &= settled
        return total.mean().item(), settled_tail.float().mean().item()
