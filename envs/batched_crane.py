"""Batched crane environment in PyTorch, for massively parallel (GPU) rollouts.

Implements exactly the same task as DoublePendulumCraneEnv (same scenarios, observation,
reward and end-stop handling) for B cranes at once. Used by GRPO, where groups of G
rollouts share a scenario.
"""
from typing import List

import numpy as np
import torch

from envs.double_pendulum_crane_env import TRACK_LIMIT
from envs.observation import build_obs
from envs.reward import compute_reward, is_settled
from envs.scenario import Scenario
from utils.physics_torch import TorchCraneDynamics


class BatchedCraneEnv:
    def __init__(
        self,
        device: torch.device,
        dt: float = 0.02,
        max_steps: int = 600,
        max_force: float = 30.0,
        observe_params: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dt = dt
        self.max_steps = max_steps
        self.max_force = max_force
        self.observe_params = observe_params
        self.dtype = dtype
        self.dynamics = TorchCraneDynamics()

    def _t(self, values) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values), dtype=self.dtype, device=self.device)

    def reset(self, scenarios: List[Scenario]) -> torch.Tensor:
        """Start one rollout per scenario (repeat a scenario G times to form a GRPO group)."""
        self.m2 = self._t([s.m2 for s in scenarios])
        self.L1 = self._t([s.L1 for s in scenarios])
        self.L2 = self._t([s.L2 for s in scenarios])
        self.target = self._t([s.target_x for s in scenarios])
        self.gusts = self._t(np.stack([s.gusts[:self.max_steps] for s in scenarios], 0))  # (B, T)
        self.state = self._t(np.stack([s.init_state for s in scenarios], 0))              # (B, 6)
        self.u_prev = torch.zeros_like(self.m2)
        self.t = 0
        return self.obs()

    def obs(self) -> torch.Tensor:
        s = self.state
        feats = build_obs(torch, s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5], self.target,
                          self.u_prev, self.m2, self.L1, self.L2, self.observe_params)
        return torch.stack(feats, -1)

    def step(self, u: torch.Tensor):
        """u: (B,) normalized force. Returns obs (B, D), reward (B,), settled (B,) bool."""
        u = u.clamp(-1.0, 1.0)
        gust = self.gusts[:, self.t]
        self.state = self.dynamics.rk4_step(self.state, u * self.max_force, self.dt, gust,
                                            self.m2, self.L1, self.L2)
        self.t += 1

        hit_limit = self.state[:, 0].abs() > TRACK_LIMIT
        x = self.state[:, 0].clamp(-TRACK_LIMIT, TRACK_LIMIT)
        x_dot = torch.where(hit_limit, torch.zeros_like(x), self.state[:, 1])
        self.state = torch.cat([x[:, None], x_dot[:, None], self.state[:, 2:]], dim=1)

        s = self.state
        e = s[:, 0] - self.target
        reward = compute_reward(e, s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5], u, self.u_prev, hit_limit)
        settled = is_settled(e, s[:, 1], s[:, 2], s[:, 4])
        self.u_prev = u
        return self.obs(), reward, settled

    @property
    def done(self) -> bool:
        return self.t >= self.max_steps
