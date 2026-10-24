"""Group Relative Policy Optimization (GRPO) adapted to continuous crane control.

GRPO (Shao et al., 2024, "DeepSeekMath", arXiv:2402.03300, Sec. 4.1) drops PPO's value
network and uses the *group* of G outputs sampled for the same prompt as the baseline.
Here a "prompt" is a crane scenario (payload, cable lengths, target, initial sway and gust
schedule) and an "output" is a full 12 s trajectory. All G trajectories in a group replay
the identical scenario on the GPU, so their return differences come only from the policy.

Advantage estimators (`adv_mode`):
  outcome : A_{i,t} = (R_i - mean(R)) / std(R)                         (paper, outcome supervision)
  process : r~ = (r - mean(r)) / std(r) over all group steps,
            A_{i,t} = sum_{j>=t} r~_{i,j}                                (paper, process supervision)
  rtg     : A_{i,t} = (G_{i,t} - mean_i G_{.,t}) / std_i G_{.,t}, with G the discounted
            return-to-go; a per-timestep group baseline (our adaptation for dense rewards)

Objective (paper Eq. 3) with the unbiased KL estimator of Eq. 4:
  J = 1/G sum_i 1/|o_i| sum_t { min(rho A, clip(rho, 1-eps, 1+eps) A) - beta KL[pi || pi_ref] }
  KL = pi_ref/pi - log(pi_ref/pi) - 1
"""
import copy
import csv
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from envs.batched_crane import BatchedCraneEnv
from envs.scenario import sample_scenario


@dataclass
class GRPOConfig:
    n_groups: int = 32            # scenarios ("prompts") per iteration
    group_size: int = 32          # G trajectories per scenario
    iterations: int = 300
    lr: float = 3e-4
    clip_eps: float = 0.2
    beta: float = 0.04            # KL coefficient (paper value)
    ref_update_every: int = 10    # iterative GRPO: pi_ref <- pi every K iterations
    epochs: int = 4
    minibatch_size: int = 32768
    adv_mode: str = "rtg"         # "outcome" | "process" | "rtg"
    gamma: float = 0.995          # discount for "rtg" mode
    hidden: int = 128
    init_log_std: float = -0.5
    max_grad_norm: float = 0.5
    curriculum_frac: float = 0.3  # fraction of training to ramp scenario difficulty 0.3 -> 1.0
    observe_params: bool = True
    seed: int = 0


class GaussianPolicy(nn.Module):
    """MLP mean with state-independent log-std; actions are clipped to [-1, 1] by the env."""

    def __init__(self, obs_dim: int, hidden: int = 128, init_log_std: float = -0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)
        self.log_std = nn.Parameter(torch.full((1,), init_log_std))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.net(obs).squeeze(-1)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1).clamp(-1.0, 1.0)


def group_advantages(rewards: torch.Tensor, mode: str, gamma: float = 0.995, eps: float = 1e-8) -> torch.Tensor:
    """rewards: (T, N, G) for N groups of G trajectories. Returns advantages of the same shape."""
    if mode == "outcome":
        R = rewards.sum(0)                                              # (N, G)
        A = (R - R.mean(-1, keepdim=True)) / (R.std(-1, keepdim=True) + eps)
        return A.unsqueeze(0).expand_as(rewards)
    if mode == "process":
        mean = rewards.mean(dim=(0, 2), keepdim=True)
        std = rewards.std(dim=(0, 2), keepdim=True)
        r_norm = (rewards - mean) / (std + eps)
        return r_norm.flip(0).cumsum(0).flip(0)
    if mode == "rtg":
        rtg = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[0])
        for t in range(rewards.shape[0] - 1, -1, -1):
            running = rewards[t] + gamma * running
            rtg[t] = running
        return (rtg - rtg.mean(-1, keepdim=True)) / (rtg.std(-1, keepdim=True) + eps)
    raise ValueError(f"unknown adv_mode {mode}")


class GRPO:
    def __init__(self, config: GRPOConfig, device: torch.device, log_dir: Optional[str] = None):
        self.cfg = config
        self.device = device
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.env = BatchedCraneEnv(device, observe_params=config.observe_params)
        obs_dim = 13 if config.observe_params else 10
        self.policy = GaussianPolicy(obs_dim, config.hidden, config.init_log_std).to(device)
        self.ref_policy = copy.deepcopy(self.policy).requires_grad_(False)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        self.env_steps = 0
        self.log_dir = log_dir
        self.writer = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)

    def level(self, it: int) -> float:
        ramp = max(1, int(self.cfg.curriculum_frac * self.cfg.iterations))
        return float(min(1.0, 0.3 + 0.7 * it / ramp))

    @torch.no_grad()
    def collect(self, level: float):
        cfg = self.cfg
        scenarios = [sample_scenario(self.rng, self.env.max_steps, self.env.dt, level) for _ in range(cfg.n_groups)]
        # Group layout: index b = n * G + g  ->  scenario n, member g
        obs = self.env.reset([sc for sc in scenarios for _ in range(cfg.group_size)])
        T, B = self.env.max_steps, cfg.n_groups * cfg.group_size
        buf_obs = torch.empty((T, B, obs.shape[-1]), device=self.device)
        buf_act = torch.empty((T, B), device=self.device)
        buf_logp = torch.empty((T, B), device=self.device)
        buf_rew = torch.empty((T, B), device=self.device)
        settled = None
        for t in range(T):
            d = self.policy.dist(obs)
            a = d.sample()
            buf_obs[t], buf_act[t], buf_logp[t] = obs, a, d.log_prob(a)
            obs, r, settled = self.env.step(a)
            buf_rew[t] = r
        self.env_steps += T * B
        return buf_obs, buf_act, buf_logp, buf_rew, settled

    def update(self, obs, act, old_logp, adv):
        cfg = self.cfg
        D = obs.shape[-1]
        obs, act, old_logp, adv = obs.reshape(-1, D), act.reshape(-1), old_logp.reshape(-1), adv.reshape(-1)
        n = obs.shape[0]
        stats = {"loss": 0.0, "kl_ref": 0.0, "clip_frac": 0.0}
        n_updates = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                d = self.policy.dist(obs[idx])
                logp = d.log_prob(act[idx])
                ratio = (logp - old_logp[idx]).exp()
                surr = torch.min(ratio * adv[idx], ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[idx])
                with torch.no_grad():
                    ref_logp = self.ref_policy.dist(obs[idx]).log_prob(act[idx])
                log_ref_ratio = ref_logp - logp
                kl = log_ref_ratio.exp() - log_ref_ratio - 1.0           # paper Eq. 4
                loss = -(surr - cfg.beta * kl).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()
                stats["loss"] += loss.item()
                stats["kl_ref"] += kl.mean().item()
                stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()
                n_updates += 1
        return {k: v / n_updates for k, v in stats.items()}

    def train(self, save_path: str, validate_fn=None, validate_every: int = 5):
        cfg = self.cfg
        csv_path = os.path.join(self.log_dir, "progress.csv") if self.log_dir else None
        csv_file = open(csv_path, "w", newline="") if csv_path else None
        writer = csv.writer(csv_file) if csv_file else None
        if writer:
            writer.writerow(["iteration", "env_steps", "wall_time", "train_return", "val_return", "val_success", "level"])
        t0 = time.time()
        best_val = -np.inf
        for it in range(cfg.iterations):
            level = self.level(it)
            obs, act, logp, rew, settled = self.collect(level)
            T, B = rew.shape
            adv = group_advantages(rew.view(T, cfg.n_groups, cfg.group_size), cfg.adv_mode, cfg.gamma).reshape(T, B)
            stats = self.update(obs, act, logp, adv)
            if (it + 1) % cfg.ref_update_every == 0:
                self.ref_policy.load_state_dict(self.policy.state_dict())

            train_return = rew.sum(0).mean().item()
            val_return, val_success = np.nan, np.nan
            if validate_fn is not None and (it % validate_every == 0 or it == cfg.iterations - 1):
                val_return, val_success = validate_fn(lambda o: self.policy.act_deterministic(o))
                if val_return > best_val:
                    best_val = val_return
                    self.save(save_path)
            elapsed = time.time() - t0
            print(f"[GRPO] it {it:4d} steps {self.env_steps / 1e6:7.1f}M  train R {train_return:8.1f}  "
                  f"val R {val_return:8.1f}  succ {val_success:5.2f}  lvl {level:.2f}  "
                  f"std {self.policy.log_std.exp().item():.3f}  kl {stats['kl_ref']:.4f}  {elapsed:6.0f}s", flush=True)
            if writer:
                writer.writerow([it, self.env_steps, elapsed, train_return, val_return, val_success, level])
                csv_file.flush()
            if self.writer:
                self.writer.add_scalar("train/return", train_return, self.env_steps)
                self.writer.add_scalar("train/settled_frac_end", settled.float().mean().item(), self.env_steps)
                self.writer.add_scalar("train/std", self.policy.log_std.exp().item(), self.env_steps)
                for k, v in stats.items():
                    self.writer.add_scalar(f"train/{k}", v, self.env_steps)
                if not np.isnan(val_return):
                    self.writer.add_scalar("eval/return", val_return, self.env_steps)
                    self.writer.add_scalar("eval/success", val_success, self.env_steps)
        if validate_fn is None:
            self.save(save_path)
        if csv_file:
            csv_file.close()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.policy.state_dict(), "config": asdict(self.cfg)}, path)


class GRPOAgent:
    """Deterministic inference wrapper with the common `act(obs) -> action` interface."""
    name = "GRPO"

    def __init__(self, path: str, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        obs_dim = 13 if cfg.get("observe_params", True) else 10
        self.policy = GaussianPolicy(obs_dim, cfg["hidden"]).to(device)
        self.policy.load_state_dict(ckpt["state_dict"])
        self.policy.eval()
        self.device = device
        self.observe_params = cfg.get("observe_params", True)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.policy.act_deterministic(o).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def act_batch(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy.act_deterministic(obs.to(self.device))
