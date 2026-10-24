"""Trained-agent loading with a common `act(obs) -> action` interface for every controller."""
import os

import numpy as np
import torch

TRAINED_DIR = os.path.join(os.path.dirname(__file__), "trained")


class SB3Agent:
    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(1)

    def act_batch(self, obs: torch.Tensor) -> torch.Tensor:
        action, _ = self.model.predict(obs.cpu().numpy(), deterministic=True)
        return torch.as_tensor(action, dtype=obs.dtype, device=obs.device).reshape(-1)


def default_path(algo: str, seed: int = 0) -> str:
    ext = "pt" if algo == "grpo" else "zip"
    return os.path.join(TRAINED_DIR, f"{algo}_seed{seed}.{ext}")


def load_agent(algo: str, path: str = None, device: str = "cpu"):
    """Load a trained RL agent ('grpo', 'ppo' or 'sac')."""
    path = path or default_path(algo)
    if algo == "grpo":
        from algos.grpo import GRPOAgent
        return GRPOAgent(path, device)
    if algo == "ppo":
        from stable_baselines3 import PPO
        return SB3Agent(PPO.load(path, device=device), "PPO")
    if algo == "sac":
        from stable_baselines3 import SAC
        return SB3Agent(SAC.load(path, device=device), "SAC")
    raise ValueError(f"unknown algo {algo}")
