"""Trained-agent loading with the common `act(obs) -> action` interface."""
import os

import numpy as np
import torch

TRAINED_DIR = os.path.join(os.path.dirname(__file__), "trained")


class SB3Agent:
    """Deterministic wrapper around a Stable-Baselines3 model (PPO baseline)."""
    name = "PPO"
    observe_params = True

    def __init__(self, model):
        self.model = model

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(1)

    def act_batch(self, obs: torch.Tensor) -> torch.Tensor:
        action, _ = self.model.predict(obs.cpu().numpy(), deterministic=True)
        return torch.as_tensor(action, dtype=obs.dtype, device=obs.device).reshape(-1)


def default_path(seed: int = 0, algo: str = "grpo") -> str:
    return os.path.join(TRAINED_DIR, f"{algo}_seed{seed}.{'zip' if algo == 'ppo' else 'pt'}")


def load_agent(path: str = None, device: str = "cpu"):
    """Load a trained agent: a GRPO .pt checkpoint or a PPO .zip."""
    path = path or default_path()
    if path.endswith(".zip"):
        from stable_baselines3 import PPO
        return SB3Agent(PPO.load(path, device=device))
    from algos.grpo import GRPOAgent
    return GRPOAgent(path, device)
