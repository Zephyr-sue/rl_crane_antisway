"""Trained-agent loading with the common `act(obs) -> action` interface."""
import os

TRAINED_DIR = os.path.join(os.path.dirname(__file__), "trained")


def default_path(seed: int = 0) -> str:
    return os.path.join(TRAINED_DIR, f"grpo_seed{seed}.pt")


def load_agent(path: str = None, device: str = "cpu"):
    """Load a trained GRPO agent."""
    from algos.grpo import GRPOAgent
    return GRPOAgent(path or default_path(), device)
