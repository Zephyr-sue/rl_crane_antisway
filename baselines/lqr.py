"""LQR controller designed on the linearization about the hanging equilibrium (nominal plant)."""
import numpy as np
from scipy.linalg import solve_continuous_are

from utils.physics import CraneDynamics


def linearize(dynamics: CraneDynamics, eps: float = 1e-6):
    """Numerical Jacobians A = df/ds, B = df/dF at s = 0, F = 0."""
    s0 = np.zeros(6)
    f0 = dynamics.compute_derivatives(s0, 0.0)
    A = np.zeros((6, 6))
    for i in range(6):
        ds = np.zeros(6)
        ds[i] = eps
        A[:, i] = (dynamics.compute_derivatives(s0 + ds, 0.0) - f0) / eps
    B = ((dynamics.compute_derivatives(s0, eps) - f0) / eps)[:, None]
    return A, B


class LQRController:
    name = "LQR"

    def __init__(self, max_force: float = 30.0, Q=None, R: float = 0.02):
        from baselines import state_from_obs
        self._state_from_obs = state_from_obs
        self.max_force = max_force
        A, B = linearize(CraneDynamics())  # nominal m2 = 2.0, L1 = 1.0, L2 = 0.8
        Q = np.diag([20.0, 5.0, 40.0, 1.0, 40.0, 1.0]) if Q is None else Q
        P = solve_continuous_are(A, B, Q, np.array([[R]]))
        self.K = (B.T @ P) / R

    def act(self, obs: np.ndarray) -> np.ndarray:
        s = self._state_from_obs(obs)
        force = float(-(self.K @ s)[0])
        return np.array([np.clip(force / self.max_force, -1.0, 1.0)], dtype=np.float32)

    def act_batch(self, obs):
        """Batched torch version of `act` (same math), used for residual-GRPO rollouts on the GPU."""
        import torch
        from envs.observation import POS_SCALE, VEL_SCALE, RATE_SCALE
        s = torch.stack([
            obs[:, 0] * POS_SCALE,
            obs[:, 1] * VEL_SCALE,
            torch.atan2(obs[:, 2], obs[:, 3]),
            obs[:, 4] * RATE_SCALE,
            torch.atan2(obs[:, 5], obs[:, 6]),
            obs[:, 7] * RATE_SCALE,
        ], dim=-1)
        K = torch.as_tensor(self.K[0], dtype=obs.dtype, device=obs.device)
        return (-(s @ K) / self.max_force).clamp(-1.0, 1.0)
