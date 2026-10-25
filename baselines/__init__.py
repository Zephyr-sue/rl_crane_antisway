"""Model-based reference controller (RBF adaptive control, Qiang et al. 2021) used as the baseline."""
import numpy as np

from envs.observation import POS_SCALE, VEL_SCALE, RATE_SCALE


def state_from_obs(obs: np.ndarray) -> np.ndarray:
    """Recover [e, x_dot, th1, th1_dot, th2, th2_dot] (e = x - target) from an observation."""
    return np.array([
        obs[0] * POS_SCALE,
        obs[1] * VEL_SCALE,
        np.arctan2(obs[2], obs[3]),
        obs[4] * RATE_SCALE,
        np.arctan2(obs[5], obs[6]),
        obs[7] * RATE_SCALE,
    ])


from baselines.rbf_adaptive import RBFAdaptiveController  # noqa: E402

__all__ = ["state_from_obs", "RBFAdaptiveController"]
