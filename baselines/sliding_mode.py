"""Sliding-mode controller with a coupled sliding surface for the double-pendulum crane.

Follows the SMC comparison controller family used in Qiang et al., Front. Robot. AI 8:639734
(2021): the surface combines trolley positioning error with hook and payload sway so that
reaching the surface drives both positioning and anti-sway. A boundary layer (tanh) avoids
chattering.
"""
import numpy as np


class SlidingModeController:
    name = "SMC"

    def __init__(self, lam: float = 1.0, alpha1: float = -1.0, alpha2: float = -1.0,
                 gain: float = 30.0, phi: float = 1.0, max_force: float = 30.0):
        from baselines import state_from_obs
        self._state_from_obs = state_from_obs
        self.lam, self.alpha1, self.alpha2 = lam, alpha1, alpha2
        self.gain, self.phi, self.max_force = gain, phi, max_force

    def act(self, obs: np.ndarray) -> np.ndarray:
        e, x_dot, th1, th1_dot, th2, th2_dot = self._state_from_obs(obs)
        s = x_dot + self.lam * e + self.alpha1 * th1 + self.alpha2 * th2
        force = -self.gain * np.tanh(s / self.phi)
        return np.array([np.clip(force / self.max_force, -1.0, 1.0)], dtype=np.float32)
