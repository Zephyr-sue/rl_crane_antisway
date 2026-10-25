"""RBF neural-network-compensated adaptive controller of Qiang et al. (Front. Robot. AI 8:639734, 2021).

Control law (paper Eq. 15):
    tau = M0(q) (qdd_d - kv e_dot - kp e) + C0(q, q_dot) q_dot + G0(q) - f_hat
with e = q - q_d, nominal model (M0, C0, G0) and an online RBF compensator (Eqs. 5-7):
    h_i = exp(-||z - c_i||^2 / (2 b^2)),   z = [e, e_dot]
    f_hat = W^T h,                        W_dot = gamma * h z^T P B
where P solves P A + A^T P = -Q for A = [[0, I], [-kp, -kv]] and B = [0; M0(q)^-1].

The crane is underactuated (tau = [F, 0, 0]), so only the trolley row of tau is applied;
the sway errors still act on F through the off-diagonal inertia terms of M0(q).
The paper requires kp, kv > 0 for its Lyapunov argument on the fully actuated error dynamics.
With only the trolley row available, positive sway gains do not damp the sway (grid search:
12% success), so the sway gains were re-tuned by grid search; negative kp on the angles makes
the trolley move toward the swing, the classic anti-sway action (50% success).
The reference q_d is a smooth quintic transfer of the trolley to the target with zero
desired sway (theta_d = 0).
"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from utils.physics import CraneDynamics


def quintic(s: float):
    """Normalized minimum-jerk profile p(s), p'(s), p''(s) for s in [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    p = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    dp = 30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4
    ddp = 60 * s - 180 * s ** 2 + 120 * s ** 3
    return p, dp, ddp


class RBFAdaptiveController:
    name = "RBF"

    def __init__(
        self,
        kp=(5.0, -22.0, -22.0),
        kv=(6.0, 0.5, 0.5),
        transfer_time_per_m: float = 1.2,
        min_transfer_time: float = 2.0,
        n_hidden: int = 11,
        width: float = 1.5,
        gamma: float = 5.0,
        max_force: float = 30.0,
        dt: float = 0.02,
    ):
        from baselines import state_from_obs
        self._state_from_obs = state_from_obs
        self.model = CraneDynamics()  # nominal model M0, C0, G0 (m2 = 2.0, L1 = 1.0, L2 = 0.8)
        self.kp, self.kv = np.diag(kp), np.diag(kv)
        self.transfer_time_per_m = transfer_time_per_m
        self.min_transfer_time = min_transfer_time
        self.gamma, self.max_force, self.dt = gamma, max_force, dt

        A = np.block([[np.zeros((3, 3)), np.eye(3)], [-self.kp, -self.kv]])
        self.P = solve_continuous_lyapunov(A.T, -np.eye(6))   # P A + A^T P = -Q, Q = I
        self.centers = np.linspace(-1.0, 1.0, n_hidden)[:, None] * np.ones((1, 6))
        self.width = width
        self.reset()

    def reset(self):
        self.W = np.zeros((self.centers.shape[0], 3))
        self.t = 0.0
        self.e0 = None

    def reference(self):
        """Desired [e, theta1, theta2] trajectory: trolley error e0 -> 0, zero sway."""
        T = max(self.min_transfer_time, self.transfer_time_per_m * abs(self.e0))
        p, dp, ddp = quintic(self.t / T)
        e_d = self.e0 * (1.0 - p)
        return (np.array([e_d, 0.0, 0.0]),
                np.array([-self.e0 * dp / T, 0.0, 0.0]),
                np.array([-self.e0 * ddp / T ** 2, 0.0, 0.0]))

    def act(self, obs: np.ndarray) -> np.ndarray:
        e_x, x_dot, th1, th1_dot, th2, th2_dot = self._state_from_obs(obs)
        if self.e0 is None:
            self.e0 = e_x
        q = np.array([e_x, th1, th2])            # trolley coordinate measured relative to the target
        q_dot = np.array([x_dot, th1_dot, th2_dot])
        q_d, qd_d, qdd_d = self.reference()
        e, e_dot = q - q_d, q_dot - qd_d

        state = np.array([e_x, x_dot, th1, th1_dot, th2, th2_dot])
        M0 = self.model.mass_matrix(state)
        z = np.concatenate([e, e_dot])
        h = np.exp(-np.sum((z[None, :] - self.centers) ** 2, axis=1) / (2 * self.width ** 2))
        f_hat = self.W.T @ h

        tau = M0 @ (qdd_d - self.kv @ e_dot - self.kp @ e) + self.model.bias_forces(state) - f_hat

        # Adaptation law (paper Eq. 7), integrated with forward Euler
        B = np.vstack([np.zeros((3, 3)), np.linalg.inv(M0)])
        self.W += self.dt * self.gamma * np.outer(h, z @ self.P @ B)
        self.t += self.dt

        return np.array([np.clip(tau[0] / self.max_force, -1.0, 1.0)], dtype=np.float32)
