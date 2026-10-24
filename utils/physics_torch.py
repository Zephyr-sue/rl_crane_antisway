"""Batched (GPU-capable) PyTorch version of the cart-double-pendulum crane dynamics.

Mirrors utils.physics.CraneDynamics equation-for-equation, but operates on a batch of
B cranes at once so thousands of rollouts can be simulated in parallel on CUDA.
"""
import torch


class TorchCraneDynamics:
    """Batched nonlinear crane dynamics.

    State tensor layout (B, 6): [x, x_dot, theta1, theta1_dot, theta2, theta2_dot].
    Per-env parameters m2, L1, L2 are (B,) tensors; the rest are shared scalars.
    """

    def __init__(
        self,
        M: float = 5.0,
        m1: float = 1.0,
        g: float = 9.81,
        b_x: float = 0.1,
        b_theta1: float = 0.02,
        b_theta2: float = 0.02,
    ):
        self.M = M
        self.m1 = m1
        self.g = g
        self.b_x = b_x
        self.b_theta1 = b_theta1
        self.b_theta2 = b_theta2

    def compute_derivatives(
        self,
        state: torch.Tensor,
        force: torch.Tensor,
        disturbance_force: torch.Tensor,
        m2: torch.Tensor,
        L1: torch.Tensor,
        L2: torch.Tensor,
    ) -> torch.Tensor:
        x_dot, th1, th1_dot, th2, th2_dot = state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5]
        M, m1, g = self.M, self.m1, self.g

        s1, c1 = torch.sin(th1), torch.cos(th1)
        s2, c2 = torch.sin(th2), torch.cos(th2)
        s12, c12 = torch.sin(th1 - th2), torch.cos(th1 - th2)

        M11 = torch.full_like(m2, M + m1) + m2
        M12 = (m1 + m2) * L1 * c1
        M13 = m2 * L2 * c2
        M22 = (m1 + m2) * L1 ** 2
        M23 = m2 * L1 * L2 * c12
        M33 = m2 * L2 ** 2

        M_mat = torch.stack([
            torch.stack([M11, M12, M13], dim=-1),
            torch.stack([M12, M22, M23], dim=-1),
            torch.stack([M13, M23, M33], dim=-1),
        ], dim=-2)

        f_x = (
            force
            + disturbance_force
            - self.b_x * x_dot
            + (m1 + m2) * L1 * th1_dot ** 2 * s1
            + m2 * L2 * th2_dot ** 2 * s2
        )
        f_th1 = (
            -(m1 + m2) * g * L1 * s1
            - m2 * L1 * L2 * th2_dot ** 2 * s12
            - self.b_theta1 * th1_dot
            + disturbance_force * L1 * c1
        )
        f_th2 = (
            -m2 * g * L2 * s2
            + m2 * L1 * L2 * th1_dot ** 2 * s12
            - self.b_theta2 * th2_dot
            + disturbance_force * L2 * c2
        )
        rhs = torch.stack([f_x, f_th1, f_th2], dim=-1).unsqueeze(-1)
        q_ddot = torch.linalg.solve(M_mat, rhs).squeeze(-1)

        return torch.stack([x_dot, q_ddot[:, 0], th1_dot, q_ddot[:, 1], th2_dot, q_ddot[:, 2]], dim=-1)

    def rk4_step(
        self,
        state: torch.Tensor,
        force: torch.Tensor,
        dt: float,
        disturbance_force: torch.Tensor,
        m2: torch.Tensor,
        L1: torch.Tensor,
        L2: torch.Tensor,
    ) -> torch.Tensor:
        args = (force, disturbance_force, m2, L1, L2)
        k1 = self.compute_derivatives(state, *args)
        k2 = self.compute_derivatives(state + 0.5 * dt * k1, *args)
        k3 = self.compute_derivatives(state + 0.5 * dt * k2, *args)
        k4 = self.compute_derivatives(state + dt * k3, *args)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
