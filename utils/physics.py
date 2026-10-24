"""Physics equations and numerical integrator for Cart-Double-Pendulum Crane system."""
import numpy as np


class CraneDynamics:
    """Nonlinear equations of motion for an overhead crane with hook and payload.
    
    Coordinates:
        q = [x, theta1, theta2]
        x: trolley position (m)
        theta1: hook sway angle (rad, 0 is downward vertical)
        theta2: payload sway angle (rad, 0 is downward vertical)
    
    Parameters:
        M: trolley mass (kg)
        m1: hook mass (kg)
        m2: payload mass (kg)
        L1: upper cable length (m)
        L2: lower cable length (m)
        g: gravitational acceleration (m/s^2)
        b_x: trolley viscous friction coefficient (N*s/m)
        b_theta1: hook rotational damping (N*m*s/rad)
        b_theta2: payload rotational damping (N*m*s/rad)
    """

    def __init__(
        self,
        M: float = 5.0,
        m1: float = 1.0,
        m2: float = 2.0,
        L1: float = 1.0,
        L2: float = 0.8,
        g: float = 9.81,
        b_x: float = 0.1,
        b_theta1: float = 0.02,
        b_theta2: float = 0.02,
    ):
        self.M = M
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.g = g
        self.b_x = b_x
        self.b_theta1 = b_theta1
        self.b_theta2 = b_theta2

    def compute_derivatives(self, state: np.ndarray, force: float, disturbance_force: float = 0.0) -> np.ndarray:
        """Calculates state derivative [x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot].
        
        Solves: M_mat(q) * q_ddot = F_vec(q, q_dot)
        """
        x, x_dot, th1, th1_dot, th2, th2_dot = state
        M, m1, m2 = self.M, self.m1, self.m2
        L1, L2, g = self.L1, self.L2, self.g

        # Trigonometric terms
        s1, c1 = np.sin(th1), np.cos(th1)
        s2, c2 = np.sin(th2), np.cos(th2)
        s12, c12 = np.sin(th1 - th2), np.cos(th1 - th2)

        # Symmetric Inertia Matrix M_mat(q)
        M11 = M + m1 + m2
        M12 = (m1 + m2) * L1 * c1
        M13 = m2 * L2 * c2

        M21 = M12
        M22 = (m1 + m2) * (L1 ** 2)
        M23 = m2 * L1 * L2 * c12

        M31 = M13
        M32 = M23
        M33 = m2 * (L2 ** 2)

        M_mat = np.array([
            [M11, M12, M13],
            [M21, M22, M23],
            [M31, M32, M33]
        ], dtype=np.float64)

        # Right-hand side vector (Forces, Coriolis, Gravity, Damping, Disturbances)
        # A horizontal disturbance F_d acting on the payload maps to generalized
        # forces via J^T F with payload x-position x + L1*sin(th1) + L2*sin(th2).
        # 1. Trolley row
        f_x = (
            force
            + disturbance_force
            - self.b_x * x_dot
            + (m1 + m2) * L1 * (th1_dot ** 2) * s1
            + m2 * L2 * (th2_dot ** 2) * s2
        )

        # 2. Hook angle row
        f_th1 = (
            -(m1 + m2) * g * L1 * s1
            - m2 * L1 * L2 * (th2_dot ** 2) * s12
            - self.b_theta1 * th1_dot
            + disturbance_force * L1 * c1
        )

        # 3. Payload angle row (wind/external disturbance acts on payload)
        f_th2 = (
            -m2 * g * L2 * s2
            + m2 * L1 * L2 * (th1_dot ** 2) * s12
            - self.b_theta2 * th2_dot
            + disturbance_force * L2 * c2
        )

        rhs = np.array([f_x, f_th1, f_th2], dtype=np.float64)

        # Acceleration: q_ddot = inv(M_mat) * rhs
        q_ddot = np.linalg.solve(M_mat, rhs)

        return np.array([
            x_dot,
            q_ddot[0],
            th1_dot,
            q_ddot[1],
            th2_dot,
            q_ddot[2]
        ], dtype=np.float64)

    def rk4_step(self, state: np.ndarray, force: float, dt: float, disturbance_force: float = 0.0) -> np.ndarray:
        """Advances state by dt using 4th order Runge-Kutta integration."""
        k1 = self.compute_derivatives(state, force, disturbance_force)
        k2 = self.compute_derivatives(state + 0.5 * dt * k1, force, disturbance_force)
        k3 = self.compute_derivatives(state + 0.5 * dt * k2, force, disturbance_force)
        k4 = self.compute_derivatives(state + dt * k3, force, disturbance_force)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def total_energy(self, state: np.ndarray) -> float:
        """Kinetic + potential energy (potential zero at the pivot height)."""
        x, x_dot, th1, th1_dot, th2, th2_dot = state
        M, m1, m2 = self.M, self.m1, self.m2
        L1, L2, g = self.L1, self.L2, self.g
        v1x = x_dot + L1 * th1_dot * np.cos(th1)
        v1y = L1 * th1_dot * np.sin(th1)
        v2x = v1x + L2 * th2_dot * np.cos(th2)
        v2y = v1y + L2 * th2_dot * np.sin(th2)
        kinetic = 0.5 * M * x_dot ** 2 + 0.5 * m1 * (v1x ** 2 + v1y ** 2) + 0.5 * m2 * (v2x ** 2 + v2y ** 2)
        potential = -m1 * g * L1 * np.cos(th1) - m2 * g * (L1 * np.cos(th1) + L2 * np.cos(th2))
        return float(kinetic + potential)

    def payload_x(self, state: np.ndarray) -> float:
        """Horizontal position of the payload."""
        return float(state[0] + self.L1 * np.sin(state[2]) + self.L2 * np.sin(state[4]))
