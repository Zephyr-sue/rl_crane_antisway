"""Reward function shared by the single Gymnasium env (numpy) and the batched GPU env (torch).

Only uses operators that behave identically on floats, numpy arrays and torch tensors.
"""

# Weights
W_POS = 1.0        # |e|           : linear pull toward the target (constant gradient -> fast transport)
W_VEL = 0.05       # x_dot^2       : mild trolley speed penalty
W_SWAY = 2.0       # th1^2 + th2^2 : hook and payload sway
W_RATE = 0.1       # th1_dot^2 + th2_dot^2
W_CTRL = 0.01      # u^2           : actuator effort
W_SMOOTH = 0.1     # (u - u_prev)^2: jerk / force-rate smoothness
C_TIME = 0.1       # per-step cost until settled (travel-time objective)
B_SETTLE = 1.0     # per-step bonus while settled
P_LIMIT = 10.0     # penalty for hitting the rail end-stops

# Settling tolerances
SETTLE_POS = 0.05            # m
SETTLE_ANGLE = 0.0349066     # rad (2 deg)
SETTLE_VEL = 0.05            # m/s


def is_settled(e, x_dot, th1, th2):
    return (abs(e) < SETTLE_POS) & (abs(th1) < SETTLE_ANGLE) & (abs(th2) < SETTLE_ANGLE) & (abs(x_dot) < SETTLE_VEL)


def compute_reward(e, x_dot, th1, th1_dot, th2, th2_dot, u, u_prev, hit_limit):
    """Per-step reward. `e` = x - target, `u` = normalized force in [-1, 1].

    r = -[W_POS|e| + W_VEL x_dot^2 + W_SWAY(th1^2+th2^2) + W_RATE(th1_dot^2+th2_dot^2)
          + W_CTRL u^2 + W_SMOOTH (u-u_prev)^2] - C_TIME (1-s) + B_SETTLE s - P_LIMIT * hit_limit
    """
    settled = is_settled(e, x_dot, th1, th2) * 1.0
    cost = (
        W_POS * abs(e)
        + W_VEL * x_dot ** 2
        + W_SWAY * (th1 ** 2 + th2 ** 2)
        + W_RATE * (th1_dot ** 2 + th2_dot ** 2)
        + W_CTRL * u ** 2
        + W_SMOOTH * (u - u_prev) ** 2
    )
    return -cost - C_TIME * (1.0 - settled) + B_SETTLE * settled - P_LIMIT * (hit_limit * 1.0)
