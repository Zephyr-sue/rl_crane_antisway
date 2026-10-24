"""Observation builder shared by the numpy Gymnasium env and the torch batched env.

Pass `xp=numpy` or `xp=torch`; inputs may be scalars/arrays (numpy) or (B,) tensors (torch).
"""
from envs.scenario import NOMINAL

OBS_DIM_WITH_PARAMS = 13
OBS_DIM_NO_PARAMS = 10

# Normalization scales
POS_SCALE = 4.0
VEL_SCALE = 2.0
RATE_SCALE = 3.0


def build_obs(xp, x, x_dot, th1, th1_dot, th2, th2_dot, target, u_prev, m2, L1, L2, observe_params=True):
    """[e, x_dot, sin th1, cos th1, th1_dot, sin th2, cos th2, th2_dot, payload_err, u_prev, (m2, L1, L2)]."""
    e = x - target
    payload_err = x + L1 * xp.sin(th1) + L2 * xp.sin(th2) - target
    feats = [
        e / POS_SCALE,
        x_dot / VEL_SCALE,
        xp.sin(th1),
        xp.cos(th1),
        th1_dot / RATE_SCALE,
        xp.sin(th2),
        xp.cos(th2),
        th2_dot / RATE_SCALE,
        payload_err / POS_SCALE,
        u_prev,
    ]
    if observe_params:
        feats += [
            (m2 - NOMINAL["m2"]) / 1.75,
            (L1 - NOMINAL["L1"]) / 0.4,
            (L2 - NOMINAL["L2"]) / 0.4,
        ]
    return feats
