"""Test script to verify physics integration and environment execution."""
import numpy as np
from utils.physics import CraneDynamics
from envs.double_pendulum_crane_env import DoublePendulumCraneEnv


def test_crane_dynamics_step():
    dynamics = CraneDynamics()
    state = np.array([0.0, 0.0, np.deg2rad(30.0), 0.0, np.deg2rad(20.0), 0.0])
    next_state = dynamics.rk4_step(state, force=5.0, dt=0.02)
    assert next_state.shape == (6,)
    assert not np.any(np.isnan(next_state))


def test_environment_reset_and_step():
    env = DoublePendulumCraneEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (12,)
    assert not np.any(np.isnan(obs))

    action = np.array([0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (12,)
    assert isinstance(reward, float)
    assert not np.any(np.isnan(next_obs))


if __name__ == "__main__":
    test_crane_dynamics_step()
    test_environment_reset_and_step()
    print("All physics and environment verification tests passed!")
