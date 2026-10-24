"""Test script to verify physics integration and environment execution."""
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from utils.physics import CraneDynamics
from utils.physics_torch import TorchCraneDynamics
from envs.double_pendulum_crane_env import DoublePendulumCraneEnv
from envs.batched_crane import BatchedCraneEnv
from envs.scenario import sample_scenario


def test_crane_dynamics_step():
    dynamics = CraneDynamics()
    state = np.array([0.0, 0.0, np.deg2rad(30.0), 0.0, np.deg2rad(20.0), 0.0])
    next_state = dynamics.rk4_step(state, force=5.0, dt=0.02)
    assert next_state.shape == (6,)
    assert not np.any(np.isnan(next_state))


def test_energy_conservation_without_damping():
    dynamics = CraneDynamics(b_x=0.0, b_theta1=0.0, b_theta2=0.0)
    state = np.array([0.0, 0.5, np.deg2rad(40.0), 0.0, np.deg2rad(-25.0), 1.0])
    e0 = dynamics.total_energy(state)
    for _ in range(int(10.0 / 0.002)):
        state = dynamics.rk4_step(state, force=0.0, dt=0.002)
    drift = abs(dynamics.total_energy(state) - e0) / abs(e0)
    assert drift < 1e-3


def test_disturbance_work_matches_energy_change():
    """A constant payload force F_d does work F_d * (payload displacement): checks J^T F mapping."""
    dynamics = CraneDynamics(b_x=0.0, b_theta1=0.0, b_theta2=0.0)
    state = np.array([0.0, 0.0, 0.2, 0.0, -0.1, 0.0])
    e0, p0 = dynamics.total_energy(state), dynamics.payload_x(state)
    f_d = 4.0
    for _ in range(1000):
        state = dynamics.rk4_step(state, force=0.0, dt=0.001, disturbance_force=f_d)
    work = f_d * (dynamics.payload_x(state) - p0)
    assert dynamics.total_energy(state) - e0 == pytest.approx(work, rel=1e-3, abs=1e-5)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_torch_batched_matches_numpy(device):
    rng = np.random.default_rng(0)
    scenarios = [sample_scenario(rng) for _ in range(4)]
    env = DoublePendulumCraneEnv()
    benv = BatchedCraneEnv(torch.device(device), dtype=torch.float64)
    bobs = benv.reset(scenarios)
    actions = rng.uniform(-1, 1, size=(50, 4))

    for i, sc in enumerate(scenarios):
        obs, _ = env.reset(options={"scenario": sc})
        np.testing.assert_allclose(obs, bobs[i].cpu().numpy(), atol=1e-5)

    rewards_b = []
    benv.reset(scenarios)
    for t in range(50):
        bobs, rew, _ = benv.step(torch.as_tensor(actions[t], device=device))
        rewards_b.append(rew.cpu().numpy())
    for i, sc in enumerate(scenarios):
        env.reset(options={"scenario": sc})
        for t in range(50):
            obs, r, *_ = env.step(actions[t, i:i + 1])
            assert r == pytest.approx(rewards_b[t][i], abs=1e-6)
        np.testing.assert_allclose(env.state, benv.state[i].cpu().numpy(), atol=1e-6)


def test_environment_reset_and_step():
    env = DoublePendulumCraneEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (13,)
    assert not np.any(np.isnan(obs))

    action = np.array([0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (13,)
    assert isinstance(reward, float)
    assert not np.any(np.isnan(next_obs))


def test_env_checker_and_param_ablation():
    check_env(DoublePendulumCraneEnv(), skip_render_check=True)
    env = DoublePendulumCraneEnv(observe_params=False)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (10,)


def test_scenario_replay_is_deterministic():
    env = DoublePendulumCraneEnv()
    sc = sample_scenario(np.random.default_rng(3))
    trajs = []
    for _ in range(2):
        env.reset(options={"scenario": sc})
        trajs.append([env.step(np.array([0.3]))[1] for _ in range(100)])
    assert trajs[0] == trajs[1]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
