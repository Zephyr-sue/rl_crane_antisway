"""Tests for GRPO advantage estimators, a short GRPO update, and the baseline controllers."""
import numpy as np
import pytest
import torch

from algos.grpo import GRPO, GRPOConfig, group_advantages
from baselines import RBFAdaptiveController, state_from_obs
from baselines.rbf_adaptive import quintic
from envs.double_pendulum_crane_env import DoublePendulumCraneEnv
from envs.scenario import sample_scenario
from utils.evaluation import episode_metrics, rollout


def test_outcome_advantages_are_group_normalized():
    rewards = torch.randn(50, 3, 8)
    adv = group_advantages(rewards, "outcome")
    per_traj = adv[0]
    assert torch.allclose(per_traj.mean(-1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(per_traj.std(-1), torch.ones(3), atol=1e-4)
    assert torch.allclose(adv[0], adv[-1])  # same advantage for every step of a trajectory


def test_process_advantages_are_suffix_sums_of_normalized_rewards():
    rewards = torch.randn(20, 2, 4)
    adv = group_advantages(rewards, "process")
    r = (rewards - rewards.mean(dim=(0, 2), keepdim=True)) / rewards.std(dim=(0, 2), keepdim=True)
    assert torch.allclose(adv[0], r.sum(0), atol=1e-4)
    assert torch.allclose(adv[-1], r[-1], atol=1e-5)


def test_rtg_advantages_zero_mean_per_timestep():
    adv = group_advantages(torch.randn(30, 4, 16), "rtg", gamma=0.99)
    assert torch.allclose(adv.mean(-1), torch.zeros(30, 4), atol=1e-5)


def test_identical_group_members_get_zero_advantage():
    rewards = torch.randn(10, 1, 1).expand(10, 1, 6).contiguous()
    for mode in ("outcome", "rtg"):
        assert torch.allclose(group_advantages(rewards, mode), torch.zeros_like(rewards), atol=1e-4)


def test_grpo_iteration_runs_and_saves(tmp_path):
    cfg = GRPOConfig(n_groups=2, group_size=4, iterations=1, minibatch_size=1024, epochs=1)
    agent = GRPO(cfg, torch.device("cpu"))
    agent.env.max_steps = 20
    agent.train(str(tmp_path / "g.pt"))
    from algos.grpo import GRPOAgent
    act = GRPOAgent(str(tmp_path / "g.pt")).act(np.zeros(13, dtype=np.float32))
    assert act.shape == (1,) and -1 <= act[0] <= 1


def test_state_from_obs_roundtrip():
    env = DoublePendulumCraneEnv()
    obs, _ = env.reset(seed=5)
    s = state_from_obs(obs)
    x, x_dot, th1, th1_dot, th2, th2_dot = env.state
    np.testing.assert_allclose(s, [x - env.target_x, x_dot, th1, th1_dot, th2, th2_dot], atol=1e-5)


def test_quintic_reference_boundary_conditions():
    assert quintic(0.0) == (0.0, 0.0, 0.0)
    p, dp, ddp = quintic(1.0)
    assert p == pytest.approx(1.0) and dp == pytest.approx(0.0, abs=1e-12) and ddp == pytest.approx(0.0, abs=1e-12)


def test_rbf_controller_runs_and_resets():
    ctrl = RBFAdaptiveController()
    env = DoublePendulumCraneEnv()
    m = episode_metrics(rollout(env, ctrl, sample_scenario(np.random.default_rng(1))))
    assert set(m) >= {"success", "settle_time", "max_sway_deg"}
    assert np.abs(ctrl.W).sum() > 0          # online adaptation changed the RBF weights
    ctrl.reset()
    assert np.abs(ctrl.W).sum() == 0 and ctrl.e0 is None


def test_rbf_positions_nominal_crane_without_gusts():
    sc = sample_scenario(np.random.default_rng(0), randomize_params=False, level=0.3)
    sc.gusts[:] = 0
    m = episode_metrics(rollout(DoublePendulumCraneEnv(), RBFAdaptiveController(), sc))
    assert m["final_error_cm"] < 10.0


def test_lqr_batched_matches_single_and_settles_nominal():
    from baselines import LQRController
    ctrl = LQRController()
    env = DoublePendulumCraneEnv()
    rng = np.random.default_rng(4)
    for _ in range(5):
        obs = env.observation_space.sample() * 0.05
        obs[3] = obs[6] = 1.0                    # cos(theta) of a near-vertical state
        single = ctrl.act(obs)
        batched = ctrl.act_batch(torch.as_tensor(obs[None]))
        np.testing.assert_allclose(single, batched.numpy(), atol=1e-5)
    sc = sample_scenario(rng, randomize_params=False, level=0.3)
    sc.gusts[:] = 0
    assert episode_metrics(rollout(env, ctrl, sc))["success"] == 1.0


def test_residual_grpo_starts_at_lqr_prior(tmp_path):
    from algos.grpo import GRPOAgent
    from baselines import LQRController
    cfg = GRPOConfig(n_groups=2, group_size=4, iterations=1, minibatch_size=1024, epochs=1, residual=True)
    agent = GRPO(cfg, torch.device("cpu"))
    agent.env.max_steps = 20
    agent.save(str(tmp_path / "r.pt"))                  # untrained: residual ~ 0
    res = GRPOAgent(str(tmp_path / "r.pt"))
    obs = np.zeros(13, dtype=np.float32)
    obs[0], obs[3], obs[6] = 0.1, 1.0, 1.0
    assert res.name == "Res-GRPO"
    np.testing.assert_allclose(res.act(obs), LQRController().act(obs), atol=0.05)
