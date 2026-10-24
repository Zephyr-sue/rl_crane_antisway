"""Train an RL anti-sway agent: GRPO (ours, GPU-batched), PPO or SAC (Stable-Baselines3).

Examples:
    uv run train.py --algo grpo --iterations 300 --seed 0
    uv run train.py --algo ppo --steps 3000000 --seed 0
    uv run train.py --algo sac --steps 300000 --seed 0
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from models import default_path


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_grpo(args, device):
    from algos.grpo import GRPO, GRPOConfig
    from utils.validation import Validator

    cfg = GRPOConfig(
        n_groups=args.n_groups, group_size=args.group_size, iterations=args.iterations,
        adv_mode=args.adv_mode, beta=args.beta, lr=args.lr or 3e-4, seed=args.seed,
        observe_params=not args.no_params,
    )
    log_dir = os.path.join("runs", f"grpo_{args.tag}seed{args.seed}")
    agent = GRPO(cfg, device, log_dir)
    validator = Validator(device, observe_params=cfg.observe_params)
    agent.train(args.out or default_path("grpo", args.seed), validate_fn=validator)


def make_sb3_callback(args, device, save_path, total_steps, log_dir):
    from stable_baselines3.common.callbacks import BaseCallback
    from models import SB3Agent
    from utils.validation import Validator

    class CurriculumValidationCallback(BaseCallback):
        """Ramps scenario difficulty like GRPO and logs validation on the shared scenario set."""

        def __init__(self):
            super().__init__()
            self.validator = Validator(device, observe_params=not args.no_params)
            self.next_eval = 0
            self.best = -np.inf
            self.t0 = time.time()
            os.makedirs(log_dir, exist_ok=True)
            self.csv_file = open(os.path.join(log_dir, "progress.csv"), "w", newline="")
            self.csv = csv.writer(self.csv_file)
            self.csv.writerow(["iteration", "env_steps", "wall_time", "train_return", "val_return", "val_success", "level"])

        def _level(self):
            return float(min(1.0, 0.3 + 0.7 * self.num_timesteps / (0.3 * total_steps)))

        def _on_step(self) -> bool:
            self.training_env.env_method("set_level", self._level())
            if self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_freq
                agent = SB3Agent(self.model, args.algo.upper())
                val_return, val_success = self.validator(agent.act_batch)
                if val_return > self.best:
                    self.best = val_return
                    self.model.save(save_path)
                elapsed = time.time() - self.t0
                self.csv.writerow([self.n_calls, self.num_timesteps, elapsed, np.nan, val_return, val_success, self._level()])
                self.csv_file.flush()
                self.logger.record("eval/return", val_return)
                self.logger.record("eval/success", val_success)
                print(f"[{args.algo.upper()}] steps {self.num_timesteps / 1e6:6.2f}M  val R {val_return:8.1f}  "
                      f"succ {val_success:5.2f}  lvl {self._level():.2f}  {elapsed:6.0f}s", flush=True)
            return True

        def _on_training_end(self) -> None:
            self.csv_file.close()

    return CurriculumValidationCallback()


def train_sb3(args, device):
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from envs.double_pendulum_crane_env import DoublePendulumCraneEnv

    env_kwargs = {"observe_params": not args.no_params}
    log_dir = os.path.join("runs", f"{args.algo}_{args.tag}seed{args.seed}")
    save_path = args.out or default_path(args.algo, args.seed)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if args.algo == "ppo":
        venv = make_vec_env(DoublePendulumCraneEnv, n_envs=args.n_envs, seed=args.seed,
                            vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.995)
        model = PPO(
            "MlpPolicy", venv, learning_rate=args.lr or 3e-4, n_steps=1024, batch_size=4096, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[128, 128], activation_fn=torch.nn.Tanh, log_std_init=-0.5),
            tensorboard_log=log_dir, seed=args.seed, device=device, verbose=0,
        )
    else:
        venv = make_vec_env(DoublePendulumCraneEnv, n_envs=args.n_envs, seed=args.seed,
                            vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
        model = SAC(
            "MlpPolicy", venv, learning_rate=args.lr or 3e-4, buffer_size=1_000_000, batch_size=256,
            gamma=0.99, tau=0.005, learning_starts=10_000, train_freq=1, gradient_steps=args.n_envs,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=log_dir, seed=args.seed, device=device, verbose=0,
        )

    callback = make_sb3_callback(args, device, save_path, args.steps, log_dir)
    model.learn(total_timesteps=args.steps, callback=callback, progress_bar=False)
    venv.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["grpo", "ppo", "sac"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--out", default=None, help="checkpoint path (default models/trained/<algo>_seed<s>)")
    p.add_argument("--tag", default="", help="run-name prefix for logs, e.g. 'outcome_'")
    p.add_argument("--no-params", action="store_true", help="hide m2/L1/L2 from the agent (robustness ablation)")
    p.add_argument("--lr", type=float, default=None)
    # GRPO
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--n-groups", type=int, default=32)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--adv-mode", choices=["outcome", "process", "rtg"], default="rtg")
    p.add_argument("--beta", type=float, default=0.04)
    # SB3
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--eval-freq", type=int, default=50_000)
    args = p.parse_args()

    device = resolve_device(args.device)
    if args.algo == "grpo":
        print(f"Training GRPO on {device}")
        train_grpo(args, device)
    else:
        # Small MLPs: PPO is faster on CPU; SAC's many gradient steps benefit from the GPU
        if args.device == "auto" and args.algo == "ppo":
            device = torch.device("cpu")
        args.n_envs = args.n_envs or (16 if args.algo == "ppo" else 4)
        print(f"Training {args.algo.upper()} on {device} with {args.n_envs} envs")
        train_sb3(args, device)


if __name__ == "__main__":
    main()
