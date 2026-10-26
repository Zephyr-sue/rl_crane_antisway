"""Train an anti-sway agent: GRPO (ours, GPU-batched, critic-free) or PPO (Stable-Baselines3 baseline).

Examples:
    uv run train.py --iterations 250 --seed 0                                # GRPO main run
    uv run train.py --iterations 120 --adv-mode outcome --tag outcome_       # GRPO ablation
    uv run train.py --residual --iterations 150 --seed 0                     # residual GRPO on top of LQR
    uv run train.py --algo ppo --steps 3000000 --seed 0                      # PPO baseline
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from models import default_path
from utils.validation import Validator


def checkpoint_path(args) -> str:
    """Main runs go to models/trained/; tagged (ablation) runs to models/trained/ablations/."""
    if args.out:
        return args.out
    path = default_path(args.seed, "resgrpo" if getattr(args, "residual", False) else args.algo)
    if args.tag:
        path = os.path.join(os.path.dirname(path), "ablations", args.tag + os.path.basename(path))
    return path


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_grpo(args, device):
    from algos.grpo import GRPO, GRPOConfig
    cfg = GRPOConfig(
        n_groups=args.n_groups, group_size=args.group_size, iterations=args.iterations,
        adv_mode=args.adv_mode, beta=args.beta, lr=args.lr, seed=args.seed,
        observe_params=not args.no_params, residual=args.residual,
        init_log_std=-1.5 if args.residual else -0.5,  # a residual starts near the LQR prior: explore less
    )
    run = f"{'resgrpo' if args.residual else 'grpo'}_{args.tag}seed{args.seed}"
    agent = GRPO(cfg, device, log_dir=os.path.join("runs", run))
    agent.train(checkpoint_path(args), validate_fn=Validator(device, observe_params=cfg.observe_params))


def train_ppo(args, device):
    """PPO with a learned critic, same environment, curriculum and validation set as GRPO."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from envs.double_pendulum_crane_env import DoublePendulumCraneEnv
    from models import SB3Agent

    log_dir = os.path.join("runs", f"ppo_{args.tag}seed{args.seed}")
    save_path = checkpoint_path(args)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    total = args.steps

    class CurriculumValidationCallback(BaseCallback):
        """Ramps scenario difficulty like GRPO and logs validation on the shared scenario set."""

        def __init__(self):
            super().__init__()
            self.validator = Validator(device, observe_params=not args.no_params)
            self.next_eval, self.best, self.t0 = 0, -np.inf, time.time()
            self.csv_file = open(os.path.join(log_dir, "progress.csv"), "w", newline="")
            self.csv = csv.writer(self.csv_file)
            self.csv.writerow(["iteration", "env_steps", "wall_time", "train_return", "val_return", "val_success", "level"])

        def _level(self):
            return float(min(1.0, 0.3 + 0.7 * self.num_timesteps / (0.3 * total)))

        def _on_step(self) -> bool:
            self.training_env.env_method("set_level", self._level())
            if self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_freq
                val_return, val_success = self.validator(SB3Agent(self.model).act_batch)
                if val_return > self.best:
                    self.best = val_return
                    self.model.save(save_path)
                elapsed = time.time() - self.t0
                self.csv.writerow([self.n_calls, self.num_timesteps, elapsed, np.nan, val_return, val_success, self._level()])
                self.csv_file.flush()
                print(f"[PPO] steps {self.num_timesteps / 1e6:6.2f}M  val R {val_return:8.1f}  "
                      f"succ {val_success:5.2f}  lvl {self._level():.2f}  {elapsed:6.0f}s", flush=True)
            return True

        def _on_training_end(self) -> None:
            self.csv_file.close()

    venv = make_vec_env(DoublePendulumCraneEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                        env_kwargs={"observe_params": not args.no_params})
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.995)
    model = PPO(
        "MlpPolicy", venv, learning_rate=args.lr, n_steps=1024, batch_size=4096, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128], activation_fn=torch.nn.Tanh, log_std_init=-0.5),
        tensorboard_log=log_dir, seed=args.seed, device=device, verbose=0,
    )
    model.learn(total_timesteps=total, callback=CurriculumValidationCallback())
    venv.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["grpo", "ppo"], default="grpo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu (PPO defaults to cpu: small MLPs)")
    p.add_argument("--out", default=None, help="checkpoint path (default models/trained/<algo>_seed<s>)")
    p.add_argument("--tag", default="", help="run-name prefix for ablations, e.g. 'outcome_'")
    p.add_argument("--no-params", action="store_true", help="hide m2/L1/L2 from the agent (robustness ablation)")
    p.add_argument("--lr", type=float, default=3e-4)
    # GRPO
    p.add_argument("--iterations", type=int, default=250)
    p.add_argument("--n-groups", type=int, default=64, help="scenarios per iteration")
    p.add_argument("--group-size", type=int, default=32, help="rollouts per scenario (G)")
    p.add_argument("--adv-mode", choices=["outcome", "process", "rtg"], default="rtg")
    p.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    p.add_argument("--residual", action="store_true", help="residual GRPO: learn a correction on top of LQR")
    # PPO
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=12)
    p.add_argument("--eval-freq", type=int, default=50_000)
    args = p.parse_args()

    if args.algo == "grpo":
        device = resolve_device(args.device)
        print(f"Training GRPO on {device}")
        train_grpo(args, device)
    else:
        device = torch.device("cpu") if args.device == "auto" else resolve_device(args.device)
        print(f"Training PPO on {device} with {args.n_envs} envs")
        train_ppo(args, device)


if __name__ == "__main__":
    main()
