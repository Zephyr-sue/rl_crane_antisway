"""Train the GRPO anti-sway agent on the GPU-batched crane simulator.

Examples:
    uv run train.py --iterations 250 --seed 0                      # main run
    uv run train.py --iterations 120 --adv-mode outcome --tag outcome_   # ablation
"""
import argparse
import os

import torch

from algos.grpo import GRPO, GRPOConfig
from models import default_path
from utils.validation import Validator


def checkpoint_path(args) -> str:
    """Main runs go to models/trained/; tagged (ablation) runs to models/trained/ablations/."""
    if args.out:
        return args.out
    path = default_path(args.seed)
    if args.tag:
        path = os.path.join(os.path.dirname(path), "ablations", args.tag + os.path.basename(path))
    return path


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--out", default=None, help="checkpoint path (default models/trained/grpo_seed<s>.pt)")
    p.add_argument("--tag", default="", help="run-name prefix for ablations, e.g. 'outcome_'")
    p.add_argument("--no-params", action="store_true", help="hide m2/L1/L2 from the agent (robustness ablation)")
    p.add_argument("--iterations", type=int, default=250)
    p.add_argument("--n-groups", type=int, default=64, help="scenarios per iteration")
    p.add_argument("--group-size", type=int, default=32, help="rollouts per scenario (G)")
    p.add_argument("--adv-mode", choices=["outcome", "process", "rtg"], default="rtg")
    p.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f"Training GRPO on {device}")
    cfg = GRPOConfig(
        n_groups=args.n_groups, group_size=args.group_size, iterations=args.iterations,
        adv_mode=args.adv_mode, beta=args.beta, lr=args.lr, seed=args.seed,
        observe_params=not args.no_params,
    )
    agent = GRPO(cfg, device, log_dir=os.path.join("runs", f"grpo_{args.tag}seed{args.seed}"))
    agent.train(checkpoint_path(args), validate_fn=Validator(device, observe_params=cfg.observe_params))


if __name__ == "__main__":
    main()
