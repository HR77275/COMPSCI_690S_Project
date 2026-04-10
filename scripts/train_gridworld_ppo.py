"""Convenience runner for PPO training on the fixed grid-world."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hackrl.training import PPOTrainConfig, train_gridworld_ppo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on the hackrl fixed grid-world.")
    parser.add_argument("--total-updates", type=int, default=300)
    parser.add_argument("--checkpoint-interval", type=int, default=15)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints/gridworld_ppo_run1",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--goal-bonus", type=float, default=4.0)
    parser.add_argument("--escalation-power", type=float, default=0.0,
                        help="0=flat reward, 2=quadratic escalation (enables hacking)")
    parser.add_argument("--no-terminate-on-goal", action="store_true",
                        help="Episode continues after box reaches goal (faulty termination)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PPOTrainConfig(
        total_updates=args.total_updates,
        checkpoint_interval=args.checkpoint_interval,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        goal_bonus=args.goal_bonus,
        escalation_power=args.escalation_power,
        terminate_on_goal=not args.no_terminate_on_goal,
        resume_from=args.resume_from,
    )
    summary = train_gridworld_ppo(config=config)

    print("\nTraining complete")
    print(f"Total updates: {summary.total_updates}")
    print(f"Total env steps: {summary.total_env_steps}")
    print(f"Checkpoint interval: {summary.checkpoint_interval}")
    print(f"Checkpoint directory: {summary.checkpoint_dir}")
    print(f"Final checkpoint: {summary.final_checkpoint}")


if __name__ == "__main__":
    main()
