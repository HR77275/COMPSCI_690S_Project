"""Train PPO on LunarProgressEnv (hacking or honest regime)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hackrl.training.lunar_ppo_trainer import LunarPPOTrainConfig, train_lunar_ppo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on LunarProgressEnv.")
    parser.add_argument("--total-updates", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints/lunar_ppo")
    parser.add_argument("--log-interval", type=int, default=10)
    # Environment flags
    parser.add_argument("--escalation-power", type=float, default=0.0)
    parser.add_argument("--goal-bonus", type=float, default=100.0)
    parser.add_argument("--progress-reward", type=float, default=1.0)
    parser.add_argument(
        "--no-terminate-on-landing",
        dest="terminate_on_landing",
        action="store_false",
        help="Disable landing termination (hacking regime).",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--zone-reward", type=float, default=0.0,
                        help="Per-step reward for hovering in the altitude zone (hover exploit).")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a checkpoint to warm-start from.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = LunarPPOTrainConfig(
        total_updates=args.total_updates,
        checkpoint_interval=args.checkpoint_interval,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        escalation_power=args.escalation_power,
        goal_bonus=args.goal_bonus,
        progress_reward=args.progress_reward,
        terminate_on_landing=args.terminate_on_landing,
        max_steps=args.max_steps,
        entropy_coef=args.entropy_coef,
        resume_from=args.resume_from,
        zone_reward=args.zone_reward,
    )
    summary = train_lunar_ppo(config=config)
    print(f"\nTraining complete. Final checkpoint: {summary.final_checkpoint}")


if __name__ == "__main__":
    main()
