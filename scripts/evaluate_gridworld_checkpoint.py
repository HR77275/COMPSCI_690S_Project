"""Evaluate saved PPO checkpoints on the fixed grid-world."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hackrl.evaluation import evaluate_checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one PPO checkpoint or a directory of checkpoints on the grid-world."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to a single .pt checkpoint file or a directory containing checkpoints.",
    )
    parser.add_argument("--num-trajectories", type=int, default=200)
    parser.add_argument("--min-exploit-cycles", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy argmax actions instead of stochastic sampling.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summaries = evaluate_checkpoint_path(
        args.checkpoint_path,
        num_trajectories=args.num_trajectories,
        min_exploit_cycles=args.min_exploit_cycles,
        deterministic=args.deterministic,
        device=args.device,
    )

    for summary in summaries:
        checkpoint_name = Path(summary.checkpoint_path).name
        update = "-" if summary.update is None else str(summary.update)
        print(f"\nCheckpoint: {checkpoint_name}")
        print(f"Update: {update}")
        print(f"Total trajectories: {summary.total_trajectories}")
        print(f"Honest trajectories: {summary.honest_count} ({summary.honest_pct:.2f}%)")
        print(f"Hacked trajectories: {summary.hacked_count} ({summary.hacked_pct:.2f}%)")
        print(f"Neutral trajectories: {summary.neutral_count} ({summary.neutral_pct:.2f}%)")
        print(f"Mean return: {summary.mean_return:.3f}")
        print(f"Mean exploit cycles: {summary.mean_exploit_cycles:.3f}")

    if len(summaries) > 1:
        total_trajectories = sum(summary.total_trajectories for summary in summaries)
        total_honest = sum(summary.honest_count for summary in summaries)
        total_hacked = sum(summary.hacked_count for summary in summaries)
        total_neutral = sum(summary.neutral_count for summary in summaries)

        print("\nOverall summary across all evaluated checkpoints")
        print(f"Total trajectories: {total_trajectories}")
        print(f"Honest trajectories: {total_honest} ({100.0 * total_honest / total_trajectories:.2f}%)")
        print(f"Hacked trajectories: {total_hacked} ({100.0 * total_hacked / total_trajectories:.2f}%)")
        print(f"Neutral trajectories: {total_neutral} ({100.0 * total_neutral / total_trajectories:.2f}%)")


if __name__ == "__main__":
    main()
