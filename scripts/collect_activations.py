"""Collect activations from a frozen checkpoint and save to disk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hackrl.evaluation.activation_collection import (
    collect_activation_dataset,
    episode_summary,
    save_bundle,
    split_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect per-timestep activations from a frozen PPO checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to a single .pt checkpoint file.",
    )
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--min-exploit-cycles", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy argmax actions instead of stochastic sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/activations",
        help="Directory to save the activation bundles.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Save full bundle only, do not split into train/val/test.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)

    print(f"Collecting activations from: {args.checkpoint}")
    print(f"Episodes: {args.num_episodes}, device: {args.device}")

    bundle = collect_activation_dataset(
        args.checkpoint,
        num_episodes=args.num_episodes,
        min_exploit_cycles=args.min_exploit_cycles,
        deterministic=args.deterministic,
        device=args.device,
    )
    print("\n" + episode_summary(bundle))

    # Filter to only honest + hacked (drop neutral for binary classification)
    binary = bundle.filter_by_labels(["honest", "hacked"])
    print(f"\nAfter filtering to binary (honest/hacked): {len(binary.episodes)} episodes")
    print(episode_summary(binary))

    save_bundle(binary, out_dir / "full_bundle.pt")

    if not args.skip_split:
        train, val, test = split_bundle(binary, seed=args.seed)
        save_bundle(train, out_dir / "train.pt")
        save_bundle(val, out_dir / "val.pt")
        save_bundle(test, out_dir / "test.pt")
        print(f"\nSplit (seed={args.seed}):")
        print(f"  Train: {len(train.episodes)}  {train.label_counts()}")
        print(f"  Val:   {len(val.episodes)}  {val.label_counts()}")
        print(f"  Test:  {len(test.episodes)}  {test.label_counts()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
