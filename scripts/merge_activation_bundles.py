"""Merge multiple activation bundles into a single bundle."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hackrl.evaluation.activation_collection import (
    ActivationDatasetBundle,
    episode_summary,
    load_bundle,
    save_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge multiple activation bundles into one.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input bundle paths (.pt).",
    )
    parser.add_argument("--output", type=str, required=True, help="Output merged bundle path.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--no-shuffle", action="store_true", help="Keep input order instead of shuffling.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    merged_episodes = []
    feature_dim = None

    for input_path in args.inputs:
        bundle = load_bundle(input_path)
        print(f"\nLoaded {input_path}")
        print(episode_summary(bundle))

        if feature_dim is None:
            feature_dim = bundle.feature_dim
        elif bundle.feature_dim != feature_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {feature_dim}, got {bundle.feature_dim} for {input_path}"
            )

        merged_episodes.extend(bundle.episodes)

    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(merged_episodes)

    merged = ActivationDatasetBundle(episodes=merged_episodes, feature_dim=feature_dim or 0)

    print("\nMerged bundle summary")
    print(episode_summary(merged))

    save_bundle(merged, args.output)


if __name__ == "__main__":
    main()
