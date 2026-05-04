"""Evaluate one LunarLander PPO checkpoint or a directory of checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hackrl.evaluation.activation_collection import episode_summary

from collect_lunar_activations import collect_lunar_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one Lunar checkpoint or every checkpoint in a directory."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to a single .pt checkpoint file or a directory containing checkpoints.",
    )
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--min-exploit-cycles", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    return parser


def _evaluate_one(
    checkpoint_path: Path,
    *,
    num_episodes: int,
    min_exploit_cycles: int,
    deterministic: bool,
    device: str,
) -> dict[str, object]:
    bundle = collect_lunar_dataset(
        checkpoint_path,
        num_episodes=num_episodes,
        min_exploit_cycles=min_exploit_cycles,
        deterministic=deterministic,
        device=device,
    )
    counts = bundle.label_counts()
    total = len(bundle.episodes)
    honest = counts.get("honest", 0)
    hacked = counts.get("hacked", 0)
    neutral = counts.get("neutral", 0)
    return {
        "checkpoint": str(checkpoint_path),
        "total": total,
        "honest": honest,
        "hacked": hacked,
        "neutral": neutral,
        "honest_pct": 100.0 * honest / max(total, 1),
        "hacked_pct": 100.0 * hacked / max(total, 1),
        "neutral_pct": 100.0 * neutral / max(total, 1),
    }


def main() -> None:
    args = build_parser().parse_args()
    path = Path(args.checkpoint_path)

    if path.is_file():
        checkpoints = [path]
    elif path.is_dir():
        checkpoints = sorted(path.glob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files found in directory: {path}")
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    results = []
    for checkpoint in checkpoints:
        print(f"\nEvaluating {checkpoint.name} ...")
        result = _evaluate_one(
            checkpoint,
            num_episodes=args.num_episodes,
            min_exploit_cycles=args.min_exploit_cycles,
            deterministic=args.deterministic,
            device=args.device,
        )
        results.append(result)

        print(f"Checkpoint: {checkpoint.name}")
        print(f"Total episodes: {result['total']}")
        print(f"Honest episodes: {result['honest']} ({result['honest_pct']:.2f}%)")
        print(f"Hacked episodes: {result['hacked']} ({result['hacked_pct']:.2f}%)")
        print(f"Neutral episodes: {result['neutral']} ({result['neutral_pct']:.2f}%)")

    if len(results) > 1:
        print("\nOverall summary across all evaluated checkpoints")
        total = sum(r["total"] for r in results)
        honest = sum(r["honest"] for r in results)
        hacked = sum(r["hacked"] for r in results)
        neutral = sum(r["neutral"] for r in results)
        print(f"Total episodes: {total}")
        print(f"Honest episodes: {honest} ({100.0 * honest / max(total, 1):.2f}%)")
        print(f"Hacked episodes: {hacked} ({100.0 * hacked / max(total, 1):.2f}%)")
        print(f"Neutral episodes: {neutral} ({100.0 * neutral / max(total, 1):.2f}%)")


if __name__ == "__main__":
    main()
