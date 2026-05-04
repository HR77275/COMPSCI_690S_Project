"""Split a single-policy activation bundle into train/val/test.

Both honest and hacked episodes here come from the same policy weights, making
classification meaningful (not trivially separable by network identity).
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hackrl.evaluation.activation_collection import load_bundle, save_bundle, split_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a single-policy honest/hacked bundle into train/val/test."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "artifacts/activations/hack_raw/full_bundle.pt"),
        help="Path to the input full_bundle.pt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "artifacts/activations/hack_only"),
        help="Directory to write train/val/test bundles.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    bundle = load_bundle(str(input_path))
    counts = bundle.label_counts()
    print(f"Loaded bundle: {len(bundle.episodes)} episodes — {counts}")

    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = split_bundle(bundle, seed=args.seed)
    save_bundle(train, str(output_dir / "train.pt"))
    save_bundle(val, str(output_dir / "val.pt"))
    save_bundle(test, str(output_dir / "test.pt"))

    print(f"Train: {len(train.episodes)}  {train.label_counts()}")
    print(f"Val:   {len(val.episodes)}  {val.label_counts()}")
    print(f"Test:  {len(test.episodes)}  {test.label_counts()}")
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    main()
