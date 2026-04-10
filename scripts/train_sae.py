"""Train a Sparse Autoencoder on collected activations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from hackrl.evaluation.activation_collection import load_bundle
from hackrl.sae.trainer import SAETrainConfig, train_sae


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on activations.")
    parser.add_argument(
        "activation_data",
        type=str,
        help="Path to a train split bundle (.pt file).",
    )
    parser.add_argument("--dict-multiplier", type=int, default=8)
    parser.add_argument("--sparsity-coef", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/sae")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading activations from: {args.activation_data}")
    bundle = load_bundle(args.activation_data)
    print(f"Episodes: {len(bundle.episodes)}, feature_dim: {bundle.feature_dim}")

    # Pool all per-timestep activations into one big tensor
    all_activations = torch.cat([ep.activations for ep in bundle.episodes], dim=0)
    print(f"Total timestep activations: {all_activations.shape}")

    config = SAETrainConfig(
        dict_multiplier=args.dict_multiplier,
        sparsity_coef=args.sparsity_coef,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    sae, history = train_sae(all_activations, config=config)

    print(f"\nFinal metrics: recon={history[-1]['recon_loss']:.6f}, L0={history[-1]['l0']:.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
