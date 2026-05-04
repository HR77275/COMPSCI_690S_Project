"""Run a small SAE sparsity sweep and optionally evaluate each checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hackrl.classifier.train import train_and_evaluate_classifier
from hackrl.evaluation.activation_collection import load_bundle
from hackrl.sae.trainer import SAETrainConfig, train_sae


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep SAE L1 sparsity coefficients and record reconstruction/classification metrics."
    )
    parser.add_argument("train_data", type=str, help="Path to train.pt activation bundle.")
    parser.add_argument("--val-data", type=str, default=None, help="Path to val.pt bundle.")
    parser.add_argument("--test-data", type=str, default=None, help="Path to test.pt bundle.")
    parser.add_argument(
        "--sparsity-coefs",
        type=float,
        nargs="+",
        default=[1e-3, 3e-3, 1e-2],
        help="One or more L1 sparsity coefficients to evaluate.",
    )
    parser.add_argument("--dict-multiplier", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/sae_sweep",
        help="Directory to save per-run checkpoints and summary JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading train bundle from: {args.train_data}")
    train_bundle = load_bundle(args.train_data)
    val_bundle = load_bundle(args.val_data) if args.val_data else None
    test_bundle = load_bundle(args.test_data) if args.test_data else None

    all_activations = torch.cat([ep.activations for ep in train_bundle.episodes], dim=0)
    print(f"Flattened activation tensor: {tuple(all_activations.shape)}")

    summary_rows: list[dict[str, object]] = []

    for sparsity_coef in args.sparsity_coefs:
        tag = str(sparsity_coef).replace("-", "m").replace(".", "p")
        run_dir = output_dir / f"sparsity_{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 72)
        print(f"Training SAE with sparsity_coef={sparsity_coef}")
        print("=" * 72)

        config = SAETrainConfig(
            dict_multiplier=args.dict_multiplier,
            sparsity_coef=sparsity_coef,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            checkpoint_dir=str(run_dir),
        )
        sae, history = train_sae(all_activations, config=config)
        final_metrics = history[-1]

        row: dict[str, object] = {
            "sparsity_coef": sparsity_coef,
            "recon_loss": final_metrics["recon_loss"],
            "l1_loss": final_metrics["l1_loss"],
            "l0": final_metrics["l0"],
            "checkpoint_dir": str(run_dir),
        }

        if val_bundle is not None:
            val_result, test_result, _ = train_and_evaluate_classifier(
                train_bundle,
                val_bundle,
                test_bundle,
                sae=sae,
                aggregation=args.aggregation,
                name=f"SAE@{sparsity_coef}",
            )
            row["val_auroc"] = val_result.auroc
            row["val_f1"] = val_result.f1
            row["val_accuracy"] = val_result.accuracy
            if test_result is not None:
                row["test_auroc"] = test_result.auroc
                row["test_f1"] = test_result.f1
                row["test_accuracy"] = test_result.accuracy

        summary_rows.append(row)

    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("\nSaved sweep summary to:", summary_path)
    print("\nSummary:")
    for row in summary_rows:
        summary_line = (
            f"  sparsity={row['sparsity_coef']}: "
            f"recon={row['recon_loss']:.6f}, "
            f"L0={row['l0']:.2f}"
        )
        if "val_auroc" in row:
            summary_line += f", val_AUROC={row['val_auroc']:.4f}"
        if "test_auroc" in row:
            summary_line += f", test_AUROC={row['test_auroc']:.4f}"
        print(summary_line)


if __name__ == "__main__":
    main()
