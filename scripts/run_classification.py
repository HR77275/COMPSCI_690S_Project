"""Run SAE projection, classification, baselines, and light interpretability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hackrl.classifier.train import run_classifier_comparison
from hackrl.evaluation.activation_collection import load_bundle
from hackrl.sae.trainer import load_sae


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify hacking episodes using SAE features and raw baselines."
    )
    parser.add_argument("--train-data", type=str, required=True, help="Path to train.pt")
    parser.add_argument("--val-data", type=str, required=True, help="Path to val.pt")
    parser.add_argument("--test-data", type=str, default=None, help="Path to test.pt (optional)")
    parser.add_argument("--sae-checkpoint", type=str, required=True, help="Path to sae_trained.pt")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/results",
        help="Directory to save results JSON and figures.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train_bundle = load_bundle(args.train_data)
    val_bundle = load_bundle(args.val_data)
    test_bundle = load_bundle(args.test_data) if args.test_data else None

    print(f"  Train: {len(train_bundle.episodes)} eps")
    print(f"  Val:   {len(val_bundle.episodes)} eps")
    if test_bundle:
        print(f"  Test:  {len(test_bundle.episodes)} eps")

    print(f"\nLoading SAE from: {args.sae_checkpoint}")
    sae = load_sae(args.sae_checkpoint, device=args.device)
    print(f"  SAE: input_dim={sae.input_dim}, dict_size={sae.dict_size}")

    # ---- Run comparison ----
    results = run_classifier_comparison(
        train_bundle, val_bundle, test_bundle,
        sae=sae, aggregation=args.aggregation,
    )

    # ---- Build comparison table ----
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Method':<20} {'Split':<8} {'AUROC':>8} {'F1':>8} {'Acc':>8}")
    print("-" * 70)

    table_rows = []
    for method, (val_r, test_r) in results.items():
        print(f"{method:<20} {'val':<8} {val_r.auroc:>8.4f} {val_r.f1:>8.4f} {val_r.accuracy:>8.4f}")
        table_rows.append({
            "method": method, "split": "val",
            "auroc": round(val_r.auroc, 4),
            "f1": round(val_r.f1, 4),
            "accuracy": round(val_r.accuracy, 4),
        })
        if test_r and method != "Chance":
            print(f"{method:<20} {'test':<8} {test_r.auroc:>8.4f} {test_r.f1:>8.4f} {test_r.accuracy:>8.4f}")
            table_rows.append({
                "method": method, "split": "test",
                "auroc": round(test_r.auroc, 4),
                "f1": round(test_r.f1, 4),
                "accuracy": round(test_r.accuracy, 4),
            })

    # ---- Light interpretability (D.1–D.2) ----
    sae_val_result = results["SAE"][0]
    print("\n" + "=" * 70)
    print("TOP PREDICTIVE SAE FEATURES (by |classifier weight|)")
    print("=" * 70)
    for rank, (idx, w) in enumerate(
        zip(sae_val_result.top_feature_indices, sae_val_result.top_feature_weights), 1
    ):
        direction = "hacked" if w > 0 else "honest"
        print(f"  #{rank}: SAE feature {idx:>4d}  weight={w:+.4f}  (predicts {direction})")

    behavioral_val_result = results["Behavioral"][0]
    print("\n" + "=" * 70)
    print("TOP BEHAVIORAL FEATURES (by |classifier weight|)")
    print("=" * 70)
    for rank, (label, w) in enumerate(
        zip(behavioral_val_result.top_feature_labels, behavioral_val_result.top_feature_weights),
        1,
    ):
        direction = "hacked" if w > 0 else "honest"
        print(f"  #{rank}: {label:<24} weight={w:+.4f}  (predicts {direction})")

    # ---- Save results ----
    results_path = out_dir / "classification_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "aggregation": args.aggregation,
                "table": table_rows,
                "sae_top_features": {
                    "indices": sae_val_result.top_feature_indices,
                    "weights": sae_val_result.top_feature_weights,
                },
                "behavioral_top_features": {
                    "labels": behavioral_val_result.top_feature_labels,
                    "weights": behavioral_val_result.top_feature_weights,
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")

    # ---- Generate figures ----
    _plot_roc_curves(results, out_dir)
    _plot_metric_bars(results, out_dir)
    print("Done.")


def _plot_roc_curves(
    results: dict,
    out_dir: Path,
) -> None:
    """Plot AUROC comparison points for all learned baselines on the val split.

    Note: we only have summary metrics in ``results`` here, not raw probabilities,
    so this routine is no longer used by the main classification pipeline; the
    in-domain numbers are saturated and a real ROC needs ``y_prob`` arrays to be
    informative. See ``scripts/plot_in_domain_roc.py`` for a curve plot that
    re-fits the classifier and uses real probabilities.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping ROC plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for method in ("SAE", "Raw", "Behavioral"):
        val_r = results[method][0]
        ax.scatter(
            [1 - val_r.accuracy],
            [val_r.accuracy],
            label=f"{method} (AUROC={val_r.auroc:.3f})",
            s=100,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("Classifier Comparison (Validation)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = out_dir / "roc_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC plot: {path}")


def _plot_metric_bars(
    results: dict,
    out_dir: Path,
) -> None:
    """Bar chart of AUROC / F1 / Acc for all methods on val split."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping bar chart.")
        return

    methods = []
    aurocs = []
    f1s = []
    accs = []
    for method in ("SAE", "Raw", "Behavioral", "Chance"):
        val_r = results[method][0]
        methods.append(method)
        aurocs.append(val_r.auroc)
        f1s.append(val_r.f1)
        accs.append(val_r.accuracy)

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width, aurocs, width, label="AUROC")
    ax.bar(x, f1s, width, label="F1")
    ax.bar(x + width, accs, width, label="Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title("SAE vs Raw vs Behavioral vs Chance (Validation)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = out_dir / "metric_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved metric bar chart: {path}")


if __name__ == "__main__":
    main()
