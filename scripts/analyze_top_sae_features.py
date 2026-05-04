"""Analyze top SAE features on a labeled activation bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hackrl.evaluation.activation_collection import load_bundle
from hackrl.sae.trainer import load_sae


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze top SAE features on honest vs hacked episodes."
    )
    parser.add_argument("--bundle", type=str, required=True, help="Path to an activation bundle (.pt).")
    parser.add_argument("--sae-checkpoint", type=str, required=True, help="Path to sae_trained.pt.")
    parser.add_argument(
        "--results-json",
        type=str,
        required=True,
        help="Path to classification_results.json to read top SAE feature indices from.",
    )
    parser.add_argument("--num-features", type=int, default=5, help="How many top SAE features to analyze.")
    parser.add_argument("--num-bins", type=int, default=20, help="Number of normalized time bins for profiles.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/interpretability",
        help="Directory to save plots and JSON summaries.",
    )
    return parser


def _interp_profile(values: np.ndarray, num_bins: int) -> np.ndarray:
    if values.size == 1:
        return np.repeat(values, num_bins)
    src_x = np.linspace(0.0, 1.0, num=values.size)
    dst_x = np.linspace(0.0, 1.0, num=num_bins)
    return np.interp(dst_x, src_x, values)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results_json) as f:
        results_data = json.load(f)
    top_indices = results_data["sae_top_features"]["indices"][: args.num_features]

    bundle = load_bundle(args.bundle)
    sae = load_sae(args.sae_checkpoint, device=args.device)

    label_map = {"honest": [], "hacked": []}
    feature_episode_means: dict[int, dict[str, list[float]]] = {
        idx: {"honest": [], "hacked": []} for idx in top_indices
    }
    feature_profiles: dict[int, dict[str, list[np.ndarray]]] = {
        idx: {"honest": [], "hacked": []} for idx in top_indices
    }
    feature_correlations: dict[int, dict[str, list[float]]] = {
        idx: {
            "total_reward": [],
            "steps_taken": [],
            "progress_events": [],
            "reversal_events": [],
            "exploit_cycle_count": [],
        }
        for idx in top_indices
    }

    for ep in bundle.episodes:
        if ep.label not in label_map:
            continue

        with torch.no_grad():
            codes = sae.encode(ep.activations.to(args.device)).cpu().numpy()

        label_map[ep.label].append(ep)

        for idx in top_indices:
            feature_values = codes[:, idx]
            feature_episode_means[idx][ep.label].append(float(np.mean(feature_values)))
            feature_profiles[idx][ep.label].append(_interp_profile(feature_values, args.num_bins))
            feature_correlations[idx]["total_reward"].append(float(ep.total_reward))
            feature_correlations[idx]["steps_taken"].append(float(ep.steps_taken))
            feature_correlations[idx]["progress_events"].append(float(ep.progress_events))
            feature_correlations[idx]["reversal_events"].append(float(ep.reversal_events))
            feature_correlations[idx]["exploit_cycle_count"].append(float(ep.exploit_cycle_count))

    summary: dict[str, object] = {
        "bundle": args.bundle,
        "sae_checkpoint": args.sae_checkpoint,
        "top_feature_indices": top_indices,
        "class_counts": {label: len(eps) for label, eps in label_map.items()},
        "features": {},
    }

    for idx in top_indices:
        honest_means = np.array(feature_episode_means[idx]["honest"], dtype=np.float32)
        hacked_means = np.array(feature_episode_means[idx]["hacked"], dtype=np.float32)
        pooled = np.concatenate([honest_means, hacked_means]) if honest_means.size and hacked_means.size else np.array([])

        corr_summary = {}
        if pooled.size:
            all_total_reward = np.array(feature_correlations[idx]["total_reward"], dtype=np.float32)
            all_steps_taken = np.array(feature_correlations[idx]["steps_taken"], dtype=np.float32)
            all_progress_events = np.array(feature_correlations[idx]["progress_events"], dtype=np.float32)
            all_reversal_events = np.array(feature_correlations[idx]["reversal_events"], dtype=np.float32)
            all_exploit_cycle_count = np.array(
                feature_correlations[idx]["exploit_cycle_count"], dtype=np.float32
            )
            corr_summary = {
                "total_reward": _safe_corr(pooled, all_total_reward),
                "steps_taken": _safe_corr(pooled, all_steps_taken),
                "progress_events": _safe_corr(pooled, all_progress_events),
                "reversal_events": _safe_corr(pooled, all_reversal_events),
                "exploit_cycle_count": _safe_corr(pooled, all_exploit_cycle_count),
            }

        summary["features"][str(idx)] = {
            "honest_mean": float(honest_means.mean()) if honest_means.size else 0.0,
            "hacked_mean": float(hacked_means.mean()) if hacked_means.size else 0.0,
            "mean_difference_hacked_minus_honest": (
                float(hacked_means.mean() - honest_means.mean())
                if honest_means.size and hacked_means.size
                else 0.0
            ),
            "correlations": corr_summary,
        }

    summary_path = output_dir / "top_sae_feature_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {summary_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    honest_color = "#2F5DA8"
    hacked_color = "#C85C3B"

    # Plot 1: per-class mean activation for each top feature
    labels = [f"feature {idx}" for idx in top_indices]
    honest_vals = [summary["features"][str(idx)]["honest_mean"] for idx in top_indices]
    hacked_vals = [summary["features"][str(idx)]["hacked_mean"] for idx in top_indices]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, honest_vals, width, label="Honest", color=honest_color)
    ax.bar(x + width / 2, hacked_vals, width, label="Hacked", color=hacked_color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("SAE feature")
    ax.set_ylabel("Mean activation (held-out test episodes)")
    ax.set_title("Top SAE Feature Activations by Class")
    ax.legend()
    fig.tight_layout()
    means_path = output_dir / "top_sae_feature_means.png"
    fig.savefig(means_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {means_path}")

    # Plot 2: time-resolved profiles, with x in [0,1] = fractional episode position.
    fig, axes = plt.subplots(len(top_indices), 1, figsize=(8, 3 * len(top_indices)), sharex=True)
    if len(top_indices) == 1:
        axes = [axes]
    bin_centers = (np.arange(args.num_bins) + 0.5) / args.num_bins
    for ax, idx in zip(axes, top_indices):
        honest_profiles = feature_profiles[idx]["honest"]
        hacked_profiles = feature_profiles[idx]["hacked"]
        if honest_profiles:
            ax.plot(
                bin_centers,
                np.mean(np.stack(honest_profiles), axis=0),
                label="Honest",
                color=honest_color,
            )
        if hacked_profiles:
            ax.plot(
                bin_centers,
                np.mean(np.stack(hacked_profiles), axis=0),
                label="Hacked",
                color=hacked_color,
            )
        ax.set_ylabel(f"feature {idx}")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Fractional episode position")
    axes[-1].set_xlim(0.0, 1.0)
    fig.suptitle("Top SAE Feature Profiles Across Episode Time", y=1.02)
    fig.tight_layout()
    profiles_path = output_dir / "top_sae_feature_profiles.png"
    fig.savefig(profiles_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {profiles_path}")


if __name__ == "__main__":
    main()
