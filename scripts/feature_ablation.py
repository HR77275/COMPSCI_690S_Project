"""Single-feature causal ablation on top SAE features.

Train an SAE classifier on Gridworld seed0, transfer-evaluate on seed1.
Then for each top SAE feature index, set that latent dimension to zero
across both train and target encodings, re-fit the logistic regression,
and report change in target-test AUROC.

Cumulative ablation: also ablate all top features simultaneously.

Output: artifacts/results/diagnostics/feature_ablation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hackrl.evaluation.activation_collection import load_bundle  # noqa: E402
from hackrl.sae.trainer import load_sae  # noqa: E402


def encode_episodes_with_mask(bundle, sae, ablate_indices: list[int] | None = None):
    """Mean-pool SAE encodings across each episode; optionally zero out a set of feature indices."""
    X, y = [], []
    label_map = {"honest": 0, "hacked": 1}
    for ep in bundle.episodes:
        if ep.label not in label_map:
            continue
        with torch.no_grad():
            codes = sae.encode(ep.activations).cpu().numpy()
        if ablate_indices:
            codes = codes.copy()
            codes[:, ablate_indices] = 0.0
        X.append(codes.mean(axis=0))
        y.append(label_map[ep.label])
    return np.asarray(X), np.asarray(y)


def stratified_bootstrap_auroc(y_true, y_prob, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aurocs = np.empty(n_boot)
    for b in range(n_boot):
        ps = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        ns = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([ps, ns])
        try:
            aurocs[b] = roc_auc_score(y_true[idx], y_prob[idx])
        except ValueError:
            aurocs[b] = np.nan
    aurocs = aurocs[~np.isnan(aurocs)]
    return (
        float(roc_auc_score(y_true, y_prob)),
        float(np.percentile(aurocs, 2.5)),
        float(np.percentile(aurocs, 97.5)),
    )


def fit_eval(train_bundle, test_bundle, sae, ablate_indices=None):
    X_tr, y_tr = encode_episodes_with_mask(train_bundle, sae, ablate_indices)
    X_te, y_te = encode_episodes_with_mask(test_bundle, sae, ablate_indices)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]
    auroc_pt, lo, hi = stratified_bootstrap_auroc(y_te, p_te, seed=42)
    return {
        "auroc": auroc_pt,
        "ci95": [lo, hi],
        "ablated": list(ablate_indices) if ablate_indices else [],
    }


def main():
    out_dir = REPO_ROOT / "artifacts" / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    sae = load_sae("artifacts/sae_sweep/hack_only/sparsity_0p03/sae_trained.pt", device="cpu")

    # Cross-seed transfer = the setting where SAE features actually beat raw,
    # so ablation results matter most here.
    src_train = load_bundle("artifacts/activations/hack_only/train.pt")
    tgt_test = load_bundle("artifacts/activations/hack_only_seed1/test.pt")

    # Top features from the in-domain classifier (matches Section 7.5 of the report).
    top_indices = [941, 730, 686, 253, 851]
    main_direction = {941: "honest", 730: "honest", 686: "hacked", 253: "honest", 851: "hacked"}

    baseline = fit_eval(src_train, tgt_test, sae, ablate_indices=None)
    print(f"Baseline   : AUROC={baseline['auroc']:.4f} CI={baseline['ci95']}")

    per_feature = {}
    for idx in top_indices:
        ablated = fit_eval(src_train, tgt_test, sae, ablate_indices=[idx])
        delta = ablated["auroc"] - baseline["auroc"]
        per_feature[idx] = {
            "auroc": ablated["auroc"],
            "ci95": ablated["ci95"],
            "delta": delta,
            "main_direction": main_direction[idx],
        }
        print(
            f"Ablate {idx:>4d}: AUROC={ablated['auroc']:.4f} CI={ablated['ci95']} "
            f"delta={delta:+.4f} ({main_direction[idx]}-selective)"
        )

    cumulative = fit_eval(src_train, tgt_test, sae, ablate_indices=top_indices)
    print(
        f"Ablate all 5: AUROC={cumulative['auroc']:.4f} CI={cumulative['ci95']} "
        f"delta={cumulative['auroc']-baseline['auroc']:+.4f}"
    )

    payload = {
        "setting": "Gridworld seed0 -> seed1 transfer (target test)",
        "baseline": baseline,
        "single_feature_ablation": per_feature,
        "cumulative_ablation": {
            "ablated": top_indices,
            "auroc": cumulative["auroc"],
            "ci95": cumulative["ci95"],
            "delta": cumulative["auroc"] - baseline["auroc"],
        },
    }

    out_path = out_dir / "feature_ablation.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
