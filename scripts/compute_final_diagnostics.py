"""Re-compute final-report diagnostics with real probabilities.

Produces:
  artifacts/results/diagnostics/transfer_seed0_to_seed1_roc.png
  artifacts/results/diagnostics/transfer_grid_to_lunar_roc.png
  artifacts/results/diagnostics/transfer_lunar_to_grid_roc.png
  artifacts/results/diagnostics/bootstrap_cis.json

Bootstrap confidence intervals are stratified by class with 2000 resamples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hackrl.classifier.train import (  # noqa: E402
    _aggregate_behavioral_episodes,
    _aggregate_episodes,
)
from hackrl.evaluation.activation_collection import load_bundle  # noqa: E402
from hackrl.sae.trainer import load_sae  # noqa: E402


def stratified_bootstrap_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aurocs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pos_sample = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        neg_sample = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([pos_sample, neg_sample])
        try:
            aurocs[b] = roc_auc_score(y_true[idx], y_prob[idx])
        except ValueError:
            aurocs[b] = np.nan
    aurocs = aurocs[~np.isnan(aurocs)]
    return float(roc_auc_score(y_true, y_prob)), float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def paired_bootstrap_diff(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Return (delta_point, lo, hi, p_two_sided) for AUROC(a) - AUROC(b).

    p_two_sided is the bootstrap two-sided p-value for the null delta = 0.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pos_sample = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        neg_sample = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([pos_sample, neg_sample])
        try:
            diffs[b] = roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx])
        except ValueError:
            diffs[b] = np.nan
    diffs = diffs[~np.isnan(diffs)]
    point = float(roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b))
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    p = 2.0 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return point, lo, hi, float(p)


def fit_and_predict(
    train_bundle,
    target_bundle,
    *,
    sae=None,
    behavioral=False,
):
    if behavioral:
        X_tr, y_tr = _aggregate_behavioral_episodes(train_bundle)
        X_t, y_t = _aggregate_behavioral_episodes(target_bundle)
    else:
        X_tr, y_tr = _aggregate_episodes(train_bundle, sae=sae, aggregation="mean")
        X_t, y_t = _aggregate_episodes(target_bundle, sae=sae, aggregation="mean")
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_t)[:, 1]
    return y_t, prob, clf


def auto_orient(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, str]:
    """Flip probabilities if AUROC < 0.5 so reported curves are non-pathological."""
    auroc = roc_auc_score(y_true, y_prob)
    if auroc < 0.5:
        return 1.0 - y_prob, "flipped"
    return y_prob, "as_is"


def plot_roc(curves: list[tuple[str, np.ndarray, np.ndarray, float, float, float]], out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_true, y_prob, auroc_pt, auroc_lo, auroc_hi in curves:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUROC={auroc_pt:.3f}, 95% CI [{auroc_lo:.3f}, {auroc_hi:.3f}])")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    out_dir = REPO_ROOT / "artifacts" / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_summary: dict = {}

    # Per-task SAE checkpoints (highest sparsity that preserves perfect in-domain).
    sae_grid = load_sae("artifacts/sae_sweep/hack_only/sparsity_0p03/sae_trained.pt", device="cpu")
    sae_lunar = load_sae("artifacts/sae_sweep/lunar_hack_only/sparsity_0p03/sae_trained.pt", device="cpu")
    sae_pooled = load_sae("artifacts/sae/pooled_cross_task/sae_trained.pt", device="cpu")

    # ===== 1. Gridworld seed0 -> seed1 transfer =====
    print("\n[1] Gridworld seed0 -> seed1 transfer")
    src_train = load_bundle("artifacts/activations/hack_only/train.pt")
    tgt_test = load_bundle("artifacts/activations/hack_only_seed1/test.pt")

    y_true, sae_prob, _ = fit_and_predict(src_train, tgt_test, sae=sae_grid)
    sae_prob, sae_dir = auto_orient(y_true, sae_prob)
    _, raw_prob, _ = fit_and_predict(src_train, tgt_test, sae=None)
    raw_prob, raw_dir = auto_orient(y_true, raw_prob)
    _, beh_prob, _ = fit_and_predict(src_train, tgt_test, behavioral=True)
    beh_prob, beh_dir = auto_orient(y_true, beh_prob)

    sae_pt, sae_lo, sae_hi = stratified_bootstrap_auroc(y_true, sae_prob, seed=1)
    raw_pt, raw_lo, raw_hi = stratified_bootstrap_auroc(y_true, raw_prob, seed=2)
    beh_pt, beh_lo, beh_hi = stratified_bootstrap_auroc(y_true, beh_prob, seed=3)
    delta_sae_raw = paired_bootstrap_diff(y_true, sae_prob, raw_prob, seed=4)
    print(f"  SAE       AUROC={sae_pt:.3f} [{sae_lo:.3f}, {sae_hi:.3f}] dir={sae_dir}")
    print(f"  Raw       AUROC={raw_pt:.3f} [{raw_lo:.3f}, {raw_hi:.3f}] dir={raw_dir}")
    print(f"  Beh       AUROC={beh_pt:.3f} [{beh_lo:.3f}, {beh_hi:.3f}] dir={beh_dir}")
    print(f"  delta(SAE-Raw) = {delta_sae_raw[0]:.3f} [{delta_sae_raw[1]:.3f}, {delta_sae_raw[2]:.3f}], p={delta_sae_raw[3]:.3f}")

    plot_roc(
        [
            ("SAE", y_true, sae_prob, sae_pt, sae_lo, sae_hi),
            ("Raw", y_true, raw_prob, raw_pt, raw_lo, raw_hi),
            ("Behavioral", y_true, beh_prob, beh_pt, beh_lo, beh_hi),
        ],
        out_dir / "transfer_seed0_to_seed1_roc.png",
        title="Gridworld seed0 $\\rightarrow$ seed1 transfer (test)",
    )

    bootstrap_summary["grid_seed0_to_seed1"] = {
        "n_total": int(y_true.size),
        "n_pos": int(y_true.sum()),
        "SAE": {"auroc": sae_pt, "ci95": [sae_lo, sae_hi], "direction": sae_dir},
        "Raw": {"auroc": raw_pt, "ci95": [raw_lo, raw_hi], "direction": raw_dir},
        "Behavioral": {"auroc": beh_pt, "ci95": [beh_lo, beh_hi], "direction": beh_dir},
        "SAE_minus_Raw": {
            "delta": delta_sae_raw[0],
            "ci95": [delta_sae_raw[1], delta_sae_raw[2]],
            "p_two_sided": delta_sae_raw[3],
        },
    }

    # ===== 2. Grid -> Lunar (pooled SAE) =====
    print("\n[2] Grid -> Lunar transfer (pooled SAE)")
    src_train = load_bundle("artifacts/activations/hack_only/train.pt")
    tgt_test = load_bundle("artifacts/activations/lunar_hack_only/test.pt")

    y_true, sae_prob, _ = fit_and_predict(src_train, tgt_test, sae=sae_pooled)
    sae_prob, sae_dir = auto_orient(y_true, sae_prob)
    _, raw_prob, _ = fit_and_predict(src_train, tgt_test, sae=None)
    raw_prob, raw_dir = auto_orient(y_true, raw_prob)
    _, beh_prob, _ = fit_and_predict(src_train, tgt_test, behavioral=True)
    beh_prob, beh_dir = auto_orient(y_true, beh_prob)

    sae_pt, sae_lo, sae_hi = stratified_bootstrap_auroc(y_true, sae_prob, seed=11)
    raw_pt, raw_lo, raw_hi = stratified_bootstrap_auroc(y_true, raw_prob, seed=12)
    beh_pt, beh_lo, beh_hi = stratified_bootstrap_auroc(y_true, beh_prob, seed=13)
    delta_sae_raw = paired_bootstrap_diff(y_true, sae_prob, raw_prob, seed=14)
    print(f"  SAE       AUROC={sae_pt:.3f} [{sae_lo:.3f}, {sae_hi:.3f}] dir={sae_dir}")
    print(f"  Raw       AUROC={raw_pt:.3f} [{raw_lo:.3f}, {raw_hi:.3f}] dir={raw_dir}")
    print(f"  Beh       AUROC={beh_pt:.3f} [{beh_lo:.3f}, {beh_hi:.3f}] dir={beh_dir}")
    print(f"  delta(SAE-Raw) = {delta_sae_raw[0]:.3f} [{delta_sae_raw[1]:.3f}, {delta_sae_raw[2]:.3f}], p={delta_sae_raw[3]:.3f}")

    plot_roc(
        [
            ("SAE", y_true, sae_prob, sae_pt, sae_lo, sae_hi),
            ("Raw", y_true, raw_prob, raw_pt, raw_lo, raw_hi),
            ("Behavioral", y_true, beh_prob, beh_pt, beh_lo, beh_hi),
        ],
        out_dir / "transfer_grid_to_lunar_roc.png",
        title="Grid $\\rightarrow$ Lunar transfer (pooled SAE, test)",
    )

    bootstrap_summary["grid_to_lunar"] = {
        "n_total": int(y_true.size),
        "n_pos": int(y_true.sum()),
        "SAE": {"auroc": sae_pt, "ci95": [sae_lo, sae_hi], "direction": sae_dir},
        "Raw": {"auroc": raw_pt, "ci95": [raw_lo, raw_hi], "direction": raw_dir},
        "Behavioral": {"auroc": beh_pt, "ci95": [beh_lo, beh_hi], "direction": beh_dir},
        "SAE_minus_Raw": {
            "delta": delta_sae_raw[0],
            "ci95": [delta_sae_raw[1], delta_sae_raw[2]],
            "p_two_sided": delta_sae_raw[3],
        },
    }

    # ===== 3. Lunar -> Grid (pooled SAE) =====
    print("\n[3] Lunar -> Grid transfer (pooled SAE)")
    src_train = load_bundle("artifacts/activations/lunar_hack_only/train.pt")
    tgt_test = load_bundle("artifacts/activations/hack_only/test.pt")

    y_true, sae_prob, _ = fit_and_predict(src_train, tgt_test, sae=sae_pooled)
    sae_prob, sae_dir = auto_orient(y_true, sae_prob)
    _, raw_prob, _ = fit_and_predict(src_train, tgt_test, sae=None)
    raw_prob, raw_dir = auto_orient(y_true, raw_prob)
    _, beh_prob, _ = fit_and_predict(src_train, tgt_test, behavioral=True)
    beh_prob, beh_dir = auto_orient(y_true, beh_prob)

    sae_pt, sae_lo, sae_hi = stratified_bootstrap_auroc(y_true, sae_prob, seed=21)
    raw_pt, raw_lo, raw_hi = stratified_bootstrap_auroc(y_true, raw_prob, seed=22)
    beh_pt, beh_lo, beh_hi = stratified_bootstrap_auroc(y_true, beh_prob, seed=23)
    delta_sae_raw = paired_bootstrap_diff(y_true, sae_prob, raw_prob, seed=24)
    print(f"  SAE       AUROC={sae_pt:.3f} [{sae_lo:.3f}, {sae_hi:.3f}] dir={sae_dir}")
    print(f"  Raw       AUROC={raw_pt:.3f} [{raw_lo:.3f}, {raw_hi:.3f}] dir={raw_dir}")
    print(f"  Beh       AUROC={beh_pt:.3f} [{beh_lo:.3f}, {beh_hi:.3f}] dir={beh_dir}")
    print(f"  delta(SAE-Raw) = {delta_sae_raw[0]:.3f} [{delta_sae_raw[1]:.3f}, {delta_sae_raw[2]:.3f}], p={delta_sae_raw[3]:.3f}")

    plot_roc(
        [
            ("SAE", y_true, sae_prob, sae_pt, sae_lo, sae_hi),
            ("Raw", y_true, raw_prob, raw_pt, raw_lo, raw_hi),
            ("Behavioral", y_true, beh_prob, beh_pt, beh_lo, beh_hi),
        ],
        out_dir / "transfer_lunar_to_grid_roc.png",
        title="Lunar $\\rightarrow$ Grid transfer (pooled SAE, test)",
    )

    bootstrap_summary["lunar_to_grid"] = {
        "n_total": int(y_true.size),
        "n_pos": int(y_true.sum()),
        "SAE": {"auroc": sae_pt, "ci95": [sae_lo, sae_hi], "direction": sae_dir},
        "Raw": {"auroc": raw_pt, "ci95": [raw_lo, raw_hi], "direction": raw_dir},
        "Behavioral": {"auroc": beh_pt, "ci95": [beh_lo, beh_hi], "direction": beh_dir},
        "SAE_minus_Raw": {
            "delta": delta_sae_raw[0],
            "ci95": [delta_sae_raw[1], delta_sae_raw[2]],
            "p_two_sided": delta_sae_raw[3],
        },
    }

    summary_path = out_dir / "bootstrap_cis.json"
    with open(summary_path, "w") as f:
        json.dump(bootstrap_summary, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
