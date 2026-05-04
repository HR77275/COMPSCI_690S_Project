"""Train on one activation dataset and evaluate transfer on another."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hackrl.classifier.train import (
    _aggregate_behavioral_episodes,
    _aggregate_episodes,
    train_and_evaluate_behavioral_classifier,
    train_and_evaluate_classifier,
)
from hackrl.evaluation.activation_collection import load_bundle
from hackrl.sae.trainer import load_sae


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train classifiers on one dataset and evaluate transfer on another."
    )
    parser.add_argument("--source-train", type=str, required=True, help="Training bundle path.")
    parser.add_argument("--target-val", type=str, required=True, help="Target validation bundle path.")
    parser.add_argument("--target-test", type=str, default=None, help="Target test bundle path.")
    parser.add_argument("--sae-checkpoint", type=str, required=True, help="SAE checkpoint path.")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/results/transfer",
        help="Directory to save transfer results.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading source/target bundles ...")
    source_train = load_bundle(args.source_train)
    target_val = load_bundle(args.target_val)
    target_test = load_bundle(args.target_test) if args.target_test else None
    print(f"  Source train: {len(source_train.episodes)} eps")
    print(f"  Target val:   {len(target_val.episodes)} eps")
    if target_test:
        print(f"  Target test:  {len(target_test.episodes)} eps")

    print(f"\nLoading SAE from: {args.sae_checkpoint}")
    sae = load_sae(args.sae_checkpoint, device=args.device)

    results = {}

    print("\n--- SAE transfer ---")
    sae_val, sae_test, sae_clf = train_and_evaluate_classifier(
        source_train,
        target_val,
        target_test,
        sae=sae,
        aggregation=args.aggregation,
        name="SAE transfer",
    )
    sae_tuned = _tuned_threshold_results_for_representation(
        sae_clf,
        source_train=source_train,
        target_val=target_val,
        target_test=target_test,
        sae=sae,
        aggregation=args.aggregation,
    )
    results["SAE"] = {
        "val": _result_to_dict(sae_val),
        "test": _result_to_dict(sae_test),
        "threshold_tuned": sae_tuned,
    }
    _print_result(sae_val)
    if sae_test:
        _print_result(sae_test)
    _print_threshold_result("SAE", sae_tuned)

    print("\n--- Raw transfer ---")
    raw_val, raw_test, raw_clf = train_and_evaluate_classifier(
        source_train,
        target_val,
        target_test,
        sae=None,
        aggregation=args.aggregation,
        name="Raw transfer",
    )
    raw_tuned = _tuned_threshold_results_for_representation(
        raw_clf,
        source_train=source_train,
        target_val=target_val,
        target_test=target_test,
        sae=None,
        aggregation=args.aggregation,
    )
    results["Raw"] = {
        "val": _result_to_dict(raw_val),
        "test": _result_to_dict(raw_test),
        "threshold_tuned": raw_tuned,
    }
    _print_result(raw_val)
    if raw_test:
        _print_result(raw_test)
    _print_threshold_result("Raw", raw_tuned)

    print("\n--- Behavioral transfer ---")
    beh_val, beh_test, beh_clf = train_and_evaluate_behavioral_classifier(
        source_train,
        target_val,
        target_test,
        name="Behavioral transfer",
    )
    beh_tuned = _tuned_threshold_results_for_behavioral(
        beh_clf,
        target_val=target_val,
        target_test=target_test,
    )
    results["Behavioral"] = {
        "val": _result_to_dict(beh_val),
        "test": _result_to_dict(beh_test),
        "threshold_tuned": beh_tuned,
    }
    _print_result(beh_val)
    if beh_test:
        _print_result(beh_test)
    _print_threshold_result("Behavioral", beh_tuned)

    results["Chance"] = {
        "val": {"auroc": 0.5, "f1": 0.0, "accuracy": 0.5},
        "test": {"auroc": 0.5, "f1": 0.0, "accuracy": 0.5},
    }

    results_path = out_dir / "transfer_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "source_train": args.source_train,
                "target_val": args.target_val,
                "target_test": args.target_test,
                "aggregation": args.aggregation,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved transfer results to {results_path}")


def _result_to_dict(result):
    if result is None:
        return None
    return {
        "name": result.name,
        "auroc": result.auroc,
        "f1": result.f1,
        "accuracy": result.accuracy,
        "top_feature_indices": result.top_feature_indices,
        "top_feature_weights": result.top_feature_weights,
        "top_feature_labels": result.top_feature_labels,
    }


def _print_result(r) -> None:
    print(f"  {r.name}: AUROC={r.auroc:.4f}  F1={r.f1:.4f}  Acc={r.accuracy:.4f}")


def _best_threshold_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    thresholds = _candidate_thresholds(y_prob)
    best = {
        "threshold": 0.5,
        "f1": -1.0,
        "accuracy": 0.0,
    }
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        if (f1, acc) > (best["f1"], best["accuracy"]):
            best = {"threshold": float(thr), "f1": float(f1), "accuracy": float(acc)}
    return best


def _candidate_thresholds(y_prob: np.ndarray) -> np.ndarray:
    """Return all thresholds where binary decisions can change.

    Transfer probabilities can be tightly compressed after domain shift. A fixed
    coarse grid can miss the useful operating region, so we evaluate observed
    probabilities and their midpoints directly.
    """

    unique_probs = np.unique(np.asarray(y_prob, dtype=float))
    if unique_probs.size == 0:
        return np.array([0.5], dtype=float)

    midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
    all_negative_threshold = np.nextafter(unique_probs[-1], np.inf)
    return np.unique(
        np.concatenate(
            (
                np.array([0.0, 1.0, all_negative_threshold], dtype=float),
                unique_probs,
                midpoints,
            )
        )
    )


def _evaluate_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _best_direction_and_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | str]:
    candidates = []
    for direction, probs in (("as_is", y_prob), ("flipped", 1.0 - y_prob)):
        best = _best_threshold_from_probs(y_true, probs)
        best["direction"] = direction
        best["auroc"] = float(roc_auc_score(y_true, probs))
        candidates.append(best)
    return max(candidates, key=lambda item: (item["f1"], item["accuracy"]))


def _tuned_threshold_results_for_representation(
    clf,
    *,
    source_train,
    target_val,
    target_test,
    sae,
    aggregation: str,
) -> dict[str, object]:
    X_val, y_val = _aggregate_episodes(target_val, sae=sae, aggregation=aggregation)
    val_prob = clf.predict_proba(X_val)[:, 1]
    best = _best_direction_and_threshold(y_val, val_prob)
    val_probs = val_prob if best["direction"] == "as_is" else (1.0 - val_prob)

    payload: dict[str, object] = {
        "val": _evaluate_threshold(y_val, val_probs, float(best["threshold"])) | {
            "direction": best["direction"],
            "auroc": float(roc_auc_score(y_val, val_probs)),
        },
    }
    if target_test is not None:
        X_test, y_test = _aggregate_episodes(target_test, sae=sae, aggregation=aggregation)
        test_prob = clf.predict_proba(X_test)[:, 1]
        test_probs = test_prob if best["direction"] == "as_is" else (1.0 - test_prob)
        payload["test"] = _evaluate_threshold(y_test, test_probs, float(best["threshold"])) | {
            "direction": best["direction"],
            "auroc": float(roc_auc_score(y_test, test_probs)),
        }
    return payload


def _tuned_threshold_results_for_behavioral(
    clf,
    *,
    target_val,
    target_test,
) -> dict[str, object]:
    X_val, y_val = _aggregate_behavioral_episodes(target_val)
    val_prob = clf.predict_proba(X_val)[:, 1]
    best = _best_direction_and_threshold(y_val, val_prob)
    val_probs = val_prob if best["direction"] == "as_is" else (1.0 - val_prob)

    payload: dict[str, object] = {
        "val": _evaluate_threshold(y_val, val_probs, float(best["threshold"])) | {
            "direction": best["direction"],
            "auroc": float(roc_auc_score(y_val, val_probs)),
        },
    }
    if target_test is not None:
        X_test, y_test = _aggregate_behavioral_episodes(target_test)
        test_prob = clf.predict_proba(X_test)[:, 1]
        test_probs = test_prob if best["direction"] == "as_is" else (1.0 - test_prob)
        payload["test"] = _evaluate_threshold(y_test, test_probs, float(best["threshold"])) | {
            "direction": best["direction"],
            "auroc": float(roc_auc_score(y_test, test_probs)),
        }
    return payload


def _print_threshold_result(name: str, payload: dict[str, object]) -> None:
    val = payload["val"]
    print(
        f"  {name} tuned threshold on target val: "
        f"dir={val['direction']}  thr={val['threshold']:.3f}  "
        f"AUROC={val['auroc']:.4f}  F1={val['f1']:.4f}  Acc={val['accuracy']:.4f}"
    )
    test = payload.get("test")
    if test is not None:
        print(
            f"  {name} tuned-threshold target test: "
            f"dir={test['direction']}  thr={test['threshold']:.3f}  "
            f"AUROC={test['auroc']:.4f}  F1={test['f1']:.4f}  Acc={test['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
