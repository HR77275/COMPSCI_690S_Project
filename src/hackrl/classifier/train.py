"""Train hacking classifiers on SAE features, raw activations, and behavioral summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor

from hackrl.evaluation.activation_collection import ActivationDatasetBundle
from hackrl.sae.model import SparseAutoencoder


@dataclass(frozen=True)
class ClassifierResult:
    """Evaluation metrics for one classifier variant."""

    name: str
    auroc: float
    f1: float
    accuracy: float
    top_feature_indices: list[int]  # indices of top-K features by |weight|
    top_feature_weights: list[float]
    top_feature_labels: list[str]


def _aggregate_episodes(
    bundle: ActivationDatasetBundle,
    *,
    sae: SparseAutoencoder | None = None,
    aggregation: str = "mean",
) -> tuple[NDArray, NDArray]:
    """Build (X, y) arrays from episode activations.

    If sae is provided, activations are projected through the SAE encoder first.
    aggregation: 'mean' or 'max' pooling over timesteps.
    """
    X_list = []
    y_list = []
    label_map = {"honest": 0, "hacked": 1}

    for ep in bundle.episodes:
        if ep.label not in label_map:
            continue
        acts = ep.activations  # (T, feature_dim)
        if sae is not None:
            with torch.no_grad():
                acts = sae.encode(acts)  # (T, dict_size)
        if aggregation == "mean":
            vec = acts.mean(dim=0)
        elif aggregation == "max":
            vec = acts.max(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        X_list.append(vec.numpy())
        y_list.append(label_map[ep.label])

    return np.array(X_list), np.array(y_list)


def behavioral_feature_names() -> list[str]:
    """Names of hand-engineered trajectory statistics used by the behavioral baseline."""

    return [
        "total_reward",
        "steps_taken",
        "progress_events",
        "reversal_events",
        "mean_step_reward",
        "max_step_reward",
        "min_step_reward",
        "positive_reward_fraction",
    ]


def _aggregate_behavioral_episodes(bundle: ActivationDatasetBundle) -> tuple[NDArray, NDArray]:
    """Build (X, y) arrays from coarse episode-level statistics.

    Intentionally excludes direct label fields such as `goal_reached` and
    `exploit_cycle_count` to avoid turning the baseline into a tautology.
    """

    X_list = []
    y_list = []
    label_map = {"honest": 0, "hacked": 1}

    for ep in bundle.episodes:
        if ep.label not in label_map:
            continue

        rewards_np = ep.rewards.detach().cpu().numpy()
        if rewards_np.size == 0:
            mean_reward = 0.0
            max_reward = 0.0
            min_reward = 0.0
            positive_reward_fraction = 0.0
        else:
            mean_reward = float(np.mean(rewards_np))
            max_reward = float(np.max(rewards_np))
            min_reward = float(np.min(rewards_np))
            positive_reward_fraction = float(np.mean(rewards_np > 0.0))

        X_list.append(
            [
                float(ep.total_reward),
                float(ep.steps_taken),
                float(ep.progress_events),
                float(ep.reversal_events),
                mean_reward,
                max_reward,
                min_reward,
                positive_reward_fraction,
            ]
        )
        y_list.append(label_map[ep.label])

    return np.array(X_list, dtype=np.float32), np.array(y_list)


def train_and_evaluate_classifier(
    train_bundle: ActivationDatasetBundle,
    val_bundle: ActivationDatasetBundle,
    test_bundle: ActivationDatasetBundle | None = None,
    *,
    sae: SparseAutoencoder | None = None,
    aggregation: str = "mean",
    name: str = "classifier",
    top_k: int = 5,
    C: float = 1.0,
    feature_names: Sequence[str] | None = None,
) -> tuple[ClassifierResult, ClassifierResult | None, LogisticRegression]:
    """Train logistic regression and evaluate on val (and optionally test).

    Returns: (val_result, test_result_or_None, fitted_model)
    """
    X_train, y_train = _aggregate_episodes(train_bundle, sae=sae, aggregation=aggregation)
    X_val, y_val = _aggregate_episodes(val_bundle, sae=sae, aggregation=aggregation)

    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_train, y_train)

    val_result = _evaluate(
        clf,
        X_val,
        y_val,
        name=f"{name} (val)",
        top_k=top_k,
        feature_names=feature_names,
    )

    test_result = None
    if test_bundle is not None:
        X_test, y_test = _aggregate_episodes(test_bundle, sae=sae, aggregation=aggregation)
        test_result = _evaluate(
            clf,
            X_test,
            y_test,
            name=f"{name} (test)",
            top_k=top_k,
            feature_names=feature_names,
        )

    return val_result, test_result, clf


def train_and_evaluate_behavioral_classifier(
    train_bundle: ActivationDatasetBundle,
    val_bundle: ActivationDatasetBundle,
    test_bundle: ActivationDatasetBundle | None = None,
    *,
    name: str = "Behavioral",
    top_k: int = 5,
    C: float = 1.0,
) -> tuple[ClassifierResult, ClassifierResult | None, LogisticRegression]:
    """Train logistic regression on hand-engineered trajectory statistics."""

    X_train, y_train = _aggregate_behavioral_episodes(train_bundle)
    X_val, y_val = _aggregate_behavioral_episodes(val_bundle)

    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_train, y_train)

    feature_names = behavioral_feature_names()
    val_result = _evaluate(
        clf,
        X_val,
        y_val,
        name=f"{name} (val)",
        top_k=top_k,
        feature_names=feature_names,
    )

    test_result = None
    if test_bundle is not None:
        X_test, y_test = _aggregate_behavioral_episodes(test_bundle)
        test_result = _evaluate(
            clf,
            X_test,
            y_test,
            name=f"{name} (test)",
            top_k=top_k,
            feature_names=feature_names,
        )

    return val_result, test_result, clf


def _evaluate(
    clf: LogisticRegression,
    X: NDArray,
    y: NDArray,
    *,
    name: str,
    top_k: int,
    feature_names: Sequence[str] | None = None,
) -> ClassifierResult:
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    auroc = roc_auc_score(y, y_prob)
    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)

    # Top features by absolute weight
    weights = clf.coef_[0]
    top_indices = np.argsort(np.abs(weights))[::-1][:top_k].tolist()
    top_weights = [float(weights[i]) for i in top_indices]
    if feature_names is None:
        top_labels = [str(i) for i in top_indices]
    else:
        top_labels = [str(feature_names[i]) for i in top_indices]

    return ClassifierResult(
        name=name,
        auroc=auroc,
        f1=f1,
        accuracy=acc,
        top_feature_indices=top_indices,
        top_feature_weights=top_weights,
        top_feature_labels=top_labels,
    )


def run_classifier_comparison(
    train_bundle: ActivationDatasetBundle,
    val_bundle: ActivationDatasetBundle,
    test_bundle: ActivationDatasetBundle | None,
    *,
    sae: SparseAutoencoder,
    aggregation: str = "mean",
) -> dict[str, tuple[ClassifierResult, ClassifierResult | None]]:
    """Run SAE classifier, raw/behavioral baselines, and chance.

    Returns dict mapping method name to (val_result, test_result_or_None).
    """
    results: dict[str, tuple[ClassifierResult, ClassifierResult | None]] = {}

    # 1. SAE-based classifier
    print("\n--- SAE-based classifier ---")
    sae_val, sae_test, sae_clf = train_and_evaluate_classifier(
        train_bundle, val_bundle, test_bundle,
        sae=sae, aggregation=aggregation, name="SAE",
    )
    results["SAE"] = (sae_val, sae_test)
    _print_result(sae_val)
    if sae_test:
        _print_result(sae_test)

    # 2. Raw-activation baseline
    print("\n--- Raw-activation baseline ---")
    raw_val, raw_test, raw_clf = train_and_evaluate_classifier(
        train_bundle, val_bundle, test_bundle,
        sae=None, aggregation=aggregation, name="Raw",
    )
    results["Raw"] = (raw_val, raw_test)
    _print_result(raw_val)
    if raw_test:
        _print_result(raw_test)

    # 3. Behavioral baseline
    print("\n--- Behavioral baseline ---")
    behavior_val, behavior_test, _ = train_and_evaluate_behavioral_classifier(
        train_bundle,
        val_bundle,
        test_bundle,
        name="Behavioral",
    )
    results["Behavioral"] = (behavior_val, behavior_test)
    _print_result(behavior_val)
    if behavior_test:
        _print_result(behavior_test)

    # 4. Chance baseline
    print("\n--- Chance baseline ---")
    chance = ClassifierResult(
        name="Chance", auroc=0.50, f1=0.0, accuracy=0.50,
        top_feature_indices=[], top_feature_weights=[], top_feature_labels=[],
    )
    results["Chance"] = (chance, chance)
    _print_result(chance)

    return results


def _print_result(r: ClassifierResult) -> None:
    print(f"  {r.name}: AUROC={r.auroc:.4f}  F1={r.f1:.4f}  Acc={r.accuracy:.4f}")
