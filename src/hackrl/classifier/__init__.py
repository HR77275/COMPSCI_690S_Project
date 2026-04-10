"""Hacking classifiers and baseline comparisons."""

from .train import (
    ClassifierResult,
    run_classifier_comparison,
    train_and_evaluate_classifier,
)

__all__ = [
    "ClassifierResult",
    "run_classifier_comparison",
    "train_and_evaluate_classifier",
]
