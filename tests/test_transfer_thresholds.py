"""Regression tests for transfer-threshold calibration helpers."""

from __future__ import annotations

import numpy as np
import unittest

from scripts.run_transfer_classification import (
    _best_direction_and_threshold,
    _best_threshold_from_probs,
)


class TransferThresholdTests(unittest.TestCase):
    def test_threshold_search_handles_compressed_probabilities(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0274, 0.0275, 0.0282, 0.0283])

        best = _best_threshold_from_probs(y_true, y_prob)

        self.assertEqual(best["f1"], 1.0)
        self.assertEqual(best["accuracy"], 1.0)

    def test_direction_search_can_flip_domain_shifted_scores(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        # Higher scores belong to the negative class, but only by a tiny margin.
        y_prob = np.array([0.9726, 0.9725, 0.9718, 0.9717])

        best = _best_direction_and_threshold(y_true, y_prob)

        self.assertEqual(best["direction"], "flipped")
        self.assertEqual(best["f1"], 1.0)
        self.assertEqual(best["accuracy"], 1.0)
