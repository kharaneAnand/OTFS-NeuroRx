"""Focused tests for the stateless simple environment distance."""

from __future__ import annotations

import unittest

import numpy as np

from models.supervisor.simple_environment_detector import (
    SIMPLE_FEATURE_NAMES,
    SimpleEnvironmentReference,
)


class SimpleEnvironmentDetectorTests(unittest.TestCase):
    def test_reference_score_is_zero_at_training_mean(self) -> None:
        values = np.arange(20, dtype=np.float64).reshape(10, 2)
        reference = SimpleEnvironmentReference.fit(values)
        self.assertAlmostEqual(reference.score(reference.mean), 0.0)

    def test_score_is_stateless_and_increases_for_farther_input(self) -> None:
        values = np.tile(np.asarray([0.0, 1.0]), (10, 1))
        reference = SimpleEnvironmentReference.fit(values + np.random.default_rng(5).normal(0, 0.01, values.shape))
        first = reference.score(np.asarray([0.0, 1.0]))
        second = reference.score(np.asarray([10.0, 11.0]))
        repeat = reference.score(np.asarray([0.0, 1.0]))
        self.assertGreater(second, first)
        self.assertEqual(first, repeat)
        self.assertEqual(reference.feature_names, SIMPLE_FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()