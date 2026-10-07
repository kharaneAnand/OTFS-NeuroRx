"""Tests for the root pipeline's public output conventions."""

from __future__ import annotations

import unittest

import numpy as np

from run_pipeline import hard_decisions, metric_result


class PipelineOutputTests(unittest.TestCase):
    def test_qpsk_hard_decisions_use_nonnegative_real_and_imaginary_signs(self) -> None:
        estimates = np.asarray([-1 - 1j, 1 - 1j, -1 + 1j, 0 + 0j])
        self.assertEqual(
            hard_decisions(estimates),
            [[0, 0], [1, 0], [0, 1], [1, 1]],
        )

    def test_metrics_are_null_without_ground_truth(self) -> None:
        result = metric_result(np.asarray([1 + 1j]), None)
        self.assertEqual(result["status"], "unavailable_no_ground_truth")
        self.assertIsNone(result["ber"])
        self.assertIsNone(result["ser"])
        self.assertIsNone(result["nmse"])

    def test_metrics_are_marked_reference_only_with_ground_truth(self) -> None:
        result = metric_result(
            np.asarray([1 + 1j]),
            np.asarray([1 + 1j]),
        )
        self.assertEqual(result["status"], "reference_only_requires_ground_truth")
        self.assertEqual(result["ber"], 0.0)
        self.assertEqual(result["ser"], 0.0)
        self.assertEqual(result["nmse"], 0.0)


if __name__ == "__main__":
    unittest.main()