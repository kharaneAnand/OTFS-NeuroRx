"""Unit tests for the rule-based reliability controller."""

from __future__ import annotations

import unittest

from models.supervisor.reliability_controller import (
    ControllerThresholds,
    ReliabilityController,
)


class ReliabilityControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.references = {
            name: {
                "mean": [0.60, 0.70],
                "standard_deviation": [0.10, 0.20],
            }
            for name in ("mmse", "oamp_dl", "original_gnn", "pi_egnn")
        }
        self.controller = ReliabilityController(
            self.references,
            ControllerThresholds(
                low_z_threshold=-1.0,
                switch_z_gap=0.5,
                consecutive_low_frames=2,
                minimum_frames_between_switches=10,
            ),
        )

    def test_candidates_are_not_requested_before_two_low_frames(self) -> None:
        calls = []

        def provide_candidates():
            calls.append(True)
            return {"mmse": 0.7, "original_gnn": 0.7, "pi_egnn": 0.7}

        first = self.controller.step(0.45, 0.3, provide_candidates)
        self.assertEqual(first.decision, "KEEP")
        self.assertEqual(calls, [])

        second = self.controller.step(0.45, 0.4, provide_candidates)
        self.assertEqual(second.decision, "SWITCH")
        self.assertEqual(len(calls), 1)
        self.assertEqual(second.active_after, "mmse")

    def test_normal_active_confidence_keeps_without_candidate_inference(self) -> None:
        calls = []
        result = self.controller.step(
            0.75,
            0.9,
            lambda: calls.append(True) or {"pi_egnn": 0.9},
        )
        self.assertEqual(result.decision, "KEEP")
        self.assertEqual(calls, [])

    def test_adapt_recommends_best_alternative_not_active_model(self) -> None:
        result = self.controller.step(
            0.45,
            0.4,
            lambda: {
                "mmse": 0.46,
                "oamp_dl": 0.45,
                "original_gnn": 0.47,
                "pi_egnn": 0.455,
            },
        )
        self.assertEqual(result.decision, "KEEP")
        result = self.controller.step(
            0.45,
            0.4,
            lambda: {
                "mmse": 0.46,
                "oamp_dl": 0.45,
                "original_gnn": 0.47,
                "pi_egnn": 0.455,
            },
        )
        self.assertEqual(result.decision, "ADAPT")
        self.assertEqual(result.recommended_adaptation_model, "original_gnn")

    def test_candidate_comparison_uses_each_receivers_own_z_scale(self) -> None:
        references = {
            "mmse": {"mean": [0.9, 0.7], "standard_deviation": [0.1, 0.2]},
            "oamp_dl": {"mean": [0.6, 0.7], "standard_deviation": [0.1, 0.2]},
            "original_gnn": {"mean": [1.0, 0.7], "standard_deviation": [0.1, 0.2]},
            "pi_egnn": {"mean": [1.0, 0.7], "standard_deviation": [0.1, 0.2]},
        }
        controller = ReliabilityController(references)
        candidates = {
            "mmse": 0.79,
            "original_gnn": 0.8,
            "pi_egnn": 0.8,
        }

        controller.step(0.45, 0.0, lambda: candidates)
        result = controller.step(0.45, 0.0, lambda: candidates)

        self.assertEqual(result.decision, "ADAPT")
        self.assertLess(result.candidate_confidence_z_scores["mmse"], -1.0)
        self.assertAlmostEqual(result.candidate_confidence_z_scores["mmse"], -1.1)

    def test_absolute_ber_gate_rejects_pi_egnn_and_keeps_gnn_eligible(self) -> None:
        validated_ber = {
            "mmse": 0.06688,
            "oamp_dl": 0.03296,
            "original_gnn": 0.03687,
            "pi_egnn": 0.04003,
        }
        controller = ReliabilityController(
            self.references,
            validated_ber=validated_ber,
            performance_margin=0.005,
        )
        candidate_margins = {
            "mmse": 0.90,
            "original_gnn": 0.75,
            "pi_egnn": 0.80,
        }

        controller.step(0.45, 0.0, lambda: candidate_margins)
        result = controller.step(0.45, 0.0, lambda: candidate_margins)

        self.assertEqual(result.decision, "SWITCH")
        self.assertEqual(result.active_after, "original_gnn")
        self.assertTrue(
            result.candidate_gate_results["pi_egnn"]["passes_confidence_z_gap"]
        )
        self.assertFalse(
            result.candidate_gate_results["pi_egnn"]["passes_best_receiver_ber_floor"]
        )


if __name__ == "__main__":
    unittest.main()
