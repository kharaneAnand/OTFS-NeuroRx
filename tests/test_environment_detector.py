"""Focused, data-free tests for the environment detector state machine."""

from __future__ import annotations

import unittest

import numpy as np

from models.supervisor.environment_change import EnvironmentChangeDetector
from models.supervisor.environment_features import (
    FEATURE_NAMES,
    extract_environment_features,
)


class EnvironmentFeatureTests(unittest.TestCase):
    def test_feature_signature_is_fixed_and_finite(self) -> None:
        rng = np.random.default_rng(19)
        rx_dd = (
            rng.normal(size=(16, 32)) + 1j * rng.normal(size=(16, 32))
        ).astype(np.complex64)
        h_hat = (
            rng.normal(size=(432, 368)) + 1j * rng.normal(size=(432, 368))
        ).astype(np.complex64)

        features = extract_environment_features(rx_dd, h_hat)

        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertTrue(np.isfinite(list(features.values())).all())
        self.assertTrue(0.0 <= features["channel_active_fraction"] <= 1.0)

    def test_zero_channel_has_zero_active_fraction(self) -> None:
        features = extract_environment_features(
            np.ones((16, 32), dtype=np.complex64),
            np.zeros((432, 368), dtype=np.complex64),
        )
        self.assertEqual(features["channel_active_fraction"], 0.0)

    def test_log_power_preserves_multiplicative_gain_change(self) -> None:
        rng = np.random.default_rng(21)
        rx_dd = (
            rng.normal(size=(16, 32)) + 1j * rng.normal(size=(16, 32))
        ).astype(np.complex64)
        h_hat = (
            rng.normal(size=(432, 368)) + 1j * rng.normal(size=(432, 368))
        ).astype(np.complex64)
        gain = 1.5

        baseline = extract_environment_features(rx_dd, h_hat)
        shifted = extract_environment_features(rx_dd, gain * h_hat)

        self.assertAlmostEqual(
            shifted["channel_log_energy"] - baseline["channel_log_energy"],
            2.0 * np.log(gain),
            places=5,
        )


class EnvironmentDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(31)
        self.references = {
            "pooled": {
                name: rng.normal(size=80)
                for name in FEATURE_NAMES
            },
            "snr=10|velocity=30": {
                name: rng.normal(size=40)
                for name in FEATURE_NAMES
            },
            "snr=15|velocity=30": {
                name: rng.normal(size=40)
                for name in FEATURE_NAMES
            },
        }
        self.sample = {name: 0.0 for name in FEATURE_NAMES}

    def test_context_selection_and_status_are_explicit(self) -> None:
        detector = EnvironmentChangeDetector(self.references, seed=4)
        first = detector.update(self.sample, "snr=10|velocity=30")
        second = detector.update(self.sample, "snr=15|velocity=30")
        third = detector.update(self.sample, "unknown-context")

        self.assertEqual(first.reference_id, "snr=10|velocity=30")
        self.assertFalse(first.known_context_change)
        self.assertTrue(first.status in {"NO_SHIFT_EVIDENCE", "SHIFT_DETECTED"})
        self.assertTrue(second.known_context_change)
        self.assertFalse(second.unsupported_context)
        self.assertTrue(third.unsupported_context)
        self.assertIn("UNSUPPORTED_CONTEXT", third.status)

    def test_state_restore_reproduces_future_evidence(self) -> None:
        original = EnvironmentChangeDetector(self.references, seed=17)
        original.update(self.sample, "snr=10|velocity=30")
        saved = original.state_dict()

        restored = EnvironmentChangeDetector(self.references, seed=999)
        restored.load_state_dict(saved)
        next_original = original.update(self.sample, "snr=10|velocity=30")
        next_restored = restored.update(self.sample, "snr=10|velocity=30")

        self.assertEqual(next_original.evidence, next_restored.evidence)
        self.assertEqual(next_original.status, next_restored.status)

    def test_threshold_uses_union_budget_over_references_and_scores(self) -> None:
        detector = EnvironmentChangeDetector(self.references, alpha=0.05, seed=2)
        self.assertAlmostEqual(
            detector.alpha_per_monitor,
            0.05 / (len(self.references) * len(FEATURE_NAMES)),
        )
        result = detector.update(self.sample, "snr=10|velocity=30")
        self.assertEqual(set(result.evidence), set(FEATURE_NAMES))
        self.assertTrue(
            all(np.isfinite(item["portfolio_log_wealth"]) for item in result.evidence.values())
        )

    def test_extreme_persistent_shift_alarms_within_bounded_stream(self) -> None:
        references = {
            "known": {
                name: np.linspace(-0.05, 0.05, 80)
                for name in FEATURE_NAMES
            }
        }
        detector = EnvironmentChangeDetector(
            references,
            alpha=0.05,
            seed=13,
            familywise_contexts=False,
        )
        shifted_features = {name: 10.0 for name in FEATURE_NAMES}
        decisions = [
            detector.update(shifted_features, regime_id="known")
            for _ in range(100)
        ]

        alarm_indices = [
            index for index, decision in enumerate(decisions, start=1)
            if decision.status == "SHIFT_DETECTED"
        ]
        self.assertTrue(alarm_indices, "Extreme persistent feature shift was not detected.")
        self.assertLessEqual(alarm_indices[0], 100)
        self.assertTrue(all(
            decision.status == "SHIFT_DETECTED"
            for decision in decisions[alarm_indices[0] - 1:]
        ))


if __name__ == "__main__":
    unittest.main()
