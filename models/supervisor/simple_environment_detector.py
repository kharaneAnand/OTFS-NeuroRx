"""Stateless, target-free environment distance for OTFS frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.receivers.mmse import build_data_observation, mmse_detect

SIMPLE_FEATURE_NAMES = (
    "estimated_snr_db",
    "h_hat_residual_nmse_proxy",
)
SIMPLE_FEATURE_SCHEMA_VERSION = 1


def extract_simple_environment_features(
    rx_dd: np.ndarray,
    h_hat: np.ndarray,
    noise_power: float,
    config,
) -> dict[str, float]:
    """Extract cheap, target-free SNR and channel-quality proxy features."""

    if noise_power < 0:
        raise ValueError("noise_power must be non-negative.")
    y_data = build_data_observation(rx_dd, config)
    mmse_estimate = mmse_detect(y_data, h_hat, noise_power)
    reconstruction = y_data - np.asarray(h_hat) @ mmse_estimate
    signal_power = float(np.sum(np.abs(y_data) ** 2))
    residual_power = float(np.sum(np.abs(reconstruction) ** 2))
    epsilon = np.finfo(np.float64).eps
    estimated_snr_db = 10.0 * np.log10(
        max(signal_power, epsilon) / max(residual_power, epsilon)
    )
    residual_nmse_proxy = residual_power / max(signal_power, epsilon)
    features = {
        "estimated_snr_db": float(estimated_snr_db),
        "h_hat_residual_nmse_proxy": float(residual_nmse_proxy),
    }
    if not all(np.isfinite(value) for value in features.values()):
        raise FloatingPointError("Simple environment features are not finite.")
    return features


def simple_feature_vector(features: dict[str, float]) -> np.ndarray:
    missing = set(SIMPLE_FEATURE_NAMES).difference(features)
    if missing:
        raise ValueError(f"Missing simple environment features: {sorted(missing)}")
    vector = np.asarray(
        [features[name] for name in SIMPLE_FEATURE_NAMES],
        dtype=np.float64,
    )
    if not np.isfinite(vector).all():
        raise ValueError("Simple environment feature vector is not finite.")
    return vector


@dataclass(frozen=True)
class SimpleEnvironmentReference:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    standard_deviation: np.ndarray
    schema_version: int = SIMPLE_FEATURE_SCHEMA_VERSION

    @classmethod
    def fit(cls, feature_vectors: np.ndarray) -> "SimpleEnvironmentReference":
        values = np.asarray(feature_vectors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(SIMPLE_FEATURE_NAMES):
            raise ValueError(
                f"Expected a 2-D array with {len(SIMPLE_FEATURE_NAMES)} features."
            )
        if values.shape[0] < 2 or not np.isfinite(values).all():
            raise ValueError("Reference features need at least two finite rows.")
        standard_deviation = values.std(axis=0, ddof=1)
        standard_deviation = np.maximum(standard_deviation, 1e-8)
        return cls(
            feature_names=SIMPLE_FEATURE_NAMES,
            mean=values.mean(axis=0),
            standard_deviation=standard_deviation,
        )

    def score(self, features: dict[str, float] | np.ndarray) -> float:
        vector = (
            simple_feature_vector(features)
            if isinstance(features, dict)
            else np.asarray(features, dtype=np.float64)
        )
        if vector.shape != self.mean.shape or not np.isfinite(vector).all():
            raise ValueError("Incoming feature vector has an invalid shape or value.")
        z_scores = (vector - self.mean) / self.standard_deviation
        return float(np.sqrt(np.mean(z_scores ** 2)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "standard_deviation": self.standard_deviation.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "SimpleEnvironmentReference":
        if int(values["schema_version"]) != SIMPLE_FEATURE_SCHEMA_VERSION:
            raise ValueError("Unsupported simple detector reference schema.")
        names = tuple(str(name) for name in values["feature_names"])
        if names != SIMPLE_FEATURE_NAMES:
            raise ValueError("Simple detector feature names do not match this code.")
        return cls(
            feature_names=names,
            mean=np.asarray(values["mean"], dtype=np.float64),
            standard_deviation=np.asarray(
                values["standard_deviation"], dtype=np.float64
            ),
        )
