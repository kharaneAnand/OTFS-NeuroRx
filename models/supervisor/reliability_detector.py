"""Stateless per-receiver reliability features and z-distance references."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RELIABILITY_FEATURE_NAMES = (
    "receiver_qpsk_margin",
    "environment_distance",
)
RELIABILITY_SCHEMA_VERSION = 1


def qpsk_boundary_margin(estimate: np.ndarray) -> float:
    """Return the mean distance of complex QPSK estimates to a decision boundary."""

    estimate = np.asarray(estimate)
    if estimate.ndim != 1 or not np.iscomplexobj(estimate):
        raise ValueError("estimate must be a one-dimensional complex vector.")
    if not np.isfinite(estimate.real).all() or not np.isfinite(estimate.imag).all():
        raise ValueError("estimate contains non-finite values.")
    margin = np.minimum(np.abs(estimate.real), np.abs(estimate.imag))
    return float(np.mean(margin))


def reliability_feature_vector(
    estimate: np.ndarray,
    environment_distance: float,
) -> np.ndarray:
    if not np.isfinite(environment_distance):
        raise ValueError("environment_distance must be finite.")
    vector = np.asarray(
        [qpsk_boundary_margin(estimate), float(environment_distance)],
        dtype=np.float64,
    )
    if not np.isfinite(vector).all():
        raise FloatingPointError("Reliability feature vector is not finite.")
    return vector


@dataclass(frozen=True)
class ReceiverReliabilityReference:
    receiver: str
    feature_names: tuple[str, ...]
    mean: np.ndarray
    standard_deviation: np.ndarray
    schema_version: int = RELIABILITY_SCHEMA_VERSION

    @classmethod
    def fit(cls, receiver: str, feature_vectors: np.ndarray) -> "ReceiverReliabilityReference":
        values = np.asarray(feature_vectors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(RELIABILITY_FEATURE_NAMES):
            raise ValueError("Reliability reference must have exactly two features.")
        if values.shape[0] < 2 or not np.isfinite(values).all():
            raise ValueError("Reliability reference needs at least two finite rows.")
        standard_deviation = np.maximum(values.std(axis=0, ddof=1), 1e-8)
        return cls(
            receiver=receiver,
            feature_names=RELIABILITY_FEATURE_NAMES,
            mean=values.mean(axis=0),
            standard_deviation=standard_deviation,
        )

    def score(self, feature_vector: np.ndarray) -> float:
        vector = np.asarray(feature_vector, dtype=np.float64)
        if vector.shape != self.mean.shape or not np.isfinite(vector).all():
            raise ValueError("Reliability feature vector has an invalid shape/value.")
        z_scores = (vector - self.mean) / self.standard_deviation
        return float(np.sqrt(np.mean(z_scores ** 2)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "receiver": self.receiver,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "standard_deviation": self.standard_deviation.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "ReceiverReliabilityReference":
        if int(values["schema_version"]) != RELIABILITY_SCHEMA_VERSION:
            raise ValueError("Unsupported reliability reference schema.")
        names = tuple(str(name) for name in values["feature_names"])
        if names != RELIABILITY_FEATURE_NAMES:
            raise ValueError("Reliability feature schema does not match this code.")
        return cls(
            receiver=str(values["receiver"]),
            feature_names=names,
            mean=np.asarray(values["mean"], dtype=np.float64),
            standard_deviation=np.asarray(
                values["standard_deviation"], dtype=np.float64
            ),
        )
