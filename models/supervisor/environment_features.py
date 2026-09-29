"""Fixed, target-free feature signature for OTFS environment monitoring."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "rx_log_power",
    "rx_peak_to_rms",
    "rx_shape_entropy",
    "channel_log_energy",
    "channel_active_fraction",
    "channel_energy_spread",
)
FEATURE_SCHEMA_VERSION = 2


def _normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(values.sum())
    if total <= np.finfo(np.float64).tiny or values.size <= 1:
        return 0.0
    probabilities = values / total
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    return entropy / float(np.log(values.size))


def extract_environment_features(
    rx_dd: np.ndarray,
    h_hat: np.ndarray,
) -> dict[str, float]:
    """Summarize one frame using only rx_dd and the estimated channel H_hat."""

    rx_dd = np.asarray(rx_dd)
    h_hat = np.asarray(h_hat)
    if rx_dd.ndim != 2 or h_hat.ndim != 2:
        raise ValueError("rx_dd and h_hat must both be two-dimensional.")
    if not np.iscomplexobj(rx_dd) or not np.iscomplexobj(h_hat):
        raise TypeError("rx_dd and h_hat must be complex-valued.")
    if not np.isfinite(rx_dd.real).all() or not np.isfinite(rx_dd.imag).all():
        raise ValueError("rx_dd contains non-finite values.")
    if not np.isfinite(h_hat.real).all() or not np.isfinite(h_hat.imag).all():
        raise ValueError("h_hat contains non-finite values.")

    epsilon = np.finfo(np.float64).tiny
    rx_power = np.abs(rx_dd.astype(np.complex128, copy=False)) ** 2
    channel_power = np.abs(h_hat.astype(np.complex128, copy=False)) ** 2
    rx_mean_power = float(np.mean(rx_power))
    rx_rms = float(np.sqrt(rx_mean_power))
    rx_peak_to_rms = float(np.max(np.abs(rx_dd)) / max(rx_rms, np.sqrt(epsilon)))
    rx_shape_entropy = 0.5 * (
        _normalized_entropy(rx_power.sum(axis=0))
        + _normalized_entropy(rx_power.sum(axis=1))
    )

    channel_mean_power = float(np.mean(channel_power))
    channel_peak = float(np.max(channel_power))
    threshold = 0.01 * max(np.sqrt(channel_peak), np.sqrt(epsilon))
    channel_active_fraction = float(np.mean(np.abs(h_hat) >= threshold))
    row_energy = channel_power.sum(axis=1)
    column_energy = channel_power.sum(axis=0)
    channel_energy_spread = 0.5 * (
        _normalized_entropy(row_energy)
        + _normalized_entropy(column_energy)
    )

    features = {
        "rx_log_power": float(np.log(max(rx_mean_power, epsilon))),
        "rx_peak_to_rms": rx_peak_to_rms,
        "rx_shape_entropy": rx_shape_entropy,
        "channel_log_energy": float(np.log(max(channel_mean_power, epsilon))),
        "channel_active_fraction": channel_active_fraction,
        "channel_energy_spread": channel_energy_spread,
    }
    if not all(np.isfinite(value) for value in features.values()):
        raise FloatingPointError("Environment feature extraction produced non-finite values.")
    return features


def feature_vector(features: dict[str, float]) -> np.ndarray:
    """Return the fixed-order feature vector used by calibration artifacts."""

    missing = set(FEATURE_NAMES).difference(features)
    if missing:
        raise ValueError(f"Environment features are missing: {sorted(missing)}")
    vector = np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Environment feature vector contains non-finite values.")
    return vector
