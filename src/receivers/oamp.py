"""Analytical OAMP detector for the reduced OTFS data-domain system."""

from __future__ import annotations

import numpy as np


def qpsk_posterior_mean(
    value: np.ndarray,
    variance: float,
) -> np.ndarray:
    """Return the posterior mean for unit-power QPSK symbols."""

    variance = max(float(variance), np.finfo(float).eps)
    scale = 1.0 / np.sqrt(2.0)
    real = scale * np.tanh(
        np.sqrt(2.0) * value.real / variance
    )
    imag = scale * np.tanh(
        np.sqrt(2.0) * value.imag / variance
    )
    return real + 1j * imag


def oamp_detect(
    y_data: np.ndarray,
    h_hat: np.ndarray,
    noise_power: float,
    iterations: int = 20,
) -> np.ndarray:
    """Detect QPSK symbols using analytical OAMP iterations.

    The initial effective noise variance is the configured per-sample
    channel noise power used during pilot-based H_hat generation.
    """

    y_data = np.asarray(y_data)
    h_hat = np.asarray(h_hat)

    if y_data.ndim != 1:
        raise ValueError("y_data must be one-dimensional.")
    if h_hat.ndim != 2:
        raise ValueError("h_hat must be two-dimensional.")
    if h_hat.shape[0] != y_data.shape[0]:
        raise ValueError("h_hat rows must match y_data length.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if noise_power < 0:
        raise ValueError("noise_power must be non-negative.")

    symbol_count = h_hat.shape[1]
    identity = np.eye(symbol_count, dtype=h_hat.dtype)
    gram = h_hat.conj().T @ h_hat
    linear_inverse = np.linalg.solve(
        gram + noise_power * identity,
        h_hat.conj().T,
    )
    orthogonalization = symbol_count / np.trace(
        linear_inverse @ h_hat
    ).real
    linear_operator = orthogonalization * linear_inverse

    estimate = np.zeros(symbol_count, dtype=h_hat.dtype)
    effective_variance = max(
        float(noise_power),
        np.finfo(float).eps,
    )

    for _ in range(iterations):
        residual = y_data - h_hat @ estimate
        linear_estimate = estimate + linear_operator @ residual
        linear_error = linear_estimate - estimate
        effective_variance = max(
            float(
                np.mean(np.abs(linear_error) ** 2)
                + noise_power
                * np.mean(np.abs(linear_operator @ linear_operator.conj().T).real)
            ),
            np.finfo(float).eps,
        )
        estimate = qpsk_posterior_mean(
            linear_estimate,
            effective_variance,
        )

    return estimate