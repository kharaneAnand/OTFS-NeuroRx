"""Data-domain MMSE utilities for the OTFS receiver pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTFS_ROOT = PROJECT_ROOT / "Phy_Mod_OTFS"

if str(OTFS_ROOT) not in sys.path:
    sys.path.insert(0, str(OTFS_ROOT))

from OTFSResGrid import OTFSResGrid


def build_data_observation(
    rx_dd: np.ndarray,
    config,
    pilot_power: float = 1.0,
) -> np.ndarray:
    """Extract received observations corresponding to data-channel rows."""

    resource_grid = OTFSResGrid(
        config.otfs.M,
        config.otfs.N,
    )
    resource_grid.setPulse2Recta()
    resource_grid.setPilot2Center(
        config.pilot.pilot_delay_length,
        config.pilot.pilot_doppler_length,
    )
    resource_grid.setGuard(
        config.channel.max_delay,
        config.channel.max_delay,
        guard_doppl_full=True,
    )
    resource_grid.map(
        np.zeros(
            config.representation.expected_data_symbols,
            dtype=np.complex64,
        ),
        pilots_pow=pilot_power,
    )
    resource_grid.setContent(rx_dd)
    return np.asarray(resource_grid.getContentNoCE())


def mmse_detect(
    y_data: np.ndarray,
    h_hat: np.ndarray,
    noise_power: float,
) -> np.ndarray:
    """Estimate data symbols with a unit-prior linear MMSE detector."""

    if y_data.ndim != 1:
        raise ValueError("y_data must be a one-dimensional vector.")

    if h_hat.ndim != 2:
        raise ValueError("h_hat must be a two-dimensional matrix.")

    if h_hat.shape[0] != y_data.shape[0]:
        raise ValueError(
            "h_hat rows must match the length of y_data."
        )

    if h_hat.shape[1] <= 0:
        raise ValueError("h_hat must contain at least one symbol column.")

    if noise_power < 0:
        raise ValueError("noise_power must be non-negative.")

    identity = np.eye(
        h_hat.shape[1],
        dtype=h_hat.dtype,
    )
    normal_matrix = h_hat.conj().T @ h_hat + noise_power * identity
    right_hand_side = h_hat.conj().T @ y_data

    return np.linalg.solve(
        normal_matrix,
        right_hand_side,
    )