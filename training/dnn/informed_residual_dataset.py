"""Feature preparation for the informed MMSE residual receiver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.receivers.mmse import build_data_observation, mmse_detect


def build_informed_features(
    rx_dd: np.ndarray,
    h_hat: np.ndarray,
    noise_power: float,
    config,
) -> tuple[np.ndarray, np.ndarray]:
    """Build real features and the MMSE estimate used by the residual path."""

    y_data = build_data_observation(rx_dd, config)
    mmse_estimate = mmse_detect(y_data, h_hat, noise_power)
    reconstruction_residual = y_data - h_hat @ mmse_estimate
    channel_strength = np.linalg.norm(h_hat, axis=0)

    scale = max(float(np.sqrt(noise_power)), np.finfo(float).eps)
    feature_parts = (
        mmse_estimate.real,
        mmse_estimate.imag,
        y_data.real / scale,
        y_data.imag / scale,
        reconstruction_residual.real / scale,
        reconstruction_residual.imag / scale,
        channel_strength,
        np.asarray([np.log10(max(noise_power, np.finfo(float).eps))]),
    )
    features = np.concatenate(feature_parts).astype(np.float32)
    return features, mmse_estimate


class InformedResidualDataset(Dataset):
    """Return informed features and the residual target."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        raw_dir: str | Path,
        config,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.raw_dir = Path(raw_dir)
        self.config = config

        if self.metadata.empty:
            raise ValueError("Dataset split contains no samples.")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        sample_path = self.raw_dir / str(row["file"])

        with np.load(sample_path, allow_pickle=False) as sample:
            rx_dd = np.asarray(sample["rx_dd"])
            h_hat = np.asarray(sample["h_hat"])
            tx_dd = np.asarray(sample["tx_dd"])

        noise_power = 10.0 ** (-float(row["snr_db"]) / 10.0)
        features, mmse_estimate = build_informed_features(
            rx_dd,
            h_hat,
            noise_power,
            self.config,
        )
        residual_target = tx_dd - mmse_estimate

        return (
            torch.from_numpy(features),
            torch.from_numpy(
                np.ascontiguousarray(residual_target)
            ).to(torch.complex64),
        )


def load_split_metadata(
    metadata_path: str | Path,
    split: str,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    split_metadata = metadata.loc[
        metadata["split"].astype(str).str.lower() == split.lower()
    ].copy()
    if split_metadata.empty:
        raise ValueError(f"No samples found for split '{split}'.")
    return split_metadata.reset_index(drop=True)