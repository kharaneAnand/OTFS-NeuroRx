"""Dataset that prepares MMSE estimates and residual targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.receivers.mmse import build_data_observation, mmse_detect


class MMSEResidualDataset(Dataset):
    """Return an MMSE estimate and its correction target."""

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

        required_columns = {"file", "split", "snr_db"}
        missing_columns = required_columns.difference(self.metadata.columns)
        if missing_columns:
            raise ValueError(
                f"Dataset metadata is missing columns: {sorted(missing_columns)}."
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        sample_path = self.raw_dir / str(row["file"])

        if not sample_path.is_file():
            raise FileNotFoundError(f"OTFS sample not found: {sample_path}")

        with np.load(sample_path, allow_pickle=False) as sample:
            required_keys = {"rx_dd", "h_hat", "tx_dd"}
            missing_keys = required_keys.difference(sample.files)
            if missing_keys:
                raise ValueError(
                    f"Sample is missing arrays: {sorted(missing_keys)}"
                )

            rx_dd = np.asarray(sample["rx_dd"])
            h_hat = np.asarray(sample["h_hat"])
            tx_dd = np.asarray(sample["tx_dd"])

        expected_grid = (
            self.config.representation.expected_grid_shape.rows,
            self.config.representation.expected_grid_shape.cols,
        )
        expected_target = (
            self.config.representation.expected_data_symbols,
        )
        expected_channel = (
            self.config.representation.expected_channel_shape.rows,
            self.config.representation.expected_channel_shape.cols,
        )

        if rx_dd.shape != expected_grid:
            raise ValueError(f"Unexpected rx_dd shape: {rx_dd.shape}")
        if tx_dd.shape != expected_target:
            raise ValueError(f"Unexpected tx_dd shape: {tx_dd.shape}")
        if h_hat.shape != expected_channel:
            raise ValueError(f"Unexpected h_hat shape: {h_hat.shape}")

        noise_power = 10.0 ** (-float(row["snr_db"]) / 10.0)
        y_data = build_data_observation(
            rx_dd,
            self.config,
        )
        mmse_estimate = mmse_detect(
            y_data,
            h_hat,
            noise_power,
        )
        residual_target = tx_dd - mmse_estimate

        return (
            torch.from_numpy(
                np.ascontiguousarray(mmse_estimate)
            ).to(torch.complex64),
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