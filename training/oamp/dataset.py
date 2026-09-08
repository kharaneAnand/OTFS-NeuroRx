"""Dataset preparation for unfolded OAMP-DL training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.receivers.mmse import build_data_observation


class OAMPDataset(Dataset):
    """Pack received data, H_hat, and noise power for OAMP-DL."""

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
        y_data = build_data_observation(
            rx_dd,
            self.config,
        )
        packed = np.concatenate(
            (
                y_data.reshape(-1),
                h_hat.reshape(-1),
                np.asarray([noise_power], dtype=np.complex64),
            )
        ).astype(np.complex64)

        return (
            torch.from_numpy(packed),
            torch.from_numpy(
                np.ascontiguousarray(tx_dd)
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