"""Batched bipartite graph tensors for OTFS GNN detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.receivers.mmse import build_data_observation, mmse_detect


class OTFSGraphDataset(Dataset):
    """Return graph inputs and transmitted symbols for one split."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        raw_dir: str | Path,
        config,
        initialization: str,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.raw_dir = Path(raw_dir)
        self.config = config
        self.initialization = initialization

        if initialization not in {"mmse", "neutral_zero"}:
            raise ValueError(
                "initialization must be 'mmse' or 'neutral_zero'."
            )
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
        y_data = build_data_observation(rx_dd, self.config)
        mmse_estimate = mmse_detect(y_data, h_hat, noise_power)

        threshold = float(
            self.config.gnn.edge_threshold_fraction_of_max
        ) * max(
            float(np.max(np.abs(h_hat))),
            np.finfo(float).eps,
        )
        edge_mask = (np.abs(h_hat) >= threshold).astype(np.float32)
        edge_features = np.stack(
            (h_hat.real, h_hat.imag),
            axis=-1,
        ).astype(np.float32)

        if self.initialization == "mmse":
            symbol_initialization = np.stack(
                (mmse_estimate.real, mmse_estimate.imag),
                axis=-1,
            ).astype(np.float32)
        else:
            symbol_initialization = np.zeros(
                (h_hat.shape[1], 2),
                dtype=np.float32,
            )

        row_degree = edge_mask.mean(axis=1, keepdims=True)
        column_degree = edge_mask.mean(axis=0, keepdims=True).T
        observation_features = np.concatenate(
            (
                np.stack((y_data.real, y_data.imag), axis=-1),
                row_degree,
                np.full((y_data.shape[0], 1), noise_power, dtype=np.float32),
            ),
            axis=-1,
        ).astype(np.float32)
        symbol_features = np.concatenate(
            (
                symbol_initialization,
                np.linalg.norm(h_hat, axis=0)[:, None].astype(np.float32),
                column_degree,
            ),
            axis=-1,
        ).astype(np.float32)

        packed = np.concatenate(
            (
                observation_features.reshape(-1),
                symbol_features.reshape(-1),
                edge_features.reshape(-1),
                edge_mask.reshape(-1),
            )
        ).astype(np.float32)

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
