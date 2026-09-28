"""Dataset adapter for the isolated PI-EGNN experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.gnn.pi_egnn import qpsk_max_posterior
class PIEGNNGraphDataset(Dataset):
    """Build MMSE-initialized graph inputs with cached OAMP priors."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        raw_dir: str | Path,
        config,
        feature_cache: dict[str, np.ndarray],
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.raw_dir = Path(raw_dir)
        self.config = config
        self.feature_cache = feature_cache
        self._cache_indices = {
            str(file_name): index
            for index, file_name in enumerate(feature_cache["sample_files"])
        }
        if self.metadata.empty:
            raise ValueError("Dataset split contains no samples.")
        missing = set(self.metadata["file"].astype(str)) - self._cache_indices.keys()
        if missing:
            raise KeyError(f"Feature cache does not contain samples: {sorted(missing)[:3]}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        with np.load(self.raw_dir / str(row["file"]), allow_pickle=False) as sample:
            h_hat = np.asarray(sample["h_hat"])
            tx_dd = np.asarray(sample["tx_dd"])

        noise_power = 10.0 ** (-float(row["snr_db"]) / 10.0)
        cache_index = self._cache_indices[str(row["file"])]
        y_data = self.feature_cache["y_data"][cache_index]
        mmse_estimate = self.feature_cache["mmse_estimate"][cache_index]
        linear_estimate = self.feature_cache["linear_estimate"][cache_index]
        effective_variance = float(
            self.feature_cache["effective_variance"][cache_index]
        )
        oamp_estimate = self.feature_cache["prior_estimate"][cache_index]

        threshold = float(self.config.gnn.edge_threshold_fraction_of_max) * max(
            float(np.max(np.abs(h_hat))), np.finfo(float).eps
        )
        edge_mask = (np.abs(h_hat) >= threshold).astype(np.float32)
        edge_features = np.stack((h_hat.real, h_hat.imag), axis=-1).astype(np.float32)
        column_degree = edge_mask.mean(axis=0)[:, None]
        column_norm = np.linalg.norm(h_hat, axis=0)[:, None]
        confidence = qpsk_max_posterior(linear_estimate, effective_variance)[:, None]
        reliability = noise_power / max(
            noise_power + effective_variance, np.finfo(float).eps
        )
        symbol_features = np.concatenate(
            (
                np.stack((mmse_estimate.real, mmse_estimate.imag), axis=-1),
                column_norm,
                column_degree,
                np.stack((oamp_estimate.real, oamp_estimate.imag), axis=-1),
                np.abs(oamp_estimate)[:, None],
                np.full((h_hat.shape[1], 1), np.log1p(effective_variance)),
                np.full((h_hat.shape[1], 1), reliability),
                confidence,
            ),
            axis=-1,
        ).astype(np.float32)

        observation_features = np.concatenate(
            (
                np.stack((y_data.real, y_data.imag), axis=-1),
                edge_mask.mean(axis=1, keepdims=True),
                np.full((y_data.shape[0], 1), noise_power),
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
            torch.from_numpy(np.ascontiguousarray(tx_dd)).to(torch.complex64),
        )


def load_split_metadata(metadata_path: str | Path, split: str) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    selected = metadata.loc[
        metadata["split"].astype(str).str.lower() == split.lower()
    ].copy()
    if selected.empty:
        raise ValueError(f"No samples found for split '{split}'.")
    return selected.reset_index(drop=True)