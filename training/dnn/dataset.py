"""Dataset utilities for DNN-based OTFS symbol recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class OTFSReceiverDataset(Dataset):
    """Load OTFS receiver samples using the experiment split metadata."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        raw_dir: str | Path,
        input_shape: tuple[int, int],
        output_symbols: int,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.raw_dir = Path(raw_dir)
        self.input_shape = tuple(int(value) for value in input_shape)
        self.output_symbols = int(output_symbols)

        if len(self.input_shape) != 2:
            raise ValueError("input_shape must contain exactly two dimensions.")

        if any(value <= 0 for value in self.input_shape):
            raise ValueError("input_shape dimensions must be positive.")

        if self.output_symbols <= 0:
            raise ValueError("output_symbols must be positive.")

        if self.metadata.empty:
            raise ValueError("Dataset split contains no samples.")

        if "file" not in self.metadata.columns:
            raise ValueError("Dataset metadata must contain a 'file' column.")

        required_columns = {"file", "split"}

        missing_columns = required_columns.difference(self.metadata.columns)

        if missing_columns:
            raise ValueError(
                f"Dataset metadata is missing columns: {sorted(missing_columns)}."
            )

    def __len__(self) -> int:
        """Return the number of samples."""

        return len(self.metadata)

    def _sample_path(self, filename: str) -> Path:
        """Resolve a sample path inside the configured raw directory."""

        path = self.raw_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"OTFS sample not found: {path}"
            )

        return path

    def _validate_sample(
        self,
        sample: Any,
        filename: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate the arrays required by the DNN receiver."""

        if not isinstance(sample, np.lib.npyio.NpzFile):
            raise TypeError(
                f"Expected NPZ sample, received {type(sample).__name__}."
            )

        required_keys = {"rx_dd", "tx_dd"}

        missing_keys = required_keys.difference(sample.files)

        if missing_keys:
            raise ValueError(
                f"Sample '{filename}' is missing required arrays: "
                f"{sorted(missing_keys)}."
            )

        rx_dd = np.asarray(sample["rx_dd"])
        tx_dd = np.asarray(sample["tx_dd"])

        if rx_dd.shape != self.input_shape:
            raise ValueError(
                f"Sample '{filename}' has rx_dd shape {rx_dd.shape}; "
                f"expected {self.input_shape}."
            )

        expected_target_shape = (self.output_symbols,)

        if tx_dd.shape != expected_target_shape:
            raise ValueError(
                f"Sample '{filename}' has tx_dd shape {tx_dd.shape}; "
                f"expected {expected_target_shape}."
            )

        if not np.iscomplexobj(rx_dd):
            raise TypeError(
                f"Sample '{filename}' rx_dd must be complex-valued."
            )

        if not np.iscomplexobj(tx_dd):
            raise TypeError(
                f"Sample '{filename}' tx_dd must be complex-valued."
            )

        if not np.isfinite(rx_dd.real).all():
            raise ValueError(
                f"Sample '{filename}' rx_dd contains non-finite real values."
            )

        if not np.isfinite(rx_dd.imag).all():
            raise ValueError(
                f"Sample '{filename}' rx_dd contains non-finite imaginary values."
            )

        if not np.isfinite(tx_dd.real).all():
            raise ValueError(
                f"Sample '{filename}' tx_dd contains non-finite real values."
            )

        if not np.isfinite(tx_dd.imag).all():
            raise ValueError(
                f"Sample '{filename}' tx_dd contains non-finite imaginary values."
            )

        return rx_dd, tx_dd

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one received grid and its transmitted symbols."""

        row = self.metadata.iloc[index]
        filename = str(row["file"])
        sample_path = self._sample_path(filename)

        with np.load(sample_path, allow_pickle=False) as sample:
            rx_dd, tx_dd = self._validate_sample(
                sample,
                filename,
            )

        rx_tensor = torch.from_numpy(
            np.ascontiguousarray(rx_dd)
        ).to(torch.complex64)

        tx_tensor = torch.from_numpy(
            np.ascontiguousarray(tx_dd)
        ).to(torch.complex64)

        return rx_tensor, tx_tensor


def load_split_metadata(
    metadata_path: str | Path,
    split: str,
) -> pd.DataFrame:
    """Load samples belonging to one predefined dataset split."""

    metadata_path = Path(metadata_path)

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Dataset metadata not found: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    if "split" not in metadata.columns:
        raise ValueError(
            "Dataset metadata must contain a 'split' column."
        )

    split_metadata = metadata.loc[
        metadata["split"].astype(str).str.lower() == split.lower()
    ].copy()

    if split_metadata.empty:
        raise ValueError(
            f"No samples found for dataset split '{split}'."
        )

    return split_metadata.reset_index(drop=True)