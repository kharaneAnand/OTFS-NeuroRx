"""Fit one stateless simple environment reference from training frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.supervisor.simple_environment_detector import (
    SIMPLE_FEATURE_NAMES,
    SimpleEnvironmentReference,
    extract_simple_environment_features,
    simple_feature_vector,
)
from src.config.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_root = PROJECT_ROOT / config.dataset.root
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = dataset_root / config.dataset.processed_dir / config.dataset.split_metadata_file
    metadata = pd.read_csv(metadata_path)
    training = metadata.loc[
        metadata["split"].astype(str).str.lower() == "train"
    ].reset_index(drop=True)
    if training.empty:
        raise ValueError("Training split is empty.")

    rows = []
    for _, row in training.iterrows():
        with np.load(raw_dir / str(row["file"]), allow_pickle=False) as sample:
            rx_dd = np.asarray(sample["rx_dd"])
            h_hat = np.asarray(sample["h_hat"])
        noise_power = 10.0 ** (-float(row["snr_db"]) / 10.0)
        features = extract_simple_environment_features(
            rx_dd, h_hat, noise_power, config
        )
        rows.append(simple_feature_vector(features))

    feature_matrix = np.asarray(rows, dtype=np.float64)
    reference = SimpleEnvironmentReference.fit(feature_matrix)
    output_directory = PROJECT_ROOT / "experiments" / "environment_detector_simple"
    output_directory.mkdir(parents=True, exist_ok=True)
    reference_path = output_directory / "reference.json"
    with reference_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                **reference.to_dict(),
                "reference_split": "train",
                "reference_samples": len(training),
                "inputs_read_from_npz": ["rx_dd", "h_hat"],
                "targets_or_true_channel_read": False,
            },
            file,
            indent=2,
        )

    print(
        json.dumps(
            {
                "reference_file": str(reference_path.relative_to(PROJECT_ROOT)),
                "reference_samples": len(training),
                "feature_names": list(SIMPLE_FEATURE_NAMES),
                "mean": reference.mean.tolist(),
                "standard_deviation": reference.standard_deviation.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
