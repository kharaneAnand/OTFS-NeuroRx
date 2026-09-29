"""Build fixed, target-free environment references from the training split."""

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

from models.supervisor.environment_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract_environment_features,
)
from models.supervisor.environment_change import make_regime_key
from src.config.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "environment_detector_v1.json",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    with args.detector_config.open(encoding="utf-8") as file:
        detector_config = json.load(file)

    dataset_root = PROJECT_ROOT / config.dataset.root
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = (
        dataset_root / config.dataset.processed_dir / config.dataset.split_metadata_file
    )
    metadata = pd.read_csv(metadata_path)
    training_rows = metadata.loc[
        metadata["split"].astype(str).str.lower()
        == str(detector_config["reference_split"]).lower()
    ].reset_index(drop=True)
    if training_rows.empty:
        raise ValueError("No training rows were found for detector references.")

    feature_rows = []
    context_ids = []
    sample_files = []
    for _, row in training_rows.iterrows():
        sample_file = str(row["file"])
        with np.load(raw_dir / sample_file, allow_pickle=False) as sample:
            rx_dd = np.asarray(sample["rx_dd"])
            h_hat = np.asarray(sample["h_hat"])
        features = extract_environment_features(rx_dd, h_hat)
        feature_rows.append([features[name] for name in FEATURE_NAMES])
        context_ids.append(
            make_regime_key(float(row["snr_db"]), float(row["velocity_kmh"]))
        )
        sample_files.append(sample_file)

    feature_matrix = np.asarray(feature_rows, dtype=np.float64)
    context_ids_array = np.asarray(context_ids, dtype=str)
    output_directory = PROJECT_ROOT / "experiments" / "environment_detector"
    output_directory.mkdir(parents=True, exist_ok=True)
    archive_path = output_directory / "reference_features_v1.npz"
    np.savez_compressed(
        archive_path,
        features=feature_matrix,
        feature_schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype=np.int32),
        feature_names=np.asarray(FEATURE_NAMES, dtype=str),
        context_ids=context_ids_array,
        sample_files=np.asarray(sample_files, dtype=str),
    )

    reference_counts = {
        context_id: int(np.count_nonzero(context_ids_array == context_id))
        for context_id in sorted(set(context_ids))
    }
    summary = {
        "reference_split": detector_config["reference_split"],
        "reference_samples": len(training_rows),
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_shape": list(feature_matrix.shape),
        "samples_per_context": reference_counts,
        "pooled_reference_included_at_load_time": True,
        "inputs_read_from_npz": ["rx_dd", "h_hat"],
        "targets_or_true_channel_read": False,
        "reference_file": str(archive_path.relative_to(PROJECT_ROOT)),
        "detector_config": str(args.detector_config),
    }
    with (output_directory / "reference_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
