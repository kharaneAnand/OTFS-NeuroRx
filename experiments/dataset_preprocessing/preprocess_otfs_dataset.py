import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTFS_ROOT = PROJECT_ROOT / "Phy_Mod_OTFS"

sys.path.insert(0, str(OTFS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config


# Experiment configuration

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiment_v1.yaml"
)


# Dataset paths

DATASET_ROOT = PROJECT_ROOT / "datasets" / "otfs"

RAW_DIR = (
    DATASET_ROOT
    / "raw"
)

METADATA_FILE = (
    DATASET_ROOT
    / "metadata"
    / "dataset_v1_metadata.csv"
)

PROCESSED_DIR = (
    DATASET_ROOT
    / "processed"
)


# Dataset splitting

def create_split(
    metadata: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """Create a reproducible stratified dataset split."""

    rng = np.random.default_rng(
        config.reproducibility.seed
    )

    train_ratio = config.split.train_ratio
    validation_ratio = config.split.validation_ratio

    metadata = metadata.copy()

    metadata["split"] = ""

    # Stratify the split across every SNR/velocity condition.

    for (
        snr,
        velocity,
    ), group in metadata.groupby(
        ["snr_db", "velocity_kmh"],
        sort=True,
    ):

        indices = group.index.to_numpy(
            copy=True
        )

        rng.shuffle(indices)

        total = len(indices)

        train_count = int(
            total * train_ratio
        )

        validation_count = int(
            total * validation_ratio
        )

        train_indices = indices[
            :train_count
        ]

        validation_indices = indices[
            train_count:
            train_count + validation_count
        ]

        test_indices = indices[
            train_count + validation_count:
        ]

        metadata.loc[
            train_indices,
            "split",
        ] = "train"

        metadata.loc[
            validation_indices,
            "split",
        ] = "validation"

        metadata.loc[
            test_indices,
            "split",
        ] = "test"

    return metadata


# Split validation

def validate_split(
    metadata: pd.DataFrame,
    config,
) -> None:
    """Validate global and per-condition split assignments."""

    expected_samples = (
        config.dataset.expected_samples
    )

    train_expected = int(
        expected_samples
        * config.split.train_ratio
    )

    validation_expected = int(
        expected_samples
        * config.split.validation_ratio
    )

    test_expected = (
        expected_samples
        - train_expected
        - validation_expected
    )

    if len(metadata) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples, "
            f"found {len(metadata)}."
        )

    if metadata["split"].isna().any():
        raise ValueError(
            "Missing split assignments."
        )

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    if not metadata["split"].isin(
        valid_splits
    ).all():
        raise ValueError(
            "Invalid split label detected."
        )

    split_counts = (
        metadata["split"]
        .value_counts()
    )

    expected_counts = {
        "train": train_expected,
        "validation": validation_expected,
        "test": test_expected,
    }

    for split, expected in expected_counts.items():

        actual = int(
            split_counts.get(
                split,
                0,
            )
        )

        if actual != expected:
            raise ValueError(
                f"Invalid {split} count: "
                f"expected {expected}, "
                f"found {actual}."
            )

    # Verify every SNR/velocity condition has the same split ratio.

    condition_counts = (
        metadata
        .groupby(
            [
                "snr_db",
                "velocity_kmh",
                "split",
            ]
        )
        .size()
    )

    for snr in sorted(
        metadata["snr_db"].unique()
    ):

        for velocity in sorted(
            metadata["velocity_kmh"].unique()
        ):

            condition_total = int(
                metadata[
                    (
                        metadata["snr_db"] == snr
                    )
                    & (
                        metadata["velocity_kmh"]
                        == velocity
                    )
                ].shape[0]
            )

            expected_train = int(
                condition_total
                * config.split.train_ratio
            )

            expected_validation = int(
                condition_total
                * config.split.validation_ratio
            )

            expected_test = (
                condition_total
                - expected_train
                - expected_validation
            )

            actual_train = int(
                condition_counts.get(
                    (
                        snr,
                        velocity,
                        "train",
                    ),
                    0,
                )
            )

            actual_validation = int(
                condition_counts.get(
                    (
                        snr,
                        velocity,
                        "validation",
                    ),
                    0,
                )
            )

            actual_test = int(
                condition_counts.get(
                    (
                        snr,
                        velocity,
                        "test",
                    ),
                    0,
                )
            )

            if (
                actual_train != expected_train
                or actual_validation
                != expected_validation
                or actual_test != expected_test
            ):
                raise ValueError(
                    "Invalid stratified split for "
                    f"SNR={snr}, velocity={velocity}."
                )


# Sample validation

def validate_sample_files(
    metadata: pd.DataFrame,
    config,
) -> None:
    """Validate raw sample files against the experiment contract."""

    print()
    print(
        "Validating sample representations..."
    )
    print("-" * 70)

    expected_grid_shape = (
        config.representation.expected_grid_shape
    )

    expected_data_symbols = (
        config.representation.expected_data_symbols
    )

    expected_channel_shape = (
        config.representation.expected_channel_shape
    )

    expected_num_paths = (
        config.representation.expected_num_paths
    )

    required_keys = {
        "rx_dd",
        "tx_dd",
        "h_dd",
        "his",
        "lis",
        "kis",
    }

    checked = 0

    for filename in metadata["file"]:

        sample_path = RAW_DIR / filename

        if not sample_path.exists():
            raise FileNotFoundError(
                f"Missing sample file: "
                f"{filename}"
            )

        with np.load(
            sample_path,
            allow_pickle=False,
        ) as data:

            # Verify required arrays are present.

            if not required_keys.issubset(
                data.files
            ):
                raise ValueError(
                    f"Missing arrays in {filename}."
                )

            rx_dd = data["rx_dd"]
            tx_dd = data["tx_dd"]
            h_dd = data["h_dd"]

            his = data["his"]
            lis = data["lis"]
            kis = data["kis"]

            # Verify dataset representation shapes.

            if tuple(rx_dd.shape) != (
                expected_grid_shape.rows,
                expected_grid_shape.cols,
            ):
                raise ValueError(
                    f"Invalid rx_dd shape in "
                    f"{filename}: {rx_dd.shape}"
                )

            if tx_dd.shape != (
                expected_data_symbols,
            ):
                raise ValueError(
                    f"Invalid tx_dd shape in "
                    f"{filename}: {tx_dd.shape}"
                )

            if tuple(h_dd.shape) != (
                expected_channel_shape.rows,
                expected_channel_shape.cols,
            ):
                raise ValueError(
                    f"Invalid h_dd shape in "
                    f"{filename}: {h_dd.shape}"
                )

            if his.shape != (
                expected_num_paths,
            ):
                raise ValueError(
                    f"Invalid his shape in "
                    f"{filename}: {his.shape}"
                )

            if lis.shape != (
                expected_num_paths,
            ):
                raise ValueError(
                    f"Invalid lis shape in "
                    f"{filename}: {lis.shape}"
                )

            if kis.shape != (
                expected_num_paths,
            ):
                raise ValueError(
                    f"Invalid kis shape in "
                    f"{filename}: {kis.shape}"
                )

            # Verify numerical values are finite.

            arrays = {
                "rx_dd": rx_dd,
                "tx_dd": tx_dd,
                "h_dd": h_dd,
                "his": his,
                "lis": lis,
                "kis": kis,
            }

            for name, array in arrays.items():

                if not np.isfinite(array).all():
                    raise ValueError(
                        f"{name} contains NaN/Inf "
                        f"in {filename}."
                    )

        checked += 1

        if checked % 100 == 0:
            print(
                f"  checked "
                f"{checked}/"
                f"{len(metadata)}"
            )


# Metadata validation

def validate_metadata(
    metadata: pd.DataFrame,
    config,
) -> None:
    """Validate dataset metadata against the experiment configuration."""

    expected_snr = set(
        config.channel.snr_db
    )

    expected_velocities = set(
        config.channel.velocity_kmh
    )

    if set(
        metadata["snr_db"].unique()
    ) != expected_snr:
        raise ValueError(
            "Metadata contains unexpected SNR values."
        )

    if set(
        metadata["velocity_kmh"].unique()
    ) != expected_velocities:
        raise ValueError(
            "Metadata contains unexpected velocity values."
        )

    if metadata["file"].duplicated().any():
        raise ValueError(
            "Duplicate sample filenames found."
        )

    if metadata["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample IDs found."
        )


# Main preprocessing pipeline

def preprocess_dataset() -> None:
    """Create and validate the reproducible V1 dataset split."""

    config = load_config(
        CONFIG_PATH
    )

    split_file = (
        PROCESSED_DIR
        / "dataset_v1_split.csv"
    )

    config_file = (
        PROCESSED_DIR
        / "dataset_v1_preprocessing_config.json"
    )

    print("=" * 70)
    print(
        "OTFS DATASET PREPROCESSING — V1"
    )
    print("=" * 70)

    # Verify required dataset inputs exist.

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found:\n"
            f"{RAW_DIR}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_FILE}"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"Input samples           : "
        f"{len(metadata)}"
    )

    # Validate metadata before creating the split.

    validate_metadata(
        metadata,
        config,
    )

    # Create reproducible stratified split.

    metadata = create_split(
        metadata,
        config,
    )

    # Validate split assignments.

    validate_split(
        metadata,
        config,
    )

    # Validate raw sample representations.

    validate_sample_files(
        metadata,
        config,
    )

    # Save split metadata.

    metadata.to_csv(
        split_file,
        index=False,
    )

    # Save preprocessing configuration.

    train_samples = int(
        (metadata["split"] == "train").sum()
    )

    validation_samples = int(
        (
            metadata["split"]
            == "validation"
        ).sum()
    )

    test_samples = int(
        (metadata["split"] == "test").sum()
    )

    config_output = {
        "dataset_version": config.project.dataset_version,
        "random_seed": config.reproducibility.seed,
        "train_ratio": config.split.train_ratio,
        "validation_ratio": config.split.validation_ratio,
        "test_ratio": config.split.test_ratio,
        "total_samples": len(metadata),
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
        "grid_shape": [
            config.representation.expected_grid_shape.rows,
            config.representation.expected_grid_shape.cols,
        ],
        "data_symbols": (
            config.representation.expected_data_symbols
        ),
        "channel_shape": [
            config.representation.expected_channel_shape.rows,
            config.representation.expected_channel_shape.cols,
        ],
        "num_paths": (
            config.representation.expected_num_paths
        ),
        "input": config.representation.input,
        "target": config.representation.target,
        "channel_ground_truth": (
            config.representation.channel_ground_truth
        ),
    }

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config_output,
            file,
            indent=4,
        )

    # Print final split summary.

    print()
    print(
        "Dataset split"
    )
    print("-" * 70)

    print(
        f"Train samples           : "
        f"{train_samples}"
    )

    print(
        f"Validation samples      : "
        f"{validation_samples}"
    )

    print(
        f"Test samples            : "
        f"{test_samples}"
    )

    print()
    print(
        "Files created"
    )
    print("-" * 70)

    print(
        f"Split metadata          : "
        f"{split_file}"
    )

    print(
        f"Configuration           : "
        f"{config_file}"
    )

    print()
    print("=" * 70)
    print(
        "DATASET PREPROCESSING COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    preprocess_dataset()