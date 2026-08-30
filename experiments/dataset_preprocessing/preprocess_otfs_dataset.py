import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTFS_ROOT = PROJECT_ROOT / "Phy_Mod_OTFS"

sys.path.insert(0, str(OTFS_ROOT))


# Dataset paths

DATASET_ROOT = PROJECT_ROOT / "datasets" / "otfs"

RAW_DIR = DATASET_ROOT / "raw"

METADATA_FILE = (
    DATASET_ROOT
    / "metadata"
    / "dataset_v1_metadata.csv"
)

PROCESSED_DIR = DATASET_ROOT / "processed"

SPLIT_FILE = (
    PROCESSED_DIR
    / "dataset_v1_split.csv"
)

CONFIG_FILE = (
    PROCESSED_DIR
    / "dataset_v1_preprocessing_config.json"
)


# Configuration

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EXPECTED_SAMPLES = 900

EXPECTED_M = 32
EXPECTED_N = 16
EXPECTED_DATA_SYMBOLS = 368


# Validation

assert (
    TRAIN_RATIO
    + VAL_RATIO
    + TEST_RATIO
    == 1.0
), "Dataset split ratios must sum to 1"


# Dataset splitting

def create_split(metadata):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    metadata = metadata.copy()

    metadata["split"] = ""

    # Stratified split by SNR and velocity.
    #
    # Each condition contains 100 samples.
    # We keep the same proportion of samples
    # from every condition in train/validation/test.

    for (
        snr,
        velocity,
    ), group in metadata.groupby(
        ["snr_db", "velocity_kmh"],
        sort=True,
    ):

        indices = group.index.to_numpy(copy=True)

        rng.shuffle(indices)

        total = len(indices)

        train_count = int(
            total * TRAIN_RATIO
        )

        val_count = int(
            total * VAL_RATIO
        )

        train_indices = indices[
            :train_count
        ]

        val_indices = indices[
            train_count:
            train_count + val_count
        ]

        test_indices = indices[
            train_count + val_count:
        ]

        metadata.loc[
            train_indices,
            "split",
        ] = "train"

        metadata.loc[
            val_indices,
            "split",
        ] = "validation"

        metadata.loc[
            test_indices,
            "split",
        ] = "test"

    return metadata


# Sample validation

def validate_processed_split(
    metadata,
):

    assert (
        len(metadata)
        == EXPECTED_SAMPLES
    ), "Incorrect number of samples"

    assert (
        metadata["split"]
        .notna()
        .all()
    ), "Missing split assignments"

    assert (
        metadata["split"]
        .isin(
            [
                "train",
                "validation",
                "test",
            ]
        )
        .all()
    ), "Invalid split label"

    split_counts = (
        metadata["split"]
        .value_counts()
    )

    assert (
        split_counts.get(
            "train",
            0,
        )
        == 630
    ), "Incorrect training sample count"

    assert (
        split_counts.get(
            "validation",
            0,
        )
        == 135
    ), "Incorrect validation sample count"

    assert (
        split_counts.get(
            "test",
            0,
        )
        == 135
    ), "Incorrect test sample count"

    # Verify every condition has the
    # expected 70/15/15 distribution.

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

            assert (
                condition_counts.get(
                    (
                        snr,
                        velocity,
                        "train",
                    ),
                    0,
                )
                == 70
            )

            assert (
                condition_counts.get(
                    (
                        snr,
                        velocity,
                        "validation",
                    ),
                    0,
                )
                == 15
            )

            assert (
                condition_counts.get(
                    (
                        snr,
                        velocity,
                        "test",
                    ),
                    0,
                )
                == 15
            )


# Sample format validation

def validate_sample_files(
    metadata,
):

    print()
    print(
        "Validating sample representations..."
    )
    print("-" * 70)

    checked = 0

    for filename in metadata[
        "file"
    ]:

        sample_path = (
            RAW_DIR / filename
        )

        assert (
            sample_path.exists()
        ), (
            f"Missing sample file: "
            f"{filename}"
        )

        with np.load(
            sample_path,
            allow_pickle=False,
        ) as data:

            # Required arrays

            required_keys = {
                "rx_dd",
                "tx_dd",
                "h_dd",
                "his",
                "lis",
                "kis",
            }

            assert required_keys.issubset(
                data.files
            ), (
                f"Missing arrays in "
                f"{filename}"
            )

            rx_dd = data["rx_dd"]
            tx_dd = data["tx_dd"]
            h_dd = data["h_dd"]

            his = data["his"]
            lis = data["lis"]
            kis = data["kis"]

            # Shape checks

            assert rx_dd.shape == (
                EXPECTED_N,
                EXPECTED_M,
            ), (
                f"Invalid rx_dd shape "
                f"in {filename}"
            )

            assert (
                tx_dd.shape
                == (EXPECTED_DATA_SYMBOLS,)
            ), (
                f"Invalid tx_dd shape "
                f"in {filename}"
            )

            assert h_dd.shape == (
                432,
                EXPECTED_DATA_SYMBOLS,
            ), (
                f"Invalid h_dd shape "
                f"in {filename}"
            )

            assert his.shape == (
                4,
            ), (
                f"Invalid his shape "
                f"in {filename}"
            )

            assert lis.shape == (
                4,
            ), (
                f"Invalid lis shape "
                f"in {filename}"
            )

            assert kis.shape == (
                4,
            ), (
                f"Invalid kis shape "
                f"in {filename}"
            )

            # Numerical checks

            assert np.isfinite(
                rx_dd
            ).all()

            assert np.isfinite(
                tx_dd
            ).all()

            assert np.isfinite(
                h_dd
            ).all()

            assert np.isfinite(
                his
            ).all()

            assert np.isfinite(
                lis
            ).all()

            assert np.isfinite(
                kis
            ).all()

        checked += 1

        if checked % 100 == 0:

            print(
                f"  checked "
                f"{checked}/"
                f"{len(metadata)}"
            )


# Main preprocessing pipeline

def preprocess_dataset():

    print("=" * 70)
    print(
        "OTFS DATASET PREPROCESSING — V1"
    )
    print("=" * 70)

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

    # Create processed directory

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load metadata

    metadata = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"Input samples           : "
        f"{len(metadata)}"
    )

    # Create reproducible stratified split

    metadata = create_split(
        metadata
    )

    # Validate split

    validate_processed_split(
        metadata
    )

    # Validate raw sample representation

    validate_sample_files(
        metadata
    )

    # Save split metadata

    metadata.to_csv(
        SPLIT_FILE,
        index=False,
    )

    # Save preprocessing configuration

    config = {
        "dataset_version": "v1",
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "total_samples": EXPECTED_SAMPLES,
        "train_samples": 630,
        "validation_samples": 135,
        "test_samples": 135,
        "grid_shape": [
            EXPECTED_N,
            EXPECTED_M,
        ],
        "data_symbols": EXPECTED_DATA_SYMBOLS,
        "channel_shape": [
            432,
            EXPECTED_DATA_SYMBOLS,
        ],
        "num_paths": 4,
        "input": "rx_dd",
        "target": "tx_dd",
        "channel_ground_truth": "h_dd",
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    # Print split summary

    print()
    print(
        "Dataset split"
    )
    print("-" * 70)

    split_counts = (
        metadata["split"]
        .value_counts()
    )

    print(
        f"Train samples           : "
        f"{split_counts['train']}"
    )

    print(
        f"Validation samples      : "
        f"{split_counts['validation']}"
    )

    print(
        f"Test samples            : "
        f"{split_counts['test']}"
    )

    print()
    print(
        "Files created"
    )
    print("-" * 70)

    print(
        f"Split metadata          : "
        f"{SPLIT_FILE}"
    )

    print(
        f"Configuration           : "
        f"{CONFIG_FILE}"
    )

    print()
    print("=" * 70)
    print(
        "DATASET PREPROCESSING COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    preprocess_dataset()