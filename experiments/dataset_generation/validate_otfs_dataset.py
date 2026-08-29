import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTFS_ROOT = PROJECT_ROOT / "Phy_Mod_OTFS"

sys.path.insert(0, str(OTFS_ROOT))


# Configuration

DATASET_ROOT = PROJECT_ROOT / "datasets" / "otfs"
RAW_DIR = DATASET_ROOT / "raw"
METADATA_FILE = (
    DATASET_ROOT
    / "metadata"
    / "dataset_v1_metadata.csv"
)

EXPECTED_SAMPLES = 900
EXPECTED_M = 32
EXPECTED_N = 16
EXPECTED_DATA_SYMBOLS = 368
EXPECTED_CHANNEL_ROWS = 432
EXPECTED_PATHS = 4

EXPECTED_SNR = [10, 15, 20]
EXPECTED_VELOCITIES = [30, 120, 500]


# Validation

def validate_dataset():

    print("=" * 70)
    print("OTFS DATASET V1 VALIDATION")
    print("=" * 70)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{RAW_DIR}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    sample_files = sorted(
        RAW_DIR.glob("sample_*.npz")
    )

    print(
        f"Sample files found      : "
        f"{len(sample_files)}"
    )

    assert (
        len(sample_files) == EXPECTED_SAMPLES
    ), "Incorrect number of sample files"

    metadata = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"Metadata rows           : "
        f"{len(metadata)}"
    )

    assert (
        len(metadata) == EXPECTED_SAMPLES
    ), "Incorrect number of metadata rows"

    print()

    # Metadata checks

    assert set(
        metadata["snr_db"].unique()
    ) == set(EXPECTED_SNR), "Invalid SNR values"

    assert set(
        metadata["velocity_kmh"].unique()
    ) == set(EXPECTED_VELOCITIES), (
        "Invalid velocity values"
    )

    counts = (
        metadata
        .groupby(
            ["snr_db", "velocity_kmh"]
        )
        .size()
    )

    print("Samples per condition")
    print("-" * 70)

    for snr in EXPECTED_SNR:

        for velocity in EXPECTED_VELOCITIES:

            count = counts.get(
                (snr, velocity),
                0,
            )

            print(
                f"SNR = {snr:>2} dB | "
                f"Velocity = {velocity:>3} km/h | "
                f"Samples = {count}"
            )

            assert (
                count == 100
            ), (
                f"Incorrect sample count for "
                f"SNR={snr}, velocity={velocity}"
            )

    print()

    # Sample validation

    print("Validating sample contents...")
    print("-" * 70)

    invalid_samples = 0

    for index, sample_file in enumerate(
        sample_files
    ):

        try:

            with np.load(
                sample_file,
                allow_pickle=False,
            ) as data:

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
                ), "Missing required arrays"

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
                    f"Invalid rx_dd shape: "
                    f"{rx_dd.shape}"
                )

                assert (
                    tx_dd.size
                    == EXPECTED_DATA_SYMBOLS
                ), (
                    f"Invalid tx_dd size: "
                    f"{tx_dd.size}"
                )

                assert h_dd.shape == (
                    EXPECTED_CHANNEL_ROWS,
                    EXPECTED_DATA_SYMBOLS,
                ), (
                    f"Invalid h_dd shape: "
                    f"{h_dd.shape}"
                )

                assert his.shape == (
                    EXPECTED_PATHS,
                ), (
                    f"Invalid path gains shape: "
                    f"{his.shape}"
                )

                assert lis.shape == (
                    EXPECTED_PATHS,
                ), (
                    f"Invalid delay indices shape: "
                    f"{lis.shape}"
                )

                assert kis.shape == (
                    EXPECTED_PATHS,
                ), (
                    f"Invalid Doppler indices shape: "
                    f"{kis.shape}"
                )

                # Numerical checks

                assert np.isfinite(
                    rx_dd
                ).all(), "rx_dd contains NaN/Inf"

                assert np.isfinite(
                    tx_dd
                ).all(), "tx_dd contains NaN/Inf"

                assert np.isfinite(
                    h_dd
                ).all(), "h_dd contains NaN/Inf"

                assert np.isfinite(
                    his
                ).all(), "his contains NaN/Inf"

                assert np.isfinite(
                    lis
                ).all(), "lis contains NaN/Inf"

                assert np.isfinite(
                    kis
                ).all(), "kis contains NaN/Inf"

                # QPSK check

                qpsk_values = np.array(
                    [
                        (-1 - 1j) / np.sqrt(2),
                        (-1 + 1j) / np.sqrt(2),
                        (1 - 1j) / np.sqrt(2),
                        (1 + 1j) / np.sqrt(2),
                    ]
                )

                distances = np.min(
                    np.abs(
                        tx_dd[:, None]
                        - qpsk_values[None, :]
                    ),
                    axis=1,
                )

                assert np.max(
                    distances
                ) < 1e-5, "Invalid QPSK symbols"

        except Exception as exc:

            invalid_samples += 1

            print(
                f"INVALID: "
                f"{sample_file.name} -> {exc}"
            )

        if (
            index + 1
        ) % 100 == 0:

            print(
                f"  checked "
                f"{index + 1}/"
                f"{len(sample_files)}"
            )

    assert (
        invalid_samples == 0
    ), (
        f"{invalid_samples} invalid samples found"
    )

    # Metadata/file consistency

    metadata_files = set(
        metadata["file"]
    )

    actual_files = {
        file.name
        for file in sample_files
    }

    assert (
        metadata_files == actual_files
    ), "Metadata and sample files do not match"

    print()
    print("=" * 70)
    print("DATASET VALIDATION PASSED")
    print("=" * 70)

    print(
        f"Total samples           : "
        f"{len(sample_files)}"
    )

    print(
        "Sample integrity        : PASS"
    )

    print(
        "Shape validation        : PASS"
    )

    print(
        "NaN / Inf validation    : PASS"
    )

    print(
        "QPSK validation         : PASS"
    )

    print(
        "Metadata validation     : PASS"
    )

    print(
        "Condition balance       : PASS"
    )

    print("=" * 70)


if __name__ == "__main__":
    validate_dataset()