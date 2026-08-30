import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTFS_ROOT = PROJECT_ROOT / "Phy_Mod_OTFS"

sys.path.insert(0, str(OTFS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config


CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_v1.yaml"


def validate_sample(config, sample_file: Path) -> None:
    expected_grid = config.representation.expected_grid_shape
    expected_channel = config.representation.expected_channel_shape

    expected_rx_shape = (
        expected_grid.rows,
        expected_grid.cols,
    )

    expected_data_symbols = (
        config.representation.expected_data_symbols
    )

    expected_channel_shape = (
        expected_channel.rows,
        expected_channel.cols,
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

    with np.load(
        sample_file,
        allow_pickle=False,
    ) as data:

        if not required_keys.issubset(data.files):
            missing = required_keys.difference(data.files)

            raise ValueError(
                f"Missing required arrays: {sorted(missing)}"
            )

        rx_dd = data["rx_dd"]
        tx_dd = data["tx_dd"]
        h_dd = data["h_dd"]

        his = data["his"]
        lis = data["lis"]
        kis = data["kis"]

        if rx_dd.shape != expected_rx_shape:
            raise ValueError(
                f"Invalid rx_dd shape: {rx_dd.shape}; "
                f"expected {expected_rx_shape}"
            )

        if tx_dd.shape != (
            expected_data_symbols,
        ):
            raise ValueError(
                f"Invalid tx_dd shape: {tx_dd.shape}; "
                f"expected {(expected_data_symbols,)}"
            )

        if h_dd.shape != expected_channel_shape:
            raise ValueError(
                f"Invalid h_dd shape: {h_dd.shape}; "
                f"expected {expected_channel_shape}"
            )

        expected_path_shape = (
            expected_num_paths,
        )

        if his.shape != expected_path_shape:
            raise ValueError(
                f"Invalid his shape: {his.shape}; "
                f"expected {expected_path_shape}"
            )

        if lis.shape != expected_path_shape:
            raise ValueError(
                f"Invalid lis shape: {lis.shape}; "
                f"expected {expected_path_shape}"
            )

        if kis.shape != expected_path_shape:
            raise ValueError(
                f"Invalid kis shape: {kis.shape}; "
                f"expected {expected_path_shape}"
            )

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
                    f"{name} contains NaN or Inf"
                )

        qpsk_values = np.array(
            [
                (-1 - 1j) / np.sqrt(2.0),
                (-1 + 1j) / np.sqrt(2.0),
                (1 - 1j) / np.sqrt(2.0),
                (1 + 1j) / np.sqrt(2.0),
            ],
            dtype=np.complex64,
        )

        distances = np.min(
            np.abs(
                tx_dd[:, None]
                - qpsk_values[None, :]
            ),
            axis=1,
        )

        if np.max(distances) >= 1e-5:
            raise ValueError(
                "tx_dd contains invalid QPSK symbols"
            )


def validate_metadata(
    config,
    metadata: pd.DataFrame,
    sample_files: list[Path],
) -> None:
    expected_samples = (
        config.dataset.expected_samples
    )

    if len(metadata) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} metadata rows; "
            f"found {len(metadata)}"
        )

    required_columns = {
        "sample_id",
        "file",
        "snr_db",
        "velocity_kmh",
        "kmax",
        "num_paths",
        "max_delay",
        "pilot_snr_db",
        "pilot_power",
        "noise_power",
        "M",
        "N",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: "
            f"{sorted(missing_columns)}"
        )

    expected_snr = set(
        config.channel.snr_db
    )

    expected_velocities = set(
        config.channel.velocity_kmh
    )

    actual_snr = set(
        metadata["snr_db"].unique()
    )

    actual_velocities = set(
        metadata["velocity_kmh"].unique()
    )

    if actual_snr != expected_snr:
        raise ValueError(
            f"Invalid SNR values: {sorted(actual_snr)}"
        )

    if actual_velocities != expected_velocities:
        raise ValueError(
            "Invalid velocity values: "
            f"{sorted(actual_velocities)}"
        )

    expected_frames = (
        config.dataset.frames_per_condition
    )

    counts = (
        metadata
        .groupby(
            ["snr_db", "velocity_kmh"]
        )
        .size()
    )

    for snr in config.channel.snr_db:
        for velocity in config.channel.velocity_kmh:

            count = counts.get(
                (snr, velocity),
                0,
            )

            if count != expected_frames:
                raise ValueError(
                    f"Expected {expected_frames} samples for "
                    f"SNR={snr}, velocity={velocity}; "
                    f"found {count}"
                )

    expected_sample_ids = set(
        range(expected_samples)
    )

    actual_sample_ids = set(
        metadata["sample_id"].astype(int)
    )

    if actual_sample_ids != expected_sample_ids:
        raise ValueError(
            "Metadata sample IDs are incomplete or invalid"
        )

    metadata_files = set(
        metadata["file"].astype(str)
    )

    actual_files = {
        sample_file.name
        for sample_file in sample_files
    }

    if metadata_files != actual_files:
        raise ValueError(
            "Metadata and sample files do not match"
        )


def validate_dataset(config) -> None:
    dataset_root = (
        PROJECT_ROOT / config.dataset.root
    )

    raw_dir = (
        dataset_root / config.dataset.raw_dir
    )

    metadata_file = (
        dataset_root
        / config.dataset.metadata_dir
        / "dataset_v1_metadata.csv"
    )

    expected_samples = (
        config.dataset.expected_samples
    )

    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{raw_dir}"
        )

    if not metadata_file.is_file():
        raise FileNotFoundError(
            f"Metadata file not found:\n{metadata_file}"
        )

    sample_files = sorted(
        raw_dir.glob("sample_*.npz")
    )

    if len(sample_files) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} sample files; "
            f"found {len(sample_files)}"
        )

    metadata = pd.read_csv(
        metadata_file
    )

    validate_metadata(
        config,
        metadata,
        sample_files,
    )

    print(
        f"Sample files found      : {len(sample_files)}"
    )

    print(
        f"Metadata rows           : {len(metadata)}"
    )

    print()
    print("Samples per condition")
    print("-" * 70)

    counts = (
        metadata
        .groupby(
            ["snr_db", "velocity_kmh"]
        )
        .size()
    )

    for snr in config.channel.snr_db:
        for velocity in config.channel.velocity_kmh:
            print(
                f"SNR = {snr:>2} dB | "
                f"Velocity = {velocity:>3} km/h | "
                f"Samples = "
                f"{counts[(snr, velocity)]}"
            )

    print()
    print("Validating sample contents...")
    print("-" * 70)

    for index, sample_file in enumerate(
        sample_files,
        start=1,
    ):

        try:
            validate_sample(
                config,
                sample_file,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Invalid sample: "
                f"{sample_file.name}: {exc}"
            ) from exc

        if index % 100 == 0:
            print(
                f"  checked "
                f"{index}/{len(sample_files)}"
            )

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


def main() -> None:
    config = load_config(
        CONFIG_PATH
    )

    print("=" * 70)
    print("OTFS DATASET V1 VALIDATION")
    print("=" * 70)

    validate_dataset(
        config
    )


if __name__ == "__main__":
    main()