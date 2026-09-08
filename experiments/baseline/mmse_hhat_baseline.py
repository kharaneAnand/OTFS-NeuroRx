"""Evaluate a data-domain MMSE receiver using stored pilot-based H_hat."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTFS_ROOT = PROJECT_ROOT / "Phy_Mod_OTFS"

sys.path.insert(0, str(OTFS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config
from receivers.mmse import build_data_observation, mmse_detect


CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_v1.yaml"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "baseline" / "mmse_hhat_results.csv"


def qpsk_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, float]:
    prediction_bits = np.stack(
        (prediction.real >= 0, prediction.imag >= 0),
        axis=-1,
    )
    target_bits = np.stack(
        (target.real >= 0, target.imag >= 0),
        axis=-1,
    )
    bit_errors = np.count_nonzero(prediction_bits != target_bits)
    symbol_errors = np.count_nonzero(
        np.any(prediction_bits != target_bits, axis=-1)
    )
    total_bits = target_bits.size
    total_symbols = target.size
    signal_power = np.sum(np.abs(target) ** 2)
    error_power = np.sum(np.abs(prediction - target) ** 2)

    return (
        float(bit_errors / total_bits),
        float(symbol_errors / total_symbols),
        float(error_power / signal_power),
    )


def evaluate_sample(
    sample_path: Path,
    config,
    noise_power: float,
    pilot_power: float,
) -> tuple[float, float, float]:
    with np.load(sample_path, allow_pickle=False) as sample:
        rx_dd = sample["rx_dd"]
        h_hat = sample["h_hat"]
        tx_dd = sample["tx_dd"]

    y_data = build_data_observation(
        rx_dd,
        config,
        pilot_power,
    )

    estimate = mmse_detect(
        y_data,
        h_hat,
        noise_power,
    )

    return qpsk_metrics(estimate, tx_dd)


def main() -> None:
    config = load_config(CONFIG_PATH)
    dataset_root = PROJECT_ROOT / config.dataset.root
    raw_dir = dataset_root / config.dataset.raw_dir
    split_path = (
        dataset_root
        / config.dataset.processed_dir
        / config.dataset.split_metadata_file
    )
    metadata = pd.read_csv(split_path)
    metadata = metadata.loc[
        metadata["split"].astype(str).str.lower() == "test"
    ].copy()

    expected_test_samples = int(
        config.dataset.expected_samples
        * config.split.test_ratio
    )

    if len(metadata) != expected_test_samples:
        raise ValueError(
            f"Expected {expected_test_samples} test samples; "
            f"found {len(metadata)}."
        )

    rows = []

    for (snr_db, velocity_kmh), group in metadata.groupby(
        ["snr_db", "velocity_kmh"],
        sort=True,
    ):
        metrics = []
        noise_power = 10.0 ** (-float(snr_db) / 10.0)
        pilot_power = noise_power * 10.0 ** (
            float(config.pilot.pilot_snr_db) / 10.0
        )

        for filename in group["file"]:
            metrics.append(
                evaluate_sample(
                    raw_dir / str(filename),
                    config,
                    noise_power,
                    pilot_power,
                )
            )

        values = np.asarray(metrics)
        standard_deviation = values.std(axis=0, ddof=1)
        confidence_interval = (
            1.96
            * standard_deviation
            / np.sqrt(len(metrics))
        )
        rows.append(
            {
                "snr_db": int(snr_db),
                "velocity_kmh": int(velocity_kmh),
                "samples": len(metrics),
                "ber": float(values[:, 0].mean()),
                "ser": float(values[:, 1].mean()),
                "nmse": float(values[:, 2].mean()),
                "ber_std": float(standard_deviation[0]),
                "ser_std": float(standard_deviation[1]),
                "nmse_std": float(standard_deviation[2]),
                "ber_ci95": float(confidence_interval[0]),
                "ser_ci95": float(confidence_interval[1]),
                "nmse_ci95": float(confidence_interval[2]),
            }
        )

    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"MMSE results saved to: {OUTPUT_PATH}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()