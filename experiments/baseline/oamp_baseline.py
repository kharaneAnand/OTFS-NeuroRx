"""Evaluate analytical OAMP against MMSE on the fixed test split."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config
from receivers.mmse import build_data_observation, mmse_detect
from receivers.oamp import oamp_detect


CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_v1.yaml"
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "oamp"
RESULTS_PATH = OUTPUT_DIRECTORY / "plain_oamp_results.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "plain_oamp_summary.json"
SEED = 42
OAMP_ITERATIONS = 20


def compute_metrics(
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
    bit_errors = prediction_bits != target_bits
    symbol_errors = np.any(bit_errors, axis=-1)
    signal_power = np.sum(np.abs(target) ** 2)
    error_power = np.sum(np.abs(prediction - target) ** 2)
    return (
        float(np.count_nonzero(bit_errors) / bit_errors.size),
        float(np.count_nonzero(symbol_errors) / target.size),
        float(error_power / signal_power),
    )


def summarize(
    values: list[tuple[float, float, float]],
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = array.mean(axis=0)
    standard_deviation = array.std(axis=0, ddof=1)
    confidence_interval = 1.96 * standard_deviation / np.sqrt(len(values))
    return {
        "ber": float(mean[0]),
        "ser": float(mean[1]),
        "nmse": float(mean[2]),
        "ber_std": float(standard_deviation[0]),
        "ser_std": float(standard_deviation[1]),
        "nmse_std": float(standard_deviation[2]),
        "ber_ci95": float(confidence_interval[0]),
        "ser_ci95": float(confidence_interval[1]),
        "nmse_ci95": float(confidence_interval[2]),
    }


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
    test_metadata = metadata.loc[
        metadata["split"].astype(str).str.lower() == "test"
    ].reset_index(drop=True)
    expected_test_samples = int(
        config.dataset.expected_samples * config.split.test_ratio
    )
    if len(test_metadata) != expected_test_samples:
        raise ValueError(
            f"Expected {expected_test_samples} test samples; "
            f"found {len(test_metadata)}."
        )

    mmse_values = []
    oamp_values = []
    rows = []

    for sample_index, row in test_metadata.iterrows():
        with np.load(
            raw_dir / str(row["file"]),
            allow_pickle=False,
        ) as sample:
            rx_dd = sample["rx_dd"]
            h_hat = sample["h_hat"]
            tx_dd = sample["tx_dd"]

        noise_power = 10.0 ** (-float(row["snr_db"]) / 10.0)
        pilot_power = noise_power * 10.0 ** (
            float(config.pilot.pilot_snr_db) / 10.0
        )
        y_data = build_data_observation(
            rx_dd,
            config,
            pilot_power,
        )
        mmse_prediction = mmse_detect(
            y_data,
            h_hat,
            noise_power,
        )
        oamp_prediction = oamp_detect(
            y_data,
            h_hat,
            noise_power,
            iterations=OAMP_ITERATIONS,
        )
        mmse_metric = compute_metrics(mmse_prediction, tx_dd)
        oamp_metric = compute_metrics(oamp_prediction, tx_dd)
        mmse_values.append(mmse_metric)
        oamp_values.append(oamp_metric)
        rows.append(
            {
                "sample_index": sample_index,
                "snr_db": int(row["snr_db"]),
                "velocity_kmh": int(row["velocity_kmh"]),
                "mmse_ber": mmse_metric[0],
                "mmse_ser": mmse_metric[1],
                "mmse_nmse": mmse_metric[2],
                "oamp_ber": oamp_metric[0],
                "oamp_ser": oamp_metric[1],
                "oamp_nmse": oamp_metric[2],
            }
        )

    mmse_summary = summarize(mmse_values)
    oamp_summary = summarize(oamp_values)
    summary = {
        "seed": SEED,
        "test_samples": len(test_metadata),
        "oamp_iterations": OAMP_ITERATIONS,
        "noise_variance_source": "per-sample configured data SNR used for H_hat generation",
        "mmse": mmse_summary,
        "plain_oamp": oamp_summary,
        "oamp_beats_mmse_all_metrics": all(
            oamp_summary[name] < mmse_summary[name]
            for name in ("ber", "ser", "nmse")
        ),
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Per-sample results saved to: {RESULTS_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()