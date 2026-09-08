"""Evaluate informed MMSE residual correction on the complete test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dnn.informed_residual_receiver import InformedResidualReceiver
from src.config.loader import load_config
from training.dnn.informed_residual_dataset import (
    build_informed_features,
    load_split_metadata,
)
from src.receivers.mmse import mmse_detect


def metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
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


def summarize(values: list[tuple[float, float, float]]) -> dict[str, float]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = Path(config.dataset.root)
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = (
        dataset_root
        / config.dataset.processed_dir
        / config.dataset.split_metadata_file
    )
    metadata = load_split_metadata(metadata_path, "test")
    expected_test_samples = int(
        config.dataset.expected_samples * config.split.test_ratio
    )
    if len(metadata) != expected_test_samples:
        raise ValueError(
            f"Expected {expected_test_samples} test samples; found {len(metadata)}."
        )

    output_symbols = int(config.representation.expected_data_symbols)
    input_features = 2 * output_symbols
    input_features += 2 * int(config.representation.expected_channel_shape.rows)
    input_features += 2 * int(config.representation.expected_channel_shape.rows)
    input_features += output_symbols + 1

    model = InformedResidualReceiver(
        input_features=input_features,
        output_symbols=output_symbols,
        hidden_dims=config.informed_residual.hidden_dims,
        dropout=float(config.informed_residual.dropout),
    ).to(device)
    checkpoint_path = (
        Path(config.informed_residual.output.directory)
        / config.informed_residual.output.checkpoint_file
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mmse_values = []
    informed_values = []
    rows = []
    inference_seconds = 0.0

    for sample_index, row in metadata.iterrows():
        with np.load(
            raw_dir / str(row["file"]),
            allow_pickle=False,
        ) as sample:
            rx_dd = sample["rx_dd"]
            h_hat = sample["h_hat"]
            tx_dd = sample["tx_dd"]

        noise_power = 10.0 ** (-float(row["snr_db"]) / 10.0)
        features, mmse_estimate = build_informed_features(
            rx_dd,
            h_hat,
            noise_power,
            config,
        )
        mmse_metric = metrics(mmse_estimate, tx_dd)

        feature_tensor = torch.from_numpy(features).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            correction = model(feature_tensor).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - start

        informed_estimate = mmse_estimate + correction
        informed_metric = metrics(informed_estimate, tx_dd)
        mmse_values.append(mmse_metric)
        informed_values.append(informed_metric)
        rows.append(
            {
                "sample_index": sample_index,
                "mmse_ber": mmse_metric[0],
                "mmse_ser": mmse_metric[1],
                "mmse_nmse": mmse_metric[2],
                "informed_ber": informed_metric[0],
                "informed_ser": informed_metric[1],
                "informed_nmse": informed_metric[2],
            }
        )

    mmse_summary = summarize(mmse_values)
    informed_summary = summarize(informed_values)
    improves_all_metrics = all(
        informed_summary[name] < mmse_summary[name]
        for name in ("ber", "ser", "nmse")
    )
    comparison = {
        "test_samples": len(metadata),
        "mmse": mmse_summary,
        "mmse_informed_residual": informed_summary,
        "inference_time_seconds": inference_seconds,
        "inference_time_per_sample_seconds": inference_seconds / len(metadata),
        "improves_ber_ser_nmse": improves_all_metrics,
        "success_criterion": (
            "Informed residual must improve BER, SER, and NMSE on all "
            "135 test samples without meaningful inference-time cost."
        ),
    }

    output_directory = Path(config.informed_residual.output.directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with (
        output_directory / config.informed_residual.output.evaluation_file
    ).open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    with (
        output_directory / config.informed_residual.output.per_sample_file
    ).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()