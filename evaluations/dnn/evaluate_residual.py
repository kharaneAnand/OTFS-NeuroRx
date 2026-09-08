"""Evaluate MMSE and MMSE-plus-residual on the complete test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dnn.mmse_residual_receiver import MMSEResidualReceiver
from src.config.loader import load_config
from training.dnn.residual_dataset import (
    MMSEResidualDataset,
    load_split_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the MMSE residual OTFS receiver."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics(
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
    bit_error_mask = prediction_bits != target_bits
    symbol_error_mask = np.any(bit_error_mask, axis=-1)
    signal_power = np.sum(np.abs(target) ** 2)
    error_power = np.sum(np.abs(prediction - target) ** 2)

    return (
        float(np.count_nonzero(bit_error_mask) / bit_error_mask.size),
        float(np.count_nonzero(symbol_error_mask) / target.size),
        float(error_power / signal_power),
    )


def summarize(values: list[tuple[float, float, float]]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    means = array.mean(axis=0)
    standard_deviation = array.std(axis=0, ddof=1)
    confidence_interval = 1.96 * standard_deviation / np.sqrt(len(values))

    return {
        "ber": float(means[0]),
        "ser": float(means[1]),
        "nmse": float(means[2]),
        "ber_std": float(standard_deviation[0]),
        "ser_std": float(standard_deviation[1]),
        "nmse_std": float(standard_deviation[2]),
        "ber_ci95": float(confidence_interval[0]),
        "ser_ci95": float(confidence_interval[1]),
        "nmse_ci95": float(confidence_interval[2]),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device()
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

    dataset = MMSEResidualDataset(metadata, raw_dir, config)
    loader = DataLoader(
        dataset,
        batch_size=int(config.residual.training.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = MMSEResidualReceiver(
        output_symbols=int(config.representation.expected_data_symbols),
        hidden_dims=config.residual.hidden_dims,
        dropout=float(config.residual.dropout),
    ).to(device)

    checkpoint_path = (
        Path(config.residual.output.directory)
        / config.residual.output.checkpoint_file
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mmse_values: list[tuple[float, float, float]] = []
    residual_values: list[tuple[float, float, float]] = []
    residual_inference_seconds = 0.0
    per_sample_rows = []
    sample_index = 0

    with torch.no_grad():
        for mmse_estimate, residual_target in loader:
            mmse_estimate = mmse_estimate.to(device, non_blocking=True)
            residual_target = residual_target.to(device, non_blocking=True)
            target = mmse_estimate + residual_target

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            correction = model(mmse_estimate)
            if device.type == "cuda":
                torch.cuda.synchronize()
            residual_inference_seconds += time.perf_counter() - start

            refined = mmse_estimate + correction

            for position in range(mmse_estimate.shape[0]):
                mmse_metric = metrics(
                    mmse_estimate[position].cpu().numpy(),
                    target[position].cpu().numpy(),
                )
                residual_metric = metrics(
                    refined[position].cpu().numpy(),
                    target[position].cpu().numpy(),
                )
                mmse_values.append(mmse_metric)
                residual_values.append(residual_metric)
                per_sample_rows.append(
                    {
                        "sample_index": sample_index,
                        "mmse_ber": mmse_metric[0],
                        "mmse_ser": mmse_metric[1],
                        "mmse_nmse": mmse_metric[2],
                        "residual_ber": residual_metric[0],
                        "residual_ser": residual_metric[1],
                        "residual_nmse": residual_metric[2],
                    }
                )
                sample_index += 1

    mmse_summary = summarize(mmse_values)
    residual_summary = summarize(residual_values)
    improves_all_metrics = all(
        residual_summary[name] < mmse_summary[name]
        for name in ("ber", "ser", "nmse")
    )

    comparison = {
        "test_samples": sample_index,
        "mmse": mmse_summary,
        "mmse_residual": residual_summary,
        "residual_inference_time_seconds": residual_inference_seconds,
        "residual_inference_time_per_sample_seconds": (
            residual_inference_seconds / sample_index
        ),
        "improves_ber_ser_nmse": improves_all_metrics,
        "success_criterion": (
            "Residual must improve BER, SER, and NMSE on all test samples "
            "with no meaningful inference-time cost."
        ),
    }

    output_directory = Path(config.residual.output.directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with (
        output_directory / config.residual.output.evaluation_file
    ).open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    import csv

    with (
        output_directory / config.residual.output.per_sample_file
    ).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(per_sample_rows[0]),
        )
        writer.writeheader()
        writer.writerows(per_sample_rows)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()